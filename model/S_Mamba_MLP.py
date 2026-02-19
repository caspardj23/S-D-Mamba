import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_EncDec import Encoder, EncoderLayer

from mamba_ssm import Mamba


class MLPEmbedding_inverted(nn.Module):
    """
    Non-linear temporal embedding for the inverted architecture.

    Replaces the single Linear(seq_len, d_model) with a 2-layer MLP:
        Linear(seq_len, d_model * expand) → GELU → Dropout → Linear(d_model * expand, d_model)

    This allows the embedding to learn non-linear temporal features — e.g., detecting
    rapid transitions (phoneme boundaries) vs. steady-state segments — which a single
    linear layer cannot distinguish.

    Input:  [B, L, N]  (batch, seq_len, num_variates)
    Output: [B, N, d_model]
    """

    def __init__(self, seq_len, d_model, expand=2, dropout=0.1):
        super(MLPEmbedding_inverted, self).__init__()
        hidden_dim = d_model * expand
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [B, L, N] → permute → [B, N, L] → MLP → [B, N, d_model]
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        return self.dropout(x)


class TemporalMambaBlock(nn.Module):
    """
    Mamba block that processes along the TIME dimension for each variate independently.

    Input:  [B, L, N]  (batch, seq_len, num_variates)
    Output: [B, L, N]
    """

    def __init__(self, n_variates, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(TemporalMambaBlock, self).__init__()
        self.mamba_fwd = Mamba(
            d_model=n_variates,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=n_variates,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(n_variates)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, N]"""
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        out = fwd + bwd
        out = self.norm(x + self.dropout(out))
        return out


class TemporalMambaEncoder(nn.Module):
    """Stack of TemporalMambaBlocks that process along the time axis."""

    def __init__(
        self, n_variates, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.1
    ):
        super(TemporalMambaEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TemporalMambaBlock(
                    n_variates=n_variates,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """x: [B, L, N] → [B, L, N]"""
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    """
    S-Mamba-MLP: Dual-Axis Mamba with Non-Linear Temporal Embedding.

    Identical to S_Mamba_Speech except the inverted embedding uses a 2-layer MLP
    instead of a single Linear layer:

        Linear(seq_len → d_model)              →  compressed, linear features only
        MLP(seq_len → d_model*2 → d_model)     →  non-linear temporal features

    This tests the hypothesis that the temporal compression bottleneck is partly
    due to the linearity of the embedding, not just the dimensionality reduction.

    Architecture:
        1. (Optional) Instance normalization
        2. Temporal Mamba encoder: [B, L, N] → [B, L, N]
        3. MLP inverted embedding: [B, L, N] → [B, N, d_model]    ← CHANGED
        4. Cross-variate Mamba encoder: [B, N, d_model] → [B, N, d_model]
        5. Linear projector: [B, N, d_model] → [B, pred_len, N]

    New hyperparameters (vs S_Mamba_Speech):
        - embed_mlp_expand: expansion factor for MLP hidden dim (default 2)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in

        # --- Temporal Mamba: processes along time axis ---
        temporal_e_layers = getattr(configs, "temporal_e_layers", 2)
        d_conv_temporal = getattr(configs, "d_conv_temporal", 4)
        expand_temporal = getattr(configs, "expand_temporal", 2)
        d_state_temporal = getattr(configs, "d_state", 32)

        self.temporal_encoder = TemporalMambaEncoder(
            n_variates=configs.enc_in,
            n_layers=temporal_e_layers,
            d_state=d_state_temporal,
            d_conv=d_conv_temporal,
            expand=expand_temporal,
            dropout=configs.dropout,
        )

        # --- MLP inverted embedding (non-linear temporal compression) ---
        embed_mlp_expand = getattr(configs, "embed_mlp_expand", 2)
        self.enc_embedding = MLPEmbedding_inverted(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            expand=embed_mlp_expand,
            dropout=configs.dropout,
        )
        self.class_strategy = configs.class_strategy

        # --- Cross-variate Mamba encoder ---
        d_conv_variate = getattr(configs, "d_conv_variate", 4)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=d_conv_variate,
                        expand=1,
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=d_conv_variate,
                        expand=1,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # --- Output projector ---
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[S_Mamba_MLP] Total params: {total:,}  Trainable: {trainable:,}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N]

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Step 1: Temporal Mamba — scan along time axis
        x_enc = self.temporal_encoder(x_enc)

        # Step 2: MLP inverted embedding — non-linear temporal compression
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Step 3: Cross-variate Mamba encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Step 4: Project to prediction length
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]

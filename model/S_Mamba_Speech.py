import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba


class TemporalMambaBlock(nn.Module):
    """
    Mamba block that processes along the TIME dimension for each variate independently.

    Input:  [B, L, N]  (batch, seq_len, num_variates)
    Output: [B, L, N]

    For each variate, the Mamba SSM scans over the L time steps, capturing
    local and long-range temporal dynamics that the inverted architecture discards.
    """

    def __init__(self, n_variates, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(TemporalMambaBlock, self).__init__()
        # Mamba operates on the feature dim of input; here each variate's time series
        # is treated as a 1D sequence with n_variates channels
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
        """
        x: [B, L, N]
        """
        # Bidirectional temporal Mamba: forward + backward (reversed) scan
        fwd = self.mamba_fwd(x)  # [B, L, N]
        bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])  # [B, L, N]
        out = fwd + bwd

        # Residual + LayerNorm
        out = self.norm(x + self.dropout(out))
        return out


class TemporalMambaEncoder(nn.Module):
    """
    Stack of TemporalMambaBlocks that process along the time axis.

    This module enriches the input time series with temporal patterns
    BEFORE the cross-variate (inverted) Mamba encoder processes it.
    """

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
        """
        x: [B, L, N]
        returns: [B, L, N]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    """
    S-Mamba-Speech: Dual-Axis Mamba for Speech Time Series Forecasting.

    Architecture:
        1. (Optional) Instance normalization (can be disabled for speech via use_norm=0)
        2. Temporal Mamba encoder: scans along the TIME axis [B, L, N] -> [B, L, N]
           - Captures phoneme-level temporal dynamics
           - Uses bidirectional Mamba (forward + reverse) for non-causal context
           - Configurable d_conv (wider for speech local correlations)
        3. Inverted embedding: [B, L, N] -> [B, N, d_model]
        4. Cross-variate Mamba encoder (from original S-D-Mamba): scans along VARIATE axis
           - Captures correlations between EMA sensors / mel bins
        5. Linear projector: [B, N, d_model] -> [B, N, pred_len] -> [B, pred_len, N]

    New hyperparameters (vs original S_Mamba):
        - temporal_e_layers: number of temporal Mamba layers (default 2)
        - d_conv_temporal: convolution width for temporal Mamba (default 4)
        - expand_temporal: expansion factor for temporal Mamba (default 2)
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

        # --- Cross-variate embedding (inverted: time -> feature dim) ---
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy

        # --- Cross-variate Mamba encoder: processes along variate axis ---
        d_conv_variate = getattr(
            configs, "d_conv_variate", 4
        )  # Direction 5: wider conv
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
        print(f"[S_Mamba_Speech] Total params: {total:,}  Trainable: {trainable:,}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, N]

        if self.use_norm:
            # Instance normalization (Non-stationary Transformer style)
            # For speech, consider setting use_norm=0 to preserve dynamics
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Step 1: Temporal Mamba — scan along time axis
        # [B, L, N] -> [B, L, N]
        x_enc = self.temporal_encoder(x_enc)

        # Step 2: Inverted embedding — each variate becomes a token
        # [B, L, N] -> [B, N, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Step 3: Cross-variate Mamba encoder
        # [B, N, d_model] -> [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Step 4: Project to prediction length
        # [B, N, d_model] -> [B, N, pred_len] -> [B, pred_len, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-normalization
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, N]

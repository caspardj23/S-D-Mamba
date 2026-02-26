"""
Transformer MAE: Masked Autoencoder with Transformer Encoder for EMA Pre-Training.

Replaces BiMamba with a standard Transformer encoder, which handles masked
reconstruction naturally — attention can look around masked blocks without
state corruption.

Architecture:
    Input [B, L, N] (batch, seq_len, num_variates)
      → Block-mask 40% of frames
      → Replace masked frames with learnable [MASK] token
      → Linear projection to d_model
      → Sinusoidal positional encoding
      → Transformer encoder (e_layers × TransformerEncoderLayer)
      → MLP decoder → [B, L, N] reconstruction
      → MSE loss on masked positions only

Usage:
    Pre-training:  model = Transformer_MAE.Model(configs)
    Fine-tuning:   model = Transformer_MAE.FinetuneModel(configs)

Drop-in replacement for S_Mamba_MAE — same interface, same masking strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal position awareness."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, L, D]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def generate_batch_block_mask(
    batch_size, seq_len, mask_ratio=0.4, block_size=8, device="cpu"
):
    """
    Generate independent block masks for each sample in a batch (vectorized).

    Same masking strategy as S_Mamba_MAE for fair comparison.

    Args:
        batch_size: Number of samples
        seq_len: Length of the sequence
        mask_ratio: Target fraction of frames to mask (default 0.4 = 40%)
        block_size: Size of each contiguous masked block
        device: torch device

    Returns:
        mask: [B, L] boolean tensor, True = masked (to be predicted)
    """
    num_blocks = seq_len // block_size
    n_to_mask = max(1, int(num_blocks * mask_ratio))

    noise = torch.rand(batch_size, num_blocks, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    block_mask = torch.zeros(batch_size, num_blocks, dtype=torch.bool, device=device)
    block_mask.scatter_(1, ids_shuffle[:, :n_to_mask], True)

    mask = block_mask.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, -1)

    if mask.shape[1] < seq_len:
        pad = torch.zeros(
            batch_size, seq_len - mask.shape[1], dtype=torch.bool, device=device
        )
        mask = torch.cat([mask, pad], dim=1)

    return mask


class TransformerEncoder(nn.Module):
    """
    Standard Transformer encoder stack.

    Uses PyTorch's nn.TransformerEncoderLayer for each layer.
    Input/Output: [B, L, D]
    """

    def __init__(
        self, d_model, n_layers=3, n_heads=4, d_ff=None, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable for pre-training)
        )
        self.layers = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: [B, L, D]
            src_key_padding_mask: optional [B, L] bool mask for padding

        Returns:
            [B, L, D]
        """
        x = self.layers(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class Model(nn.Module):
    """
    Transformer MAE: Masked Autoencoder with Transformer Encoder.

    Pre-training model that learns articulatory representations through
    masked frame prediction on EMA data. Attention can see around masked
    blocks, avoiding the state corruption problem of recurrent models.

    Configs:
        enc_in (int): Number of input variates (e.g., 48 for Haskins EMA)
        seq_len (int): Input sequence length
        d_model (int): Encoder hidden dimension (default 128)
        n_heads (int): Number of attention heads (default 4)
        e_layers (int): Number of encoder layers (default 3)
        d_ff (int): FFN hidden dimension (default d_model * 4)
        dropout (float): Dropout rate (default 0.2)
        mask_ratio (float): Fraction of frames to mask (default 0.4)
        block_size (int): Size of contiguous masked blocks (default 8)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 128)

        # MAE-specific config
        self.mask_ratio = getattr(configs, "mask_ratio", 0.4)
        self.block_size = getattr(configs, "block_size", 8)

        # Encoder config
        n_layers = getattr(configs, "e_layers", 3)
        n_heads = getattr(configs, "n_heads", 4)
        d_ff = getattr(configs, "d_ff", self.d_model * 4)
        dropout = getattr(configs, "dropout", 0.2)

        # --- Input projection: N variates → d_model ---
        self.input_proj = nn.Linear(self.enc_in, self.d_model)

        # --- Learnable [MASK] token ---
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=max(5000, self.seq_len + 100), dropout=dropout
        )

        # --- Transformer encoder ---
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # --- Decoder: lightweight MLP for reconstruction ---
        decoder_dim = self.d_model
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, self.enc_in),
        )

        # Initialize weights
        self._init_weights()
        self._print_param_count()

    def _init_weights(self):
        """Initialize weights following MAE/ViT convention."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        proj_params = sum(p.numel() for p in self.input_proj.parameters())
        print(
            f"[Transformer_MAE] Total: {total:,} | "
            f"Encoder: {encoder_params:,} | Decoder: {decoder_params:,} | "
            f"InputProj: {proj_params:,} | "
            f"mask_ratio={self.mask_ratio} block_size={self.block_size}"
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for MAE pre-training.

        Args:
            x_enc: [B, L, N] - Input EMA frames
            x_mark_enc, x_dec, x_mark_dec: ignored (interface compatibility)
            mask: [B, L] boolean tensor (optional). If None, generated automatically.

        Returns:
            dict with keys:
                'loss': scalar MSE loss on masked positions
                'pred': [B, L, N] full reconstruction
                'mask': [B, L] the mask used
                'target': [B, L, N] original input
        """
        B, L, N = x_enc.shape

        target = x_enc

        # Generate mask if not provided
        if mask is None:
            mask = generate_batch_block_mask(
                B, L, self.mask_ratio, self.block_size, device=x_enc.device
            )

        # Project input to d_model
        x = self.input_proj(x_enc)  # [B, L, d_model]

        # Replace masked positions with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, L, d_model]
        x = torch.where(mask_expanded, self.mask_token.expand(B, L, -1), x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Encode — attention sees all positions, visible tokens inform masked ones
        x = self.encoder(x)  # [B, L, d_model]

        # Decode to reconstruct
        pred = self.decoder(x)  # [B, L, N]

        # Compute loss on masked positions only
        mask_idx = mask.unsqueeze(-1).expand_as(pred)  # [B, L, N]
        pred_masked = pred[mask_idx].view(-1, N)
        target_masked = target[mask_idx].view(-1, N)
        loss = F.mse_loss(pred_masked, target_masked)

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "target": target,
        }

    def encode(self, x_enc):
        """
        Encode without masking — for extracting representations after pre-training.

        Args:
            x_enc: [B, L, N]

        Returns:
            [B, L, d_model] encoder hidden states
        """
        x = self.input_proj(x_enc)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x

    def get_encoder_state_dict(self):
        """Extract encoder weights for transfer to a forecasting model."""
        state = {}
        state["input_proj"] = self.input_proj.state_dict()
        state["pos_enc"] = {k: v for k, v in self.pos_enc.state_dict().items()}
        state["encoder"] = self.encoder.state_dict()
        state["mask_token"] = self.mask_token.data
        return state


class FinetuneModel(nn.Module):
    """
    Transformer MAE Fine-tune Model for Forecasting.

    Uses the pre-trained Transformer encoder and attaches a forecasting head.
    Supports three fine-tuning strategies:
        - 'full': Train everything (encoder + head) with differential LR
        - 'freeze': Freeze encoder, only train forecasting head
        - 'partial': Freeze all but last N encoder layers + head

    Architecture:
        Input [B, L, N]
          → Input projection → [B, L, d_model]
          → Positional encoding
          → Transformer encoder (pre-trained) → [B, L, d_model]
          → Forecasting head → [B, pred_len, N]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 128)
        self.use_norm = getattr(configs, "use_norm", 0)

        # Fine-tune strategy
        self.finetune_strategy = getattr(configs, "finetune_strategy", "full")
        self.unfreeze_layers = getattr(configs, "unfreeze_layers", 2)

        # Encoder config (must match pre-training)
        n_layers = getattr(configs, "e_layers", 3)
        n_heads = getattr(configs, "n_heads", 4)
        d_ff = getattr(configs, "d_ff", self.d_model * 4)
        dropout = getattr(configs, "dropout", 0.2)

        # --- Input projection (from pre-training) ---
        self.input_proj = nn.Linear(self.enc_in, self.d_model)

        # --- Positional encoding (from pre-training) ---
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=max(5000, self.seq_len + 100), dropout=dropout
        )

        # --- Transformer encoder (from pre-training) ---
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # --- Forecasting head ---
        self.temporal_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        # Learned temporal projection: [B, seq_len, d_model] → [B, pred_len, d_model]
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.enc_in),
        )

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[Transformer_MAE_Finetune] Total: {total:,} | "
            f"Trainable: {trainable:,} | strategy={self.finetune_strategy}"
        )

    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pre-trained Transformer MAE encoder weights.

        Supports both Transformer MAE and BiMamba MAE checkpoints.
        For BiMamba MAE, only the input_proj and pos_enc are transferred
        (encoder architectures differ).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if "encoder_state" in ckpt:
            encoder_state = ckpt["encoder_state"]
        elif "model_state_dict" in ckpt:
            full_state = ckpt["model_state_dict"]
            encoder_state = {
                "input_proj": {
                    k.replace("input_proj.", ""): v
                    for k, v in full_state.items()
                    if k.startswith("input_proj.")
                },
                "pos_enc": {
                    k.replace("pos_enc.", ""): v
                    for k, v in full_state.items()
                    if k.startswith("pos_enc.")
                },
                "encoder": {
                    k.replace("encoder.", ""): v
                    for k, v in full_state.items()
                    if k.startswith("encoder.")
                },
            }
        else:
            encoder_state = ckpt

        # Load input projection
        self.input_proj.load_state_dict(encoder_state["input_proj"])
        # Load positional encoding (non-buffer parameters)
        self.pos_enc.load_state_dict(encoder_state["pos_enc"], strict=False)

        # Try to load encoder — may fail if architecture differs (e.g., BiMamba ckpt)
        try:
            self.encoder.load_state_dict(encoder_state["encoder"])
            print(
                f"[Transformer_MAE_Finetune] Loaded full encoder from {checkpoint_path}"
            )
        except RuntimeError as e:
            print(
                f"[Transformer_MAE_Finetune] WARNING: Could not load encoder weights "
                f"(architecture mismatch). Only loaded input_proj + pos_enc."
            )
            print(f"  Error: {e}")

        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        """Freeze/unfreeze parameters based on fine-tuning strategy."""
        if self.finetune_strategy == "freeze":
            for param in self.input_proj.parameters():
                param.requires_grad = False
            for param in self.pos_enc.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[Transformer_MAE_Finetune] Froze encoder (freeze strategy)")

        elif self.finetune_strategy == "partial":
            for param in self.input_proj.parameters():
                param.requires_grad = False
            for param in self.pos_enc.parameters():
                param.requires_grad = False
            # nn.TransformerEncoder stores layers as .layers.layers (our wrapper)
            actual_layers = self.encoder.layers.layers
            n_total = len(actual_layers)
            for i, layer in enumerate(actual_layers):
                if i < n_total - self.unfreeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(
                f"[Transformer_MAE_Finetune] Partial freeze: "
                f"unfroze last {self.unfreeze_layers}/{n_total} encoder layers"
            )

        elif self.finetune_strategy == "full":
            print("[Transformer_MAE_Finetune] Full fine-tune (all params trainable)")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Transformer_MAE_Finetune] Trainable: {trainable:,} / {total:,}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Standard forecasting forward pass."""
        B, L, N = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        # Encode
        x = self.input_proj(x_enc)  # [B, L, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)  # [B, L, d_model]

        # Temporal pooling + projection
        x = self.temporal_pool(x)  # [B, L, d_model]
        x = self.temporal_proj(x.transpose(1, 2)).transpose(
            1, 2
        )  # [B, pred_len, d_model]
        dec_out = self.forecast_head(x)  # [B, pred_len, N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]

    def get_param_groups(self, lr_encoder=1e-5, lr_head=1e-4):
        """Get parameter groups with differential learning rates."""
        encoder_params = (
            list(self.input_proj.parameters())
            + list(self.pos_enc.parameters())
            + list(self.encoder.parameters())
        )
        head_params = (
            list(self.temporal_pool.parameters())
            + list(self.temporal_proj.parameters())
            + list(self.forecast_head.parameters())
        )
        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ]

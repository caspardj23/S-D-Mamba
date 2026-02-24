"""
S-Mamba Masked Autoencoder (MAE) for Self-Supervised Pre-Training on EMA Data.

Approach 3.1 from RESEARCH_STRATEGY.md:
  - Pre-training task: Masked Frame Prediction with block masking
  - Encoder: Bidirectional Mamba operating along the time axis
  - Decoder: Lightweight MLP that reconstructs masked frames
  - Loss: MSE on masked positions only

After pre-training, the encoder weights can be transferred to S_Mamba_Speech
(or a new forecasting model) for fine-tuning.

Architecture:
    Input [B, L, N] (batch, seq_len, num_variates)
      → Block-mask 40% of frames (chunks of block_size consecutive frames)
      → Replace masked frames with learnable [MASK] token
      → Positional encoding
      → Bidirectional Mamba encoder (n_layers)
      → MLP decoder head → [B, L, N] reconstruction
      → MSE loss on masked positions only

Usage:
    Pre-training:  model = S_Mamba_MAE.Model(configs)  with mode='pretrain'
    Fine-tuning:   Load encoder weights into S_Mamba_MAE_Finetune.Model(configs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from mamba_ssm import Mamba


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


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for temporal processing.

    Forward + backward Mamba scans with residual connection and layer norm.
    Input/Output: [B, L, D]
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN within the block (like a standard transformer block)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [B, L, D]"""
        # Bidirectional Mamba
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        out = fwd + bwd

        # Residual + LayerNorm
        x = self.norm(x + self.dropout(out))

        # FFN with residual
        x = self.ffn_norm(x + self.ffn(x))

        return x


class BiMambaEncoder(nn.Module):
    """
    Stack of BiMambaBlocks.

    Takes [B, L, D] → [B, L, D] with rich temporal representations.
    """

    def __init__(
        self, d_model, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def generate_block_mask(seq_len, mask_ratio=0.4, block_size=8, device="cpu"):
    """
    Generate a block mask for a single sequence.

    Randomly selects contiguous blocks to mask, targeting ~mask_ratio fraction.

    Args:
        seq_len: Length of the sequence
        mask_ratio: Target fraction of frames to mask (default 0.4 = 40%)
        block_size: Size of each contiguous masked block
        device: torch device

    Returns:
        mask: [L] boolean tensor, True = masked (to be predicted)
    """
    num_frames_to_mask = int(seq_len * mask_ratio)
    num_blocks = max(1, num_frames_to_mask // block_size)

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # Randomly place blocks
    for _ in range(num_blocks):
        # Ensure we don't go out of bounds
        max_start = seq_len - block_size
        if max_start <= 0:
            mask[:] = True
            break
        start = torch.randint(0, max_start + 1, (1,)).item()
        mask[start : start + block_size] = True

    return mask


def generate_batch_block_mask(
    batch_size, seq_len, mask_ratio=0.4, block_size=8, device="cpu"
):
    """
    Generate independent block masks for each sample in a batch.

    Returns:
        mask: [B, L] boolean tensor, True = masked
    """
    masks = torch.stack(
        [
            generate_block_mask(seq_len, mask_ratio, block_size, device)
            for _ in range(batch_size)
        ]
    )
    return masks


class Model(nn.Module):
    """
    S-Mamba MAE: Masked Autoencoder with Bidirectional Mamba Encoder.

    Pre-training model that learns articulatory representations through
    masked frame prediction on EMA data.

    Configs:
        enc_in (int): Number of input variates (e.g., 48 for Haskins EMA)
        seq_len (int): Input sequence length
        d_model (int): Encoder hidden dimension
        e_layers (int): Number of encoder Mamba layers
        d_state (int): SSM state dimension
        d_conv_temporal (int): Local convolution width
        expand_temporal (int): Block expansion factor
        dropout (float): Dropout rate
        mask_ratio (float): Fraction of frames to mask (default 0.4)
        block_size (int): Size of contiguous masked blocks (default 8)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 256)

        # MAE-specific config
        self.mask_ratio = getattr(configs, "mask_ratio", 0.4)
        self.block_size = getattr(configs, "block_size", 8)

        # Encoder config
        n_layers = getattr(configs, "e_layers", 4)
        d_state = getattr(configs, "d_state", 32)
        d_conv = getattr(configs, "d_conv_temporal", 4)
        expand = getattr(configs, "expand_temporal", 2)
        dropout = getattr(configs, "dropout", 0.1)

        # --- Input projection: N variates → d_model ---
        self.input_proj = nn.Linear(self.enc_in, self.d_model)

        # --- Learnable [MASK] token ---
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=max(5000, self.seq_len + 100), dropout=dropout
        )

        # --- Bidirectional Mamba encoder ---
        self.encoder = BiMambaEncoder(
            d_model=self.d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
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

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(
            f"[S_Mamba_MAE] Total: {total:,} | Encoder: {encoder_params:,} | "
            f"Decoder: {decoder_params:,} | mask_ratio={self.mask_ratio} "
            f"block_size={self.block_size}"
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for MAE pre-training.

        Args:
            x_enc: [B, L, N]  - Input EMA frames
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

        # Store original for computing loss
        target = x_enc.clone()

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

        # Encode
        x = self.encoder(x)  # [B, L, d_model]

        # Decode to reconstruct
        pred = self.decoder(x)  # [B, L, N]

        # Compute loss on masked positions only
        mask_float = mask.unsqueeze(-1).float()  # [B, L, 1]
        num_masked = mask_float.sum().clamp(min=1.0)

        loss = (((pred - target) ** 2) * mask_float).sum() / (num_masked * N)

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
    S-Mamba MAE Fine-tune Model for Forecasting.

    Uses the pre-trained BiMamba encoder from MAE and attaches a forecasting head.
    Supports three fine-tuning strategies:
        - 'full': Train everything (encoder + head) with differential LR
        - 'freeze': Freeze encoder, only train forecasting head
        - 'partial': Freeze all but last N encoder layers + head

    Architecture:
        Input [B, L, N]
          → Input projection → [B, L, d_model]
          → Positional encoding
          → BiMamba encoder (pre-trained) → [B, L, d_model]
          → Forecasting head → [B, pred_len, N]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 256)
        self.use_norm = getattr(configs, "use_norm", 0)

        # Fine-tune strategy
        self.finetune_strategy = getattr(configs, "finetune_strategy", "full")
        self.unfreeze_layers = getattr(configs, "unfreeze_layers", 2)

        # Encoder config (must match pre-training)
        n_layers = getattr(configs, "e_layers", 4)
        d_state = getattr(configs, "d_state", 32)
        d_conv = getattr(configs, "d_conv_temporal", 4)
        expand = getattr(configs, "expand_temporal", 2)
        dropout = getattr(configs, "dropout", 0.1)

        # --- Input projection (from pre-training) ---
        self.input_proj = nn.Linear(self.enc_in, self.d_model)

        # --- Positional encoding (from pre-training) ---
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=max(5000, self.seq_len + 100), dropout=dropout
        )

        # --- Encoder (from pre-training) ---
        self.encoder = BiMambaEncoder(
            d_model=self.d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # --- Forecasting head ---
        # Pool temporal dimension and project to prediction
        self.forecast_head = nn.Sequential(
            nn.Linear(self.d_model * self.seq_len, self.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(self.d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.pred_len * self.enc_in),
        )

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[S_Mamba_MAE_Finetune] Total: {total:,} | Trainable: {trainable:,} | "
            f"strategy={self.finetune_strategy}"
        )

    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pre-trained MAE encoder weights.

        Args:
            checkpoint_path: Path to the MAE checkpoint (full model or encoder-only dict)
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Handle both full model checkpoint and encoder-only state dict
        if "encoder_state" in ckpt:
            encoder_state = ckpt["encoder_state"]
        elif "model_state_dict" in ckpt:
            # Extract encoder weights from full model checkpoint
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

        # Load weights
        self.input_proj.load_state_dict(encoder_state["input_proj"])
        # Only load non-buffer parameters for pos_enc
        self.pos_enc.load_state_dict(encoder_state["pos_enc"], strict=False)
        self.encoder.load_state_dict(encoder_state["encoder"])

        print(
            f"[S_Mamba_MAE_Finetune] Loaded pre-trained encoder from {checkpoint_path}"
        )

        # Apply fine-tuning strategy
        self._apply_finetune_strategy()

    def _apply_finetune_strategy(self):
        """Freeze/unfreeze parameters based on fine-tuning strategy."""
        if self.finetune_strategy == "freeze":
            # Freeze everything except the forecasting head
            for param in self.input_proj.parameters():
                param.requires_grad = False
            for param in self.pos_enc.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[S_Mamba_MAE_Finetune] Froze encoder (freeze strategy)")

        elif self.finetune_strategy == "partial":
            # Freeze input_proj and pos_enc
            for param in self.input_proj.parameters():
                param.requires_grad = False
            for param in self.pos_enc.parameters():
                param.requires_grad = False
            # Freeze all encoder layers except the last `unfreeze_layers`
            n_total = len(self.encoder.layers)
            for i, layer in enumerate(self.encoder.layers):
                if i < n_total - self.unfreeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(
                f"[S_Mamba_MAE_Finetune] Partial freeze: "
                f"unfroze last {self.unfreeze_layers}/{n_total} encoder layers"
            )

        elif self.finetune_strategy == "full":
            # All parameters trainable (use differential LR in optimizer)
            print("[S_Mamba_MAE_Finetune] Full fine-tune (all params trainable)")

        # Print updated trainable count
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[S_Mamba_MAE_Finetune] Trainable: {trainable:,} / {total:,}")

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

        # Flatten temporal dimension and project to forecast
        x = x.reshape(B, -1)  # [B, L * d_model]
        dec_out = self.forecast_head(x)  # [B, pred_len * N]
        dec_out = dec_out.reshape(B, self.pred_len, N)  # [B, pred_len, N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]

    def get_param_groups(self, lr_encoder=1e-5, lr_head=1e-4):
        """
        Get parameter groups with differential learning rates.

        For 'full' fine-tuning: encoder uses lower LR, head uses higher LR.
        """
        encoder_params = (
            list(self.input_proj.parameters())
            + list(self.pos_enc.parameters())
            + list(self.encoder.parameters())
        )
        head_params = list(self.forecast_head.parameters())

        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ]

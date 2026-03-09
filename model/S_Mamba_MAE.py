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
from utils.losses import spectral_loss as _spectral_loss
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


def generate_batch_block_mask(
    batch_size, seq_len, mask_ratio=0.4, block_size=8, device="cpu"
):
    """
    Generate independent block masks for each sample in a batch (vectorized).

    Uses non-overlapping block placement for consistent effective mask ratio.
    Divides the sequence into num_blocks candidate slots, shuffles them per sample,
    and selects the first `n_to_mask` slots.

    Args:
        batch_size: Number of samples
        seq_len: Length of the sequence
        mask_ratio: Target fraction of frames to mask (default 0.4 = 40%)
        block_size: Size of each contiguous masked block
        device: torch device

    Returns:
        mask: [B, L] boolean tensor, True = masked (to be predicted)
    """
    # Number of non-overlapping block slots
    num_blocks = seq_len // block_size
    n_to_mask = max(1, int(num_blocks * mask_ratio))

    # Generate random permutations for all samples at once: [B, num_blocks]
    noise = torch.rand(batch_size, num_blocks, device=device)
    # argsort gives permutation indices; take first n_to_mask as masked slots
    ids_shuffle = torch.argsort(noise, dim=1)
    block_mask = torch.zeros(batch_size, num_blocks, dtype=torch.bool, device=device)
    block_mask.scatter_(1, ids_shuffle[:, :n_to_mask], True)

    # Expand each block slot to block_size frames: [B, num_blocks * block_size]
    mask = block_mask.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, -1)

    # Handle remainder frames (seq_len % block_size != 0) — leave unmasked
    if mask.shape[1] < seq_len:
        pad = torch.zeros(
            batch_size, seq_len - mask.shape[1], dtype=torch.bool, device=device
        )
        mask = torch.cat([mask, pad], dim=1)

    return mask


class NextPatchDecoder(nn.Module):
    """
    Next-patch prediction decoder for causal auxiliary pre-training objective.

    Given encoder output at a causal boundary (context window ending at
    position k), predicts the next P frames [k+1 .. k+P].

    Uses a small window of encoder states before the boundary (positions
    [k-W+1 .. k]) to aggregate context, then projects through an MLP
    to predict P future frames in variate space.

    Architecture:
        encoder_out[:, k-W+1:k+1, :] → mean pool → [B, d_model]
          → MLP → [B, P * enc_in] → reshape → [B, P, enc_in]
    """

    def __init__(self, d_model, enc_in, max_pred_len=96, context_window=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.enc_in = enc_in
        self.max_pred_len = max_pred_len
        self.context_window = context_window

        # Project aggregated context to future frames
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, max_pred_len * enc_in),
        )

    def forward(self, encoder_out, boundary_idx, pred_len):
        """
        Args:
            encoder_out: [B, L, d_model] — full encoder output
            boundary_idx: int — last context position (pred starts at boundary_idx+1)
            pred_len: int — number of future frames to predict

        Returns:
            [B, pred_len, enc_in] — predicted future frames
        """
        B = encoder_out.shape[0]
        # Aggregate context: mean pool over [boundary_idx - W + 1 .. boundary_idx]
        start = max(0, boundary_idx - self.context_window + 1)
        context = encoder_out[:, start:boundary_idx + 1, :]  # [B, W, d_model]
        context = context.mean(dim=1)  # [B, d_model]

        # Predict all max_pred_len frames, then truncate
        out = self.head(context)  # [B, max_pred_len * enc_in]
        out = out.view(B, self.max_pred_len, self.enc_in)  # [B, max_pred_len, enc_in]
        return out[:, :pred_len, :]  # [B, pred_len, enc_in]


class Model(nn.Module):
    """
    S-Mamba MAE: Masked Autoencoder with Bidirectional Mamba Encoder.

    Pre-training model that learns articulatory representations through
    masked frame prediction on EMA data.

    Supports two pre-training objectives:
      1. Masked reconstruction (standard MAE) — loss on masked positions
      2. Next-patch prediction (causal auxiliary) — predict future frames
         from a random causal boundary in the sequence

    Combined loss: alpha_mask * L_mask + beta_next * L_next

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
        alpha_mask (float): Weight for masked reconstruction loss (default 1.0)
        beta_next (float): Weight for next-patch prediction loss (default 0.0)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 256)

        # MAE-specific config
        self.mask_ratio = getattr(configs, "mask_ratio", 0.4)
        self.block_size = getattr(configs, "block_size", 8)

        # Loss weighting
        self.alpha_mask = getattr(configs, "alpha_mask", 1.0)
        self.beta_next = getattr(configs, "beta_next", 0.0)
        self.gamma_spectral = getattr(configs, "gamma_spectral", 0.0)

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

        # --- Next-patch prediction decoder (auxiliary causal objective) ---
        if self.beta_next > 0:
            # max_pred_len=96 covers the longest fine-tuning horizon
            self.next_patch_decoder = NextPatchDecoder(
                d_model=self.d_model,
                enc_in=self.enc_in,
                max_pred_len=96,
                context_window=16,
                dropout=dropout,
            )
        else:
            self.next_patch_decoder = None

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        np_params = sum(p.numel() for p in self.next_patch_decoder.parameters()) if self.next_patch_decoder else 0
        print(
            f"[S_Mamba_MAE] Total: {total:,} | Encoder: {encoder_params:,} | "
            f"Decoder: {decoder_params:,} | NextPatch: {np_params:,} | "
            f"mask_ratio={self.mask_ratio} block_size={self.block_size} "
            f"alpha={self.alpha_mask} beta={self.beta_next}"
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
                'loss': combined loss (alpha * mask_loss + beta * next_loss)
                'loss_mask': MSE loss on masked positions
                'loss_next': MSE loss on next-patch prediction (0 if disabled)
                'pred': [B, L, N] full reconstruction
                'mask': [B, L] the mask used
                'target': [B, L, N] original input
        """
        B, L, N = x_enc.shape

        # Use input directly as target (no in-place modification of x_enc)
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

        # Encode
        encoder_out = self.encoder(x)  # [B, L, d_model]

        # Decode to reconstruct
        pred = self.decoder(encoder_out)  # [B, L, N]

        # Compute loss on masked positions only (index directly, avoid full-tensor ops)
        mask_idx = mask.unsqueeze(-1).expand_as(pred)  # [B, L, N]
        pred_masked = pred[mask_idx].view(-1, N)  # [num_masked_frames, N]
        target_masked = target[mask_idx].view(-1, N)  # [num_masked_frames, N]
        loss_mask = F.mse_loss(pred_masked, target_masked)

        # --- Next-patch prediction (causal auxiliary objective) ---
        loss_next = torch.tensor(0.0, device=x_enc.device)
        if self.next_patch_decoder is not None and self.training:
            # Sample a random prediction length and causal boundary
            pred_len = np.random.choice([24, 48, 96])
            # Boundary: last context position, must leave room for pred_len target
            max_boundary = L - pred_len - 1
            min_boundary = L // 4  # at least 25% context
            if max_boundary > min_boundary:
                boundary = np.random.randint(min_boundary, max_boundary + 1)
                # Target: the ground truth frames after the boundary
                target_next = target[:, boundary + 1:boundary + 1 + pred_len, :]  # [B, pred_len, N]
                # Predict from encoder output at the boundary
                pred_next = self.next_patch_decoder(encoder_out, boundary, pred_len)  # [B, pred_len, N]
                loss_next = F.mse_loss(pred_next, target_next)

        # --- Spectral loss (frequency-domain regularisation) ---
        loss_spectral = torch.tensor(0.0, device=x_enc.device)
        if self.gamma_spectral > 0:
            loss_spectral = _spectral_loss(pred, target)

        # Combined loss
        loss = (self.alpha_mask * loss_mask
                + self.beta_next * loss_next
                + self.gamma_spectral * loss_spectral)

        return {
            "loss": loss,
            "loss_mask": loss_mask,
            "loss_next": loss_next,
            "loss_spectral": loss_spectral,
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
    S-Mamba MAE Fine-tune Model for Forecasting (mask-token approach).

    Uses the pre-trained BiMamba encoder from MAE and attaches a forecasting head.
    Resolves the sequence-length mismatch between pre-training and fine-tuning:
    the encoder always processes seq_len tokens (same as pre-training), with the
    last pred_len positions filled by learnable [MASK] tokens instead of real data.

    This makes fine-tuning structurally identical to MAE pre-training:
      - Pre-training:  encoder fills in randomly masked positions within seq_len
      - Fine-tuning:   encoder fills in the last pred_len positions (suffix mask)

    Supports three fine-tuning strategies:
        - 'full': Train everything (encoder + head) with differential LR
        - 'freeze': Freeze encoder, only train forecasting head
        - 'partial': Freeze all but last N encoder layers + head

    Architecture:
        Input [B, seq_len, N]
          → Split: context [B, seq_len-pred_len, N] | target [B, pred_len, N]
          → Input projection on context → [B, seq_len-pred_len, d_model]
          → Append pred_len [MASK] tokens → [B, seq_len, d_model]
          → Positional encoding (1..seq_len, same as pre-training)
          → BiMamba encoder (pre-trained) → [B, seq_len, d_model]
          → Extract last pred_len positions → [B, pred_len, d_model]
          → Forecast head MLP → [B, pred_len, N]
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

        # --- Learnable [MASK] token (loaded from pre-training) ---
        # During fine-tuning, the last pred_len positions are replaced with this
        # token, making the task structurally identical to MAE pre-training.
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # --- Forecasting head ---
        # Projects encoder output at masked (future) positions to variate predictions.
        # Operates per-timestep on [B, pred_len, d_model] → [B, pred_len, enc_in].
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

        # Load mask token if available (it was pre-trained with the encoder)
        if "mask_token" in encoder_state:
            self.mask_token.data.copy_(encoder_state["mask_token"])
            print("[S_Mamba_MAE_Finetune] Loaded pre-trained mask token")

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
        """
        Forecasting forward pass using mask-token approach.

        Maintains the same sequence length as pre-training by replacing the last
        pred_len frames with learnable [MASK] tokens. The encoder output at these
        masked positions is projected to variate predictions.

        Input x_enc: [B, seq_len, N] where:
          - x_enc[:, :input_len, :] = observed context (input_len = seq_len - pred_len)
          - x_enc[:, -pred_len:, :] = ground truth target (not seen by the encoder)
        """
        B, L, N = x_enc.shape
        input_len = L - self.pred_len  # context length (e.g., 336 for L=384, pred_len=48)

        if self.use_norm:
            # Normalize based on context portion only
            context_for_norm = x_enc[:, :input_len, :]
            means = context_for_norm.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(context_for_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = (x_enc - means) / stdev

        # Split: only the context portion is projected and seen by the encoder
        context = x_enc[:, :input_len, :]  # [B, input_len, N]

        # Project context to d_model
        x_context = self.input_proj(context)  # [B, input_len, d_model]

        # Create mask tokens for the prediction positions
        mask_tokens = self.mask_token.expand(B, self.pred_len, -1)  # [B, pred_len, d_model]

        # Concatenate: total length = seq_len (same as pre-training)
        x = torch.cat([x_context, mask_tokens], dim=1)  # [B, seq_len, d_model]

        # Positional encoding (positions 1..seq_len, same as pre-training)
        x = self.pos_enc(x)

        # Encode (same sequence length as pre-training)
        x = self.encoder(x)  # [B, seq_len, d_model]

        # Extract encoder output at the masked (future) positions
        x_future = x[:, -self.pred_len :, :]  # [B, pred_len, d_model]

        # Project to output variates
        dec_out = self.forecast_head(x_future)  # [B, pred_len, N]

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
        The mask_token is grouped with the encoder (it was pre-trained with it).
        """
        encoder_params = (
            list(self.input_proj.parameters())
            + list(self.pos_enc.parameters())
            + list(self.encoder.parameters())
            + [self.mask_token]
        )
        head_params = list(self.forecast_head.parameters())

        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ]

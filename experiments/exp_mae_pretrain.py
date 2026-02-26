"""
Experiment class for MAE Self-Supervised Pre-Training on EMA Data.

Implements the masked autoencoder pre-training loop:
  - Block masking of input frames
  - MSE loss on masked positions only
  - Saves encoder checkpoints for downstream fine-tuning
  - Monitors reconstruction quality on validation set
  - Logs mask ratio coverage statistics
"""

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class Exp_MAE_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_MAE_Pretrain, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
        )
        return model_optim

    def _select_scheduler(self, optimizer):
        """Cosine annealing scheduler for pre-training."""
        use_cosine = getattr(self.args, "use_cosine_scheduler", True)
        warmup_epochs = getattr(self.args, "warmup_epochs", 5)

        if use_cosine:
            # Linear warmup + cosine decay
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / max(
                        1, self.args.train_epochs - warmup_epochs
                    )
                    return 0.5 * (1.0 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return scheduler
        return None

    def vali(self, vali_data, vali_loader):
        """Validate reconstruction loss on validation set."""
        total_loss = []
        total_masked_ratio = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                if i >= 2000:
                    break

                batch_x = batch_x.float().to(self.device)

                # Forward pass (MAE model returns dict with loss)
                result = self.model(batch_x)

                loss = result["loss"]
                mask = result["mask"]

                total_loss.append(loss.item())
                total_masked_ratio.append(mask.float().mean().item())

        avg_loss = np.mean(total_loss)
        avg_mask_ratio = np.mean(total_masked_ratio)
        self.model.train()
        return avg_loss, avg_mask_ratio

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # Early stopping on validation loss
        patience = getattr(self.args, "patience", 10)
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        max_grad_norm = getattr(self.args, "max_grad_norm", 1.0)

        # Loss tracking for plotting
        all_iter_losses = []  # per-iteration training loss
        epoch_train_losses = []  # per-epoch mean training loss
        epoch_vali_losses = []  # per-epoch validation loss

        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x)
                        loss = result["loss"]

                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    result = self.model(batch_x)
                    loss = result["loss"]

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    model_optim.step()

                train_losses.append(loss.item())
                all_iter_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    avg_recent = np.mean(train_losses[-100:])
                    mask_pct = result["mask"].float().mean().item() * 100
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | "
                        f"loss: {loss.item():.6f} (avg: {avg_recent:.6f}) | "
                        f"mask: {mask_pct:.1f}%"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.1f}s")
                    iter_count = 0
                    time_now = time.time()

            epoch_duration = time.time() - epoch_time
            train_loss = np.mean(train_losses)
            vali_loss, vali_mask_ratio = self.vali(vali_data, vali_loader)

            epoch_train_losses.append(train_loss)
            epoch_vali_losses.append(vali_loss)

            lr = model_optim.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1} | cost: {epoch_duration:.1f}s | "
                f"Train Loss: {train_loss:.6f} | Vali Loss: {vali_loss:.6f} | "
                f"Vali Mask%: {vali_mask_ratio * 100:.1f}% | LR: {lr:.2e}"
            )

            # Save best model
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Save periodic checkpoints (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(path, f"checkpoint_epoch{epoch + 1}.pth")
                self._save_checkpoint(ckpt_path, epoch, train_loss, vali_loss)
                print(f"  Saved checkpoint: {ckpt_path}")

            # LR scheduling
            if scheduler is not None:
                scheduler.step()

        # Load best model and save final encoder state
        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        # Save encoder-only weights for downstream fine-tuning
        encoder_path = os.path.join(path, "encoder_checkpoint.pth")
        self._save_encoder_checkpoint(encoder_path)
        print(f"Saved encoder checkpoint: {encoder_path}")

        # Plot training loss curves — save to test_results_mae folder
        path_after_dataset = self.args.root_path.split("dataset/")[-1].rstrip("/")
        model_name = self.args.model
        plot_dir = f"./test_results_mae/{path_after_dataset}/{model_name}/{setting}/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self._plot_training_loss(
            all_iter_losses, epoch_train_losses, epoch_vali_losses, train_steps, plot_dir
        )

        return self.model

    def _plot_training_loss(
        self, iter_losses, epoch_train, epoch_vali, steps_per_epoch, save_dir
    ):
        """
        Plot training loss curves and save to checkpoint directory.

        Creates two subplots:
          1. Per-iteration training loss (with smoothed overlay)
          2. Per-epoch train vs validation loss
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # --- Subplot 1: Per-iteration loss ---
        iterations = np.arange(1, len(iter_losses) + 1)
        ax1.plot(
            iterations,
            iter_losses,
            alpha=0.3,
            color="steelblue",
            linewidth=0.5,
            label="Raw",
        )
        # Smoothed with exponential moving average
        if len(iter_losses) > 1:
            window = max(1, len(iter_losses) // 50)
            smoothed = np.convolve(iter_losses, np.ones(window) / window, mode="valid")
            ax1.plot(
                iterations[window - 1 :],
                smoothed,
                color="coral",
                linewidth=1.5,
                label=f"Smoothed (window={window})",
            )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss per Iteration")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: Per-epoch train vs vali ---
        epochs = np.arange(1, len(epoch_train) + 1)
        ax2.plot(
            epochs,
            epoch_train,
            marker="o",
            markersize=3,
            label="Train",
            color="steelblue",
        )
        ax2.plot(
            epochs,
            epoch_vali,
            marker="s",
            markersize=3,
            label="Validation",
            color="coral",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Train vs Validation Loss per Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("MAE Pre-Training Loss Curves", fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_loss_curves.pdf")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved training loss plot: {plot_path}")

        # Also save raw data for later analysis
        np.savez(
            os.path.join(save_dir, "training_loss_data.npz"),
            iter_losses=np.array(iter_losses),
            epoch_train_losses=np.array(epoch_train),
            epoch_vali_losses=np.array(epoch_vali),
            steps_per_epoch=steps_per_epoch,
        )

    def _save_checkpoint(self, path, epoch, train_loss, vali_loss):
        """Save full model checkpoint with metadata."""
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "train_loss": train_loss,
                "vali_loss": vali_loss,
                "args": vars(self.args),
            },
            path,
        )

    def _save_encoder_checkpoint(self, path):
        """Save encoder-only weights for transfer learning."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        encoder_state = model.get_encoder_state_dict()
        torch.save(
            {
                "encoder_state": encoder_state,
                "args": vars(self.args),
            },
            path,
        )

    def test(self, setting, test=0):
        """
        Evaluate pre-training quality on test set.

        Reports reconstruction MSE on masked and unmasked positions,
        and saves sample reconstructions with visualization plots.
        """
        test_data, test_loader = self._get_data(flag="test")

        if test:
            path = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(path))

        # Setup output directory
        path_after_dataset = self.args.root_path.split("dataset/")[-1].rstrip("/")
        model_name = self.args.model
        folder_path = f"./test_results_mae/{path_after_dataset}/{model_name}/{setting}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        total_masked_loss = []
        total_unmasked_loss = []
        total_full_loss = []

        # Collect predictions, targets, and masks for R2 computation
        all_preds = []
        all_targets = []
        all_masks = []

        N = self.args.enc_in  # number of variates

        self.model.eval()
        plots_saved = False

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                if i >= 10000:
                    break

                batch_x = batch_x.float().to(self.device)

                result = self.model(batch_x)
                pred = result["pred"]
                target = result["target"]
                mask = result["mask"]

                # MSE on masked positions (per-element, divide by N variates)
                mask_float = mask.unsqueeze(-1).float()  # [B, L, 1]
                num_masked = mask_float.sum().clamp(min=1.0)
                masked_mse = (
                    ((pred - target) ** 2 * mask_float).sum() / (num_masked * N)
                ).item()

                # MSE on unmasked positions (per-element, divide by N variates)
                unmask_float = (~mask).unsqueeze(-1).float()
                num_unmasked = unmask_float.sum().clamp(min=1.0)
                unmasked_mse = (
                    ((pred - target) ** 2 * unmask_float).sum() / (num_unmasked * N)
                ).item()

                # Full MSE
                full_mse = ((pred - target) ** 2).mean().item()

                total_masked_loss.append(masked_mse)
                total_unmasked_loss.append(unmasked_mse)
                total_full_loss.append(full_mse)

                # Store for R2 computation
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

                # --- Visualization plots ---
                if not plots_saved and i == 0:
                    pred_np = pred.cpu().numpy()
                    target_np = target.cpu().numpy()
                    mask_np = mask.cpu().numpy()

                    # Save raw arrays
                    np.save(os.path.join(folder_path, "sample_pred.npy"), pred_np)
                    np.save(os.path.join(folder_path, "sample_target.npy"), target_np)
                    np.save(os.path.join(folder_path, "sample_mask.npy"), mask_np)

                    # Select representative variates for plotting
                    if N == 48:
                        # Haskins EMA: sample across articulators
                        plot_indices = list(range(N))
                    elif N in (12, 24, 36):
                        plot_indices = list(range(N))
                    else:
                        plot_indices = list(range(min(N, 12)))

                    sample_idx = 0  # first sample in batch

                    # Plot 1: Per-variate reconstruction (GT vs Pred with mask shading)
                    self._plot_reconstruction(
                        target_np[sample_idx],
                        pred_np[sample_idx],
                        mask_np[sample_idx],
                        plot_indices[:8],  # first 8 variates for overview
                        os.path.join(folder_path, "reconstruction_overview.pdf"),
                    )

                    # Plot 2: Individual variate reconstructions (like original S_Mamba)
                    for plot_idx in plot_indices:
                        gt = target_np[sample_idx, :, plot_idx]
                        pd = pred_np[sample_idx, :, plot_idx]
                        visual(
                            gt,
                            pd,
                            os.path.join(folder_path, f"variate_{plot_idx}.pdf"),
                        )

                    # Plot 3: Mask pattern visualization
                    self._plot_mask_pattern(
                        mask_np[sample_idx],
                        os.path.join(folder_path, "mask_pattern.pdf"),
                    )

                    # Plot 4: Per-variate MSE heatmap
                    per_var_mse = (pred_np[sample_idx] - target_np[sample_idx]) ** 2
                    self._plot_error_heatmap(
                        per_var_mse,
                        mask_np[sample_idx],
                        os.path.join(folder_path, "error_heatmap.pdf"),
                    )

                    plots_saved = True
                    print(
                        f"  Saved {len(plot_indices)} variate plots + overview to {folder_path}"
                    )

        avg_masked = np.mean(total_masked_loss)
        avg_unmasked = np.mean(total_unmasked_loss)
        avg_full = np.mean(total_full_loss)

        # Compute R2 scores from collected predictions/targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)  # [B, L]

        # R2 on full reconstruction
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_full = 1 - ss_res / (ss_tot + 1e-8)

        # R2 on masked positions only
        mask_expanded = all_masks[:, :, np.newaxis]  # [B, L, 1]
        mask_bool = np.broadcast_to(mask_expanded, all_preds.shape).astype(bool)
        masked_preds = all_preds[mask_bool]
        masked_targets = all_targets[mask_bool]
        ss_res_masked = np.sum((masked_targets - masked_preds) ** 2)
        ss_tot_masked = np.sum((masked_targets - np.mean(masked_targets)) ** 2)
        r2_masked = 1 - ss_res_masked / (ss_tot_masked + 1e-8)

        print(
            f"MAE Test Results:\n"
            f"  Masked MSE:   {avg_masked:.6f}\n"
            f"  Unmasked MSE: {avg_unmasked:.6f}\n"
            f"  Full MSE:     {avg_full:.6f}\n"
            f"  R2 (full):    {r2_full:.6f}\n"
            f"  R2 (masked):  {r2_masked:.6f}"
        )

        # Save results
        folder_results = f"./results/{setting}/"
        if not os.path.exists(folder_results):
            os.makedirs(folder_results)

        np.save(
            os.path.join(folder_results, "mae_metrics.npy"),
            np.array([avg_masked, avg_unmasked, avg_full, r2_full, r2_masked]),
        )

        f = open("result_mae_pretrain.txt", "a")
        f.write(f"{setting}\n")
        f.write(
            f"  masked_mse: {avg_masked:.6f} | "
            f"unmasked_mse: {avg_unmasked:.6f} | "
            f"full_mse: {avg_full:.6f} | "
            f"r2_full: {r2_full:.6f} | "
            f"r2_masked: {r2_masked:.6f}\n\n"
        )
        f.close()

        return

    def _plot_reconstruction(self, target, pred, mask, variate_indices, save_path):
        """
        Plot reconstruction for multiple variates with masked regions shaded.

        Args:
            target: [L, N] ground truth
            pred: [L, N] reconstruction
            mask: [L] boolean mask
            variate_indices: list of variate indices to plot
            save_path: output PDF path
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_vars = len(variate_indices)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 2.5 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]

        for ax, idx in zip(axes, variate_indices):
            ax.plot(
                target[:, idx], label="Ground Truth", color="steelblue", linewidth=1.5
            )
            ax.plot(
                pred[:, idx],
                label="Reconstruction",
                color="coral",
                linewidth=1.5,
                alpha=0.8,
            )

            # Shade masked regions
            masked_regions = np.where(mask)[0]
            if len(masked_regions) > 0:
                # Find contiguous blocks
                diffs = np.diff(masked_regions)
                block_starts = [masked_regions[0]]
                block_ends = []
                for j, d in enumerate(diffs):
                    if d > 1:
                        block_ends.append(masked_regions[j])
                        block_starts.append(masked_regions[j + 1])
                block_ends.append(masked_regions[-1])

                for s, e in zip(block_starts, block_ends):
                    ax.axvspan(s, e, alpha=0.15, color="red", label=None)

            ax.set_ylabel(f"Var {idx}", fontsize=9)
            if idx == variate_indices[0]:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Frame", fontsize=10)
        fig.suptitle("MAE Reconstruction (red shading = masked regions)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def _plot_mask_pattern(self, mask, save_path):
        """
        Visualize the block mask pattern.

        Args:
            mask: [L] boolean mask
            save_path: output PDF path
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 1.5))
        ax.imshow(
            mask[np.newaxis, :].astype(float),
            aspect="auto",
            cmap="Reds",
            interpolation="nearest",
        )
        ax.set_xlabel("Frame")
        ax.set_yticks([])
        mask_pct = mask.mean() * 100
        ax.set_title(f"Block Mask Pattern ({mask_pct:.1f}% masked)", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    def _plot_error_heatmap(self, error, mask, save_path):
        """
        Plot squared error heatmap [L, N] with mask overlay.

        Args:
            error: [L, N] squared errors
            mask: [L] boolean mask
            save_path: output PDF path
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(
            error.T,
            aspect="auto",
            cmap="hot",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Squared Error")

        # Draw mask boundaries
        masked_frames = np.where(mask)[0]
        if len(masked_frames) > 0:
            diffs = np.diff(masked_frames)
            block_starts = [masked_frames[0]]
            for j, d in enumerate(diffs):
                if d > 1:
                    block_starts.append(masked_frames[j + 1])
            for s in block_starts:
                ax.axvline(x=s, color="cyan", linewidth=0.5, alpha=0.6)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Variate")
        ax.set_title("Reconstruction Error Heatmap (cyan lines = mask block starts)")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

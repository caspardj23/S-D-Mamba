"""
Probing experiment for evaluating MAE encoder representations.

Freezes the pre-trained encoder and trains a lightweight classifier on top
to test whether the learned representations encode speech-relevant structure
(phoneme identity, manner of articulation, speaker identity).

Three representation sources are compared:
  1. 'encoder': Pre-trained MAE encoder representations [B, L, d_model]
  2. 'random':  Randomly initialized encoder (same architecture, no pre-training)
  3. 'raw':     Raw EMA features directly [B, L, enc_in]

Two probe heads:
  - Linear: single Linear(d_in, num_classes) — tests linear separability
  - MLP: Linear → ReLU → Linear — tests with minimal nonlinearity

Metrics: per-frame accuracy, per-class F1, confusion matrix.
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_provider.data_loader import Dataset_Haskins_Probe
from model.S_Mamba_MAE import Model as MAEModel

warnings.filterwarnings("ignore")


class LinearProbe(nn.Module):
    def __init__(self, d_in, num_classes):
        super().__init__()
        self.head = nn.Linear(d_in, num_classes)

    def forward(self, x):
        # x: [B, L, d_in] → [B, L, num_classes]
        return self.head(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in, num_classes, hidden_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class Exp_Probe:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu}" if args.use_gpu else "cpu"
        )
        self.probe_task = getattr(args, "probe_task", "speaker")
        self.probe_type = getattr(args, "probe_type", "linear")  # 'linear' or 'mlp'

    def _build_encoder(self, pretrained_path=None):
        """Build MAE encoder and optionally load pre-trained weights."""
        encoder = MAEModel(self.args).to(self.device)

        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=self.device)

            if "encoder_state" in ckpt:
                # encoder_checkpoint.pth format (encoder-only state dict)
                state = ckpt["encoder_state"]
                encoder.input_proj.load_state_dict(state["input_proj"])
                encoder.pos_enc.load_state_dict(state["pos_enc"], strict=False)
                encoder.encoder.load_state_dict(state["encoder"])
                if "mask_token" in state:
                    encoder.mask_token.data.copy_(state["mask_token"])
                print(f"[Probe] Loaded encoder-only weights from {pretrained_path}")
            elif "model_state_dict" in ckpt:
                # Full model checkpoint format
                encoder.load_state_dict(ckpt["model_state_dict"])
                print(f"[Probe] Loaded full model weights from {pretrained_path}")
            else:
                # Direct state dict
                encoder.load_state_dict(ckpt)
                print(f"[Probe] Loaded state dict from {pretrained_path}")
        elif pretrained_path:
            print(f"[Probe] WARNING: checkpoint not found: {pretrained_path}")
            print("[Probe] Using random initialization.")

        # Freeze all encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        return encoder

    def _build_probe(self, d_in, num_classes):
        if self.probe_type == "mlp":
            return MLPProbe(d_in, num_classes, hidden_dim=128).to(self.device)
        else:
            return LinearProbe(d_in, num_classes).to(self.device)

    def _get_data(self, flag):
        """Build probe dataset and dataloader."""
        stride = getattr(self.args, "mae_stride", 80)

        dataset = Dataset_Haskins_Probe(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            scale=True,
            stride=stride,
            probe_task=self.probe_task,
        )

        shuffle = flag == "train"
        batch_size = self.args.batch_size if flag == "train" else 64
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataset, loader

    def _extract_representations(self, encoder, batch_x, mode="encoder"):
        """
        Extract representations from a batch.

        Args:
            encoder: MAEModel (frozen)
            batch_x: [B, L, N] raw EMA input
            mode: 'encoder' (pre-trained), 'random' (random init), or 'raw'

        Returns:
            [B, L, d_in] representations
        """
        if mode == "raw":
            return batch_x  # [B, L, enc_in]
        else:
            with torch.no_grad():
                return encoder.encode(batch_x)  # [B, L, d_model]

    def _train_probe(self, encoder, probe, train_loader, val_loader, mode, save_dir):
        """Train a single probe head."""
        optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.train_epochs, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        patience = getattr(self.args, "patience", 15)

        for epoch in range(self.args.train_epochs):
            probe.train()
            train_correct = 0
            train_total = 0
            train_loss_sum = 0.0

            for batch_x, batch_labels in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                reps = self._extract_representations(encoder, batch_x, mode)
                logits = probe(reps)  # [B, L, num_classes]

                # Flatten for cross-entropy
                B, L, C = logits.shape
                loss = criterion(logits.reshape(B * L, C), batch_labels.reshape(B * L))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                train_correct += (preds == batch_labels).sum().item()
                train_total += batch_labels.numel()
                train_loss_sum += loss.item() * B

            scheduler.step()
            train_acc = train_correct / max(train_total, 1)

            # Validation
            val_acc, val_loss = self._eval_probe(encoder, probe, val_loader, mode, criterion)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  [{mode}] Epoch {epoch+1:3d} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [{mode}] Early stopping at epoch {epoch+1}")
                    break

        # Restore best
        if best_state is not None:
            probe.load_state_dict(best_state)
        print(f"  [{mode}] Best val accuracy: {best_val_acc:.4f}")
        return best_val_acc

    def _eval_probe(self, encoder, probe, loader, mode, criterion=None):
        """Evaluate probe on a dataset split. Returns (accuracy, avg_loss)."""
        probe.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_x, batch_labels in loader:
                batch_x = batch_x.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                reps = self._extract_representations(encoder, batch_x, mode)
                logits = probe(reps)

                B, L, C = logits.shape
                loss = criterion(logits.reshape(B * L, C), batch_labels.reshape(B * L))
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_labels.append(batch_labels.cpu().numpy().reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / max(n_batches, 1)

        probe.train()
        return acc, avg_loss

    def _full_eval(self, encoder, probe, loader, mode, class_names):
        """Full evaluation with per-class metrics and confusion matrix."""
        probe.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_labels in loader:
                batch_x = batch_x.float().to(self.device)
                batch_labels = batch_labels.long().to(self.device)

                reps = self._extract_representations(encoder, batch_x, mode)
                logits = probe(reps)
                preds = logits.argmax(dim=-1)

                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_labels.append(batch_labels.cpu().numpy().reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # Only include classes that appear in the data
        present_classes = sorted(set(all_labels))
        present_names = [str(class_names[i]) for i in present_classes]

        report = classification_report(
            all_labels, all_preds,
            labels=present_classes,
            target_names=present_names,
            zero_division=0,
        )
        cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "report": report,
            "confusion_matrix": cm,
            "class_names": present_names,
        }

    def run(self, setting):
        """
        Run the full probing experiment.

        For each representation source (encoder, random, raw), trains a probe
        head and evaluates on the test set. Prints comparison table.
        """
        print(f"\n{'='*70}")
        print(f"PROBING EXPERIMENT: {self.probe_task} | probe_type: {self.probe_type}")
        print(f"{'='*70}")

        # Load data
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        num_classes = train_data.num_classes
        class_names = train_data.class_names
        enc_in = self.args.enc_in
        d_model = getattr(self.args, "d_model", 256)

        print(f"Classes ({num_classes}): {class_names}")
        print(f"Train: {len(train_data)} windows | Val: {len(val_data)} | Test: {len(test_data)}")

        # Save directory
        save_dir = os.path.join(
            "./probe_results",
            f"{self.probe_task}_{self.probe_type}_{setting}",
        )
        os.makedirs(save_dir, exist_ok=True)

        # Build encoders
        pretrain_ckpt = getattr(self.args, "pretrain_checkpoint", None)

        results = {}
        modes_and_dims = []

        # 1. Pre-trained encoder
        if pretrain_ckpt and os.path.exists(pretrain_ckpt):
            modes_and_dims.append(("encoder", d_model))
        else:
            print("[Probe] No pre-trained checkpoint — skipping 'encoder' mode.")

        # 2. Random encoder (same architecture, random weights)
        modes_and_dims.append(("random", d_model))

        # 3. Raw features (no encoder)
        modes_and_dims.append(("raw", enc_in))

        for mode, d_in in modes_and_dims:
            print(f"\n--- Mode: {mode} (d_in={d_in}) ---")
            t0 = time.time()

            # Build encoder
            if mode == "encoder":
                encoder = self._build_encoder(pretrain_ckpt)
            elif mode == "random":
                encoder = self._build_encoder(None)  # random init, frozen
            else:
                encoder = None  # raw mode, no encoder

            # Build fresh probe head
            probe = self._build_probe(d_in, num_classes)
            n_params = sum(p.numel() for p in probe.parameters())
            print(f"  Probe params: {n_params:,}")

            # Train
            best_val = self._train_probe(
                encoder, probe, train_loader, val_loader, mode, save_dir
            )

            # Test evaluation
            test_results = self._full_eval(
                encoder, probe, test_loader, mode, class_names
            )
            results[mode] = test_results
            elapsed = time.time() - t0

            print(f"\n  [{mode}] TEST RESULTS ({elapsed:.1f}s):")
            print(f"    Accuracy:    {test_results['accuracy']:.4f}")
            print(f"    F1 (macro):  {test_results['f1_macro']:.4f}")
            print(f"    F1 (weighted): {test_results['f1_weighted']:.4f}")
            print(f"\n  Classification Report:\n{test_results['report']}")

            # Clean up encoder to free GPU memory
            del encoder
            torch.cuda.empty_cache()

        # Summary comparison table
        print(f"\n{'='*70}")
        print(f"SUMMARY: {self.probe_task} classification | {self.probe_type} probe")
        print(f"{'='*70}")
        print(f"  {'Mode':<12} {'Accuracy':>10} {'F1 macro':>10} {'F1 weighted':>12}")
        print(f"  {'-'*46}")
        for mode, d_in in modes_and_dims:
            r = results[mode]
            print(
                f"  {mode:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
                f"{r['f1_weighted']:>12.4f}"
            )
        print()

        # Save results
        save_path = os.path.join(save_dir, "probe_results.npz")
        save_dict = {}
        for mode in results:
            for k, v in results[mode].items():
                if isinstance(v, (float, int, np.floating)):
                    save_dict[f"{mode}_{k}"] = v
                elif isinstance(v, np.ndarray):
                    save_dict[f"{mode}_{k}"] = v
        np.savez(save_path, **save_dict)
        print(f"Saved results to {save_path}")

        # Save summary text
        summary_path = os.path.join(save_dir, "probe_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Probe task: {self.probe_task}\n")
            f.write(f"Probe type: {self.probe_type}\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"Checkpoint: {pretrain_ckpt}\n\n")
            for mode, d_in in modes_and_dims:
                r = results[mode]
                f.write(f"--- {mode} (d_in={d_in}) ---\n")
                f.write(f"Accuracy:     {r['accuracy']:.4f}\n")
                f.write(f"F1 (macro):   {r['f1_macro']:.4f}\n")
                f.write(f"F1 (weighted): {r['f1_weighted']:.4f}\n")
                f.write(f"\n{r['report']}\n\n")
        print(f"Saved summary to {summary_path}")

        # Plot confusion matrices
        self._plot_confusion_matrices(results, modes_and_dims, save_dir)

        return results

    def _plot_confusion_matrices(self, results, modes_and_dims, save_dir):
        """Plot confusion matrices for all modes side by side."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_modes = len(modes_and_dims)
            fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
            if n_modes == 1:
                axes = [axes]

            for ax, (mode, d_in) in zip(axes, modes_and_dims):
                r = results[mode]
                cm = r["confusion_matrix"]
                names = r["class_names"]

                # Normalize rows
                cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

                im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"{mode}\nAcc={r['accuracy']:.3f} F1={r['f1_macro']:.3f}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")

                tick_marks = np.arange(len(names))
                fontsize = max(4, min(8, 80 // len(names)))
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fontsize)
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(names, fontsize=fontsize)

            plt.suptitle(
                f"Probe: {self.probe_task} ({self.probe_type})",
                fontsize=13,
            )
            plt.tight_layout()
            plot_path = os.path.join(save_dir, "confusion_matrices.pdf")
            plt.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved confusion matrices to {plot_path}")
        except Exception as e:
            print(f"Warning: could not plot confusion matrices: {e}")

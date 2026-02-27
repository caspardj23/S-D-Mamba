"""
Experiment class for Fine-Tuning a Pre-Trained MAE Encoder for Forecasting.

After MAE pre-training (exp_mae_pretrain.py), this experiment:
  1. Loads the pre-trained encoder weights
  2. Attaches a forecasting head
  3. Fine-tunes with configurable strategy (freeze / partial / full)
  4. Evaluates with speech-specific metrics (per-variate, per-step)

Supports differential learning rates: lower LR for encoder, higher for head.
"""

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from model.S_Mamba_MAE import FinetuneModel as MambaFinetuneModel
from model.Transformer_MAE import FinetuneModel as TransformerFinetuneModel
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class Exp_MAE_Finetune(Exp_Basic):
    def __init__(self, args):
        super(Exp_MAE_Finetune, self).__init__(args)

        # Load pre-trained encoder if specified
        pretrain_ckpt = getattr(args, "pretrain_checkpoint", None)
        if pretrain_ckpt and os.path.exists(pretrain_ckpt):
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.load_pretrained_encoder(pretrain_ckpt)
            print(f"Loaded pre-trained encoder from: {pretrain_ckpt}")
        elif pretrain_ckpt:
            print(f"WARNING: Pre-trained checkpoint not found: {pretrain_ckpt}")
            print("Training from scratch (no pre-training).")

    def _build_model(self):
        # Select FinetuneModel based on the model name
        model_name = getattr(self.args, "model", "S_Mamba_MAE_Finetune")
        if "Transformer" in model_name:
            model = TransformerFinetuneModel(self.args).float()
        else:
            model = MambaFinetuneModel(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        strategy = getattr(self.args, "finetune_strategy", "full")

        if strategy == "full" and hasattr(model, "get_param_groups"):
            # Differential LR: encoder gets lower LR
            lr_encoder = getattr(self.args, "lr_encoder", self.args.learning_rate * 0.1)
            lr_head = self.args.learning_rate
            param_groups = model.get_param_groups(
                lr_encoder=lr_encoder, lr_head=lr_head
            )
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            print(
                f"Using differential LR: encoder={lr_encoder:.2e}, head={lr_head:.2e}"
            )
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=1e-4,
            )
        return optimizer

    def _select_criterion(self):
        if self.args.loss == "L1":
            return nn.L1Loss()
        elif self.args.loss == "SmoothL1":
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                if i >= 5000:
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                    if batch_x_mark is not None
                    else None
                )
                batch_y_mark = (
                    batch_y_mark.float().to(self.device)
                    if batch_y_mark is not None
                    else None
                )

                # Decoder input (interface compatibility — not used by MAE finetune model)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        avg_loss = np.mean(total_loss)
        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        max_grad_norm = getattr(self.args, "max_grad_norm", 1.0)

        # Optional cosine scheduler
        use_cosine = getattr(self.args, "use_cosine_scheduler", False)
        if use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=self.args.train_epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        # Loss tracking for plotting
        all_iter_losses = []  # per-iteration training loss
        epoch_train_losses = []  # per-epoch mean training loss
        epoch_vali_losses = []  # per-epoch validation loss

        global_step = 0
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
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                    if batch_x_mark is not None
                    else None
                )
                batch_y_mark = (
                    batch_y_mark.float().to(self.device)
                    if batch_y_mark is not None
                    else None
                )

                # Decoder input (interface compatibility)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                        loss = criterion(outputs, batch_y)

                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, batch_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    model_optim.step()

                train_losses.append(loss.item())
                all_iter_losses.append(loss.item())
                global_step += 1

                # W&B per-iteration logging
                if self.use_wandb and (i + 1) % 10 == 0:
                    log_dict = {"train/iter_loss": loss.item()}
                    if (i + 1) % 100 == 0:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
                        log_dict["train/grad_norm"] = total_norm**0.5
                    self._wandb_log(log_dict, step=global_step)

                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            epoch_train_losses.append(train_loss)
            epoch_vali_losses.append(vali_loss)

            print(
                f"Epoch: {epoch + 1} cost: {epoch_duration:.1f}s | "
                f"Train: {train_loss:.7f} | Vali: {vali_loss:.7f} | Test: {test_loss:.7f}"
            )

            # W&B epoch-level logging
            self._wandb_log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/loss": vali_loss,
                    "test/loss": test_loss,
                    "train/lr": model_optim.param_groups[0]["lr"],
                    "train/epoch_time_s": epoch_duration,
                },
                step=global_step,
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None:
                scheduler.step()
                print(f"  LR: {[g['lr'] for g in model_optim.param_groups]}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        # Plot training loss curves — save to test_results_mae folder
        path_after_dataset = self.args.root_path.split("dataset/")[-1].rstrip("/")
        model_name = self.args.model
        plot_dir = f"./test_results_mae/{path_after_dataset}/{model_name}/{setting}/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self._plot_training_loss(
            all_iter_losses,
            epoch_train_losses,
            epoch_vali_losses,
            train_steps,
            plot_dir,
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
        # Smoothed with moving average
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

        plt.suptitle("MAE Fine-Tuning Loss Curves", fontsize=13)
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            ckpt_path = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        trues = []

        path_after_dataset = self.args.root_path.split("dataset/")[-1].rstrip("/")
        model_name = self.args.model
        folder_path = f"./test_results_mae/{path_after_dataset}/{model_name}/{setting}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                if i >= 20000:
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                    if batch_x_mark is not None
                    else None
                )
                batch_y_mark = (
                    batch_y_mark.float().to(self.device)
                    if batch_y_mark is not None
                    else None
                )

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                # Save plots
                if i % 2000 == 0 or i == 2000:
                    input_data = batch_x.detach().cpu().numpy()

                    if i % 2000 == 0:
                        if self.args.enc_in == 80:
                            plot_idx = 40  # for librivoxspeech
                        elif self.args.enc_in in (12, 24, 36, 116):
                            plot_idx = 3  # for mngu0
                        elif self.args.enc_in == 48:
                            plot_idx = 7  # for haskins ema 6
                        else:
                            plot_idx = 0  # default

                        gt = np.concatenate(
                            (input_data[0, :, plot_idx], true[0, :, plot_idx]), axis=0
                        )
                        pd_arr = np.concatenate(
                            (input_data[0, :, plot_idx], pred[0, :, plot_idx]), axis=0
                        )
                        visual(gt, pd_arr, os.path.join(folder_path, f"{i}.pdf"))

                    if i == 2000:
                        if self.args.enc_in == 80:
                            plot_indices = [1, 11, 21, 31, 41, 51, 61, 71, 78, 79]  # for librivoxspeech
                        elif self.args.enc_in == 36:
                            plot_indices = list(range(36))  # for mngu0
                        elif self.args.enc_in == 12:
                            plot_indices = list(range(12))  # for mngu0 first 12 features
                        elif self.args.enc_in == 24:
                            plot_indices = list(range(24))  # for mngu0 first 24 features
                        elif self.args.enc_in == 116:
                            plot_indices = list(range(36)) + list(range(36, 116, 10))  # for mngu0 first 36 ema + every 10th msg feature
                        elif self.args.enc_in == 48:
                            plot_indices = list(range(48))  # for haskins ema 6
                        else:
                            plot_indices = [0, 5, 10, 15, 20]  # default

                        for plot_idx in plot_indices:
                            gt = np.concatenate(
                                (input_data[0, :, plot_idx], true[0, :, plot_idx]), axis=0
                            )
                            pd_arr = np.concatenate(
                                (input_data[0, :, plot_idx], pred[0, :, plot_idx]), axis=0
                            )
                            visual(
                                gt,
                                pd_arr,
                                os.path.join(folder_path, f"{i}_{plot_idx}.pdf"),
                            )

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # Save results
        folder_results = f"./results/{setting}/"
        if not os.path.exists(folder_results):
            os.makedirs(folder_results)

        mae_val, mse_val, rmse_val, mape_val, mspe_val, r2_val = metric(preds, trues)
        print(f"mse:{mse_val}, mae:{mae_val}, rmse:{rmse_val}, r2:{r2_val}")

        # Per-variate and per-step metrics
        per_variate_mse = np.mean((preds - trues) ** 2, axis=(0, 1))
        per_variate_mae = np.mean(np.abs(preds - trues), axis=(0, 1))
        ss_res = np.sum((preds - trues) ** 2, axis=(0, 1))
        ss_tot = np.sum((trues - trues.mean(axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))
        per_variate_r2 = 1 - ss_res / (ss_tot + 1e-8)
        per_step_mse = np.mean((preds - trues) ** 2, axis=(0, 2))
        per_step_mae = np.mean(np.abs(preds - trues), axis=(0, 2))

        print(f"Per-variate MSE: {per_variate_mse}")
        print(f"Per-variate R2:  {per_variate_r2}")
        print(f"Per-step MSE (first 10): {per_step_mse[:10]}")
        print(f"Per-step MSE (last 10):  {per_step_mse[-10:]}")

        # W&B test metrics
        if self.use_wandb:
            test_metrics = {
                "test/mse": mse_val,
                "test/mae": mae_val,
                "test/rmse": rmse_val,
                "test/r2": r2_val,
            }
            for vi, v_mse in enumerate(per_variate_mse):
                test_metrics[f"test/variate_{vi}_mse"] = v_mse
            for vi, v_r2 in enumerate(per_variate_r2):
                test_metrics[f"test/variate_{vi}_r2"] = v_r2
            self._wandb_log(test_metrics)

        f = open("result_mae_finetune.txt", "a")
        f.write(f"{setting}\n")
        f.write(f"mse:{mse_val}, mae:{mae_val}, rmse:{rmse_val}, r2:{r2_val}\n")
        f.write(f"per_variate_mse: {per_variate_mse.tolist()}\n")
        f.write(f"per_variate_r2: {per_variate_r2.tolist()}\n")
        f.write(f"per_step_mse: {per_step_mse.tolist()}\n\n")
        f.close()

        np.save(
            os.path.join(folder_results, "metrics.npy"),
            np.array([mae_val, mse_val, rmse_val, mape_val, mspe_val, r2_val]),
        )
        np.save(os.path.join(folder_results, "pred.npy"), preds)
        np.save(os.path.join(folder_results, "true.npy"), trues)
        np.save(os.path.join(folder_results, "per_variate_mse.npy"), per_variate_mse)
        np.save(os.path.join(folder_results, "per_variate_r2.npy"), per_variate_r2)
        np.save(os.path.join(folder_results, "per_step_mse.npy"), per_step_mse)

        return

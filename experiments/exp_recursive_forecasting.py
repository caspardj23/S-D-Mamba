from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class Exp_Recursive_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Recursive_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "R2":
            from utils.metrics import R2Loss

            criterion = R2Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def test_recursive(self, setting):
        # Load test data with 'recursive' flag
        test_data, test_loader = self._get_data(flag="recursive")

        # Load trained model
        print("loading model")
        self.model.load_state_dict(
            torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
        )

        preds = []
        trues = []

        self.model.eval()

        # Recursive prediction parameters
        cycles = self.args.recursive_cycles
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len
        label_len = self.args.label_len

        print(
            f"Starting recursive prediction for {cycles} cycles of {pred_len} steps each."
        )

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                if i % self.args.stride != 0:
                    continue

                # i corresponds to the index in the dataset
                # We need to ensure we have enough future data in the dataset for 'cycles' predictions
                # Total needed future steps = cycles * pred_len

                # Check bounds
                if i + seq_len + cycles * pred_len > len(test_data.data_x):
                    break

                current_batch_x = batch_x.float().to(self.device)
                current_batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                    if batch_x_mark is not None
                    else None
                )

                # Store predictions for this sample
                sample_preds = []

                # Initial prediction loop
                for cycle in range(cycles):
                    # Prepare dec_inp
                    # Note: For S_Mamba and other non-autoregressive models in this repo,
                    # dec_inp is usually zeros for the pred_len part, concatenated with label_len part of history.

                    # We need the last 'label_len' from the CURRENT input history (current_batch_x)
                    # current_batch_x is [batch, seq_len, features]

                    dec_inp_start = current_batch_x[:, -label_len:, :]
                    dec_inp_zero = (
                        torch.zeros(
                            (
                                current_batch_x.shape[0],
                                pred_len,
                                current_batch_x.shape[2],
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    dec_inp = (
                        torch.cat([dec_inp_start, dec_inp_zero], dim=1)
                        .float()
                        .to(self.device)
                    )

                    # Prepare batch_y_mark for the NEXT pred_len steps
                    # We need to fetch this from the dataset based on current time index
                    # Current start index in dataset: i + cycle * pred_len
                    # The model input covers [start, start + seq_len]
                    # The prediction covers [start + seq_len, start + seq_len + pred_len]

                    # If we have time features
                    current_step_idx = i + cycle * pred_len
                    if test_data.timeenc != 0 and current_batch_x_mark is not None:
                        # We need mark for [current_step_idx, current_step_idx + seq_len + pred_len] (approx, depending on model usage)
                        # Usually batch_y_mark covers [label_start, pred_end]
                        # label_start = current_step_idx + seq_len - label_len
                        # pred_end = current_step_idx + seq_len + pred_len

                        r_begin = current_step_idx + seq_len - label_len
                        r_end = r_begin + label_len + pred_len

                        # Fetch from dataset
                        # Check bounds first (though outer check should suffice)
                        if r_end > len(test_data.data_stamp):
                            break

                        new_batch_y_mark = test_data.data_stamp[r_begin:r_end]
                        new_batch_y_mark = (
                            torch.from_numpy(new_batch_y_mark)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )  # Add batch dim

                        # Also need batch_x_mark for the current input
                        # s_begin = current_step_idx
                        # s_end = s_begin + seq_len
                        # We already have update logic for x, but marks are static in time
                        s_begin = current_step_idx
                        s_end = s_begin + seq_len
                        new_batch_x_mark = test_data.data_stamp[s_begin:s_end]
                        new_batch_x_mark = (
                            torch.from_numpy(new_batch_x_mark)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )

                        # Update current marks
                        current_batch_x_mark = new_batch_x_mark
                        current_batch_y_mark = new_batch_y_mark
                    else:
                        current_batch_x_mark = None
                        current_batch_y_mark = None

                    # Run Model
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(
                                    current_batch_x,
                                    current_batch_x_mark,
                                    dec_inp,
                                    current_batch_y_mark,
                                )[0]
                            else:
                                outputs = self.model(
                                    current_batch_x,
                                    current_batch_x_mark,
                                    dec_inp,
                                    current_batch_y_mark,
                                )
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                current_batch_x,
                                current_batch_x_mark,
                                dec_inp,
                                current_batch_y_mark,
                            )[0]
                        else:
                            outputs = self.model(
                                current_batch_x,
                                current_batch_x_mark,
                                dec_inp,
                                current_batch_y_mark,
                            )

                    f_dim = -1 if self.args.features == "MS" else 0
                    # outputs shape: [batch, pred_len, features]
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]

                    sample_preds.append(outputs.detach().cpu().numpy())

                    # Prepare for next cycle
                    # Shift x: remove old start, append new prediction
                    # current_batch_x: [batch, seq_len, features]
                    # outputs: [batch, pred_len, features]

                    # Assumption: pred_len <= seq_len.
                    if pred_len > seq_len:
                        # This naive rolling only works if we shift by pred_len.
                        # If pred_len > seq_len, we would lose more history than we add.
                        # But let's assume we want to advance by pred_len.
                        # New history = old history [pred_len:] + prediction
                        pass

                    outputs_tensor = outputs  # keep in tensor
                    new_x = torch.cat(
                        [current_batch_x[:, pred_len:, :], outputs_tensor], dim=1
                    )
                    current_batch_x = new_x

                # Flatten sample preds: [cycles, batch, pred_len, features] -> [cycles*pred_len, features]
                # Note batch=1
                sample_preds = np.array(sample_preds)  # [cycles, 1, pred_len, features]
                sample_preds = sample_preds.squeeze(1).reshape(
                    -1, sample_preds.shape[-1]
                )

                # Get Ground Truth
                # Start: index + seq_len
                # End: index + seq_len + cycles * pred_len
                gt_start = i + seq_len
                gt_end = gt_start + cycles * pred_len

                sample_trues = test_data.data_y[gt_start:gt_end]
                if self.args.features == "MS":
                    sample_trues = sample_trues[:, -1:]

                # Inverse Transform if needed
                if test_data.scale and self.args.inverse:
                    sample_preds = test_data.inverse_transform(sample_preds)
                    sample_trues = test_data.inverse_transform(sample_trues)

                preds.append(sample_preds)
                trues.append(sample_trues)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1} samples...")

        preds = np.array(preds)  # [num_samples, total_pred_len, features]
        trues = np.array(trues)
        print("recursive test shape:", preds.shape, trues.shape)

        # Result save
        folder_path = "./test_results_recursive/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Calculate metrics for the whole sequence
        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print("Total mse:{}, mae:{}, rmse:{}, r2:{}".format(mse, mae, rmse, r2))

        # Calculate metrics per cycle (to see error propagation)
        print("Metrics per cycle:")
        f = open(os.path.join(folder_path, "result_recursive_forecast.txt"), "a")
        f.write(setting + "  \n")
        f.write("Total mse:{}, mae:{}, rmse:{}, r2:{}\n".format(mse, mae, rmse, r2))

        cycle_metrics = []
        for c in range(cycles):
            # slice: [:, c*pred_len : (c+1)*pred_len, :]
            start_slice = c * pred_len
            end_slice = (c + 1) * pred_len
            p_slice = preds[:, start_slice:end_slice, :]
            t_slice = trues[:, start_slice:end_slice, :]

            c_mae, c_mse, c_rmse, c_mape, c_mspe, c_r2 = metric(p_slice, t_slice)
            print(f"Cycle {c}: mse:{c_mse}, mae:{c_mae}, rmse:{c_rmse}, r2:{c_r2}")
            f.write(f"Cycle {c}: mse:{c_mse}, mae:{c_mae}, rmse:{c_rmse}, r2:{c_r2}\n")
            cycle_metrics.append([c_mae, c_mse, c_rmse, c_mape, c_mspe, c_r2])

        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + "cycle_metrics.npy", np.array(cycle_metrics))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        # --- Visualizations ---
        print("Generating visualizations...")

        # 1. Error Propagation Plot
        metrics_np = np.array(
            cycle_metrics
        )  # Shape: (cycles, 5) -> mae, mse, rmse, mape, mspe
        cycles_range = np.arange(cycles)

        plt.figure(figsize=(10, 6))
        plt.plot(cycles_range, metrics_np[:, 0], marker="o", label="MAE")
        plt.plot(cycles_range, metrics_np[:, 1], marker="s", label="MSE")
        plt.title("Recursive Forecast Error Propagation")
        plt.xlabel("Cycle Index")
        plt.ylabel("Error Value")
        plt.ylim(0, 1.5)
        plt.xticks(cycles_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(folder_path, "error_propagation.pdf"))
        plt.close()

        # 2. Sample Predictions Plot
        # Plot up to 5 samples
        plot_indices = np.linspace(0, len(preds) - 1, num=5, dtype=int)

        # Use the last feature for visualization (target)
        f_idx = -1

        for idx in plot_indices:
            gt = trues[idx, :, f_idx]
            pd = preds[idx, :, f_idx]

            # Using the existing visual tool for consistency, but custom plot might be better for this specific need
            # Let's start with visual()
            visual(gt, pd, os.path.join(folder_path, f"sample_{idx}_target.pdf"))

            # Also create a more detailed plot showing cycle boundaries
            plt.figure(figsize=(15, 5))
            plt.plot(gt, label="Ground Truth", color="black", alpha=0.7)
            plt.plot(pd, label="Recursive Prediction", color="red", linestyle="--")

            # Add vertical lines for cycle boundaries
            for c in range(1, cycles):
                plt.axvline(x=c * pred_len, color="gray", linestyle=":", alpha=0.5)

            plt.title(f"Recursive Forecast vs Ground Truth (Sample {idx})")
            plt.legend()
            plt.savefig(os.path.join(folder_path, f"sample_{idx}_target_detailed.pdf"))
            plt.close()

        return

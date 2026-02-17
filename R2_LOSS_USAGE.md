# Using R-squared Loss Function

## Overview
The S-Mamba model now supports R-squared (R²) as an alternative loss function to MSE (Mean Squared Error). This is particularly useful for time series forecasting tasks where you want to maximize the explained variance.

## R-squared Loss Function
R² measures the proportion of variance in the dependent variable that is predictable from the independent variables:

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- SS_res = Sum of squared residuals (prediction errors)
- SS_tot = Total sum of squares (variance in the data)

The R² loss function is defined as:
```
Loss = 1 - R²
```

By minimizing this loss, we maximize R² (which ranges from -∞ to 1, where 1 is perfect prediction).

## Usage

To use R² loss instead of MSE, simply add the `--loss R2` argument when running your experiments:

### Training Example
```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_ema_norm_1000_96_6 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --e_layers 4 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --loss R2
```

### Testing Example
```bash
python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_ema_norm_1000_96_6 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --loss R2
```

## Available Loss Functions

- `MSE` (default): Mean Squared Error
- `R2`: R-squared loss (1 - R²)

## Notes

1. **Default Behavior**: If you don't specify `--loss`, the model will use MSE by default.
2. **Compatibility**: R² loss works with all existing models and configurations.
3. **Evaluation Metrics**: Regardless of the loss function used for training, the model will report all standard metrics (MAE, MSE, RMSE, MAPE, MSPE, R²) during testing.
4. **Use Case**: R² loss is particularly useful when:
   - You want to maximize explained variance
   - Your data has varying scales across different features
   - You want to focus on the overall fit rather than individual error magnitudes

## Example Scripts

See `scripts/multivariate_forecasting/Mngu0/S_Mamba_R2.sh` for a complete example of using R² loss with the Mngu0 dataset.

## Implementation Details

The R² loss is implemented in `utils/metrics.py` as a PyTorch module:

```python
class R2Loss(nn.Module):
    def forward(self, pred, true):
        ss_res = torch.sum((true - pred) ** 2)
        ss_tot = torch.sum((true - torch.mean(true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return 1 - r2  # Minimize (1 - R²) to maximize R²
```

**Note on R² Calculation:** The loss computes R² across all dimensions (batch, time, features) to produce a single scalar loss value for the optimizer. This is the standard approach for batch-based training, where we want to maximize the overall explained variance across the entire batch. The small epsilon (1e-8) prevents division by zero when the total variance is very small.

The loss function selection is handled in the base experiment class (`exp_basic.py`) through the `_select_criterion()` method, which is inherited by all experiment classes.

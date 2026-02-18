export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# ABLATION STUDY: S_Mamba_Speech on Mngu0 EMA, Horizon=48
#
# This script runs ablations to isolate the effect of each modification.
# All experiments use pred_len=48 (the horizon where S_Mamba starts degrading).
#
# Ablations:
#   A) Baseline S_Mamba (original, use_norm=1, d_conv=2)
#   B) S_Mamba baseline + use_norm=0 (Direction 5 only)
#   C) S_Mamba_Speech with use_norm=1 (Direction 1 only)
#   D) S_Mamba_Speech with use_norm=0 (Direction 1+5, full model)
#   E) S_Mamba_Speech with use_norm=0, d_conv_temporal=8 (wider temporal conv)
# ============================================================================

# --- A) Baseline S_Mamba (original) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id ablation_A_baseline_48 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 1 \
  --loss MSE \
  --itr 1 \
  --per_variate_scoring

# --- B) S_Mamba baseline + use_norm=0 (Direction 5 partial) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id ablation_B_nonorm_48 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 0 \
  --loss MSE \
  --itr 1 \
  --per_variate_scoring

# --- C) S_Mamba_Speech with use_norm=1 (Direction 1 only) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id ablation_C_temporal_norm_48 \
  --model S_Mamba_Speech \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 32 \
  --temporal_e_layers 2 \
  --d_conv_temporal 4 \
  --d_conv_variate 4 \
  --expand_temporal 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 1 \
  --loss MSE \
  --exp_name speech \
  --itr 1 \
  --per_variate_scoring

# --- D) S_Mamba_Speech with use_norm=0 (Direction 1+5, full model) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id ablation_D_temporal_nonorm_48 \
  --model S_Mamba_Speech \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 32 \
  --temporal_e_layers 2 \
  --d_conv_temporal 4 \
  --d_conv_variate 4 \
  --expand_temporal 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 0 \
  --loss MSE \
  --exp_name speech \
  --itr 1 \
  --per_variate_scoring

# --- E) S_Mamba_Speech with use_norm=0, d_conv_temporal=8 (wider conv) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id ablation_E_temporal_nonorm_wideconv_48 \
  --model S_Mamba_Speech \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 32 \
  --temporal_e_layers 2 \
  --d_conv_temporal 8 \
  --d_conv_variate 4 \
  --expand_temporal 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 0 \
  --loss MSE \
  --exp_name speech \
  --itr 1 \
  --per_variate_scoring

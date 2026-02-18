export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba_Speech

# ============================================================================
# S_Mamba_Speech on Mngu0 EMA (36 variates, 200Hz)
# Direction 1: Dual-Axis Mamba (temporal + cross-variate)
# Direction 5: use_norm=0, d_conv=4, expand_temporal=2
# ============================================================================

# --- Horizon 6 (30ms) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_speech_96_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
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

# --- Horizon 12 (60ms) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_speech_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
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

# --- Horizon 48 (240ms) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_speech_96_48 \
  --model $model_name \
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

# --- Horizon 96 (480ms) ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_speech_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
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

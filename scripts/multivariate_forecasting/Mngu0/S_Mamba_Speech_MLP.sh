export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba_Speech_MLP

# ============================================================================
# S_Mamba_MLP on Mngu0 EMA (36 variates)
# Config D + MLP embedding: Temporal Mamba + use_norm=0 + MLP(seq_len → d_model)
# Tests whether non-linear temporal compression improves over linear
# ============================================================================

# # --- Horizon 6 ---
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/mngu0/ \
#   --data_path ema_norm_1000.csv \
#   --model_id mngu0_mlp_96_6 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 6 \
#   --e_layers 2 \
#   --enc_in 36 \
#   --dec_in 36 \
#   --c_out 36 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --d_state 32 \
#   --temporal_e_layers 2 \
#   --d_conv_temporal 4 \
#   --d_conv_variate 4 \
#   --expand_temporal 2 \
#   --embed_mlp_expand 2 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --use_norm 0 \
#   --loss MSE \
#   --exp_name speech \
#   --itr 1 \
#   --per_variate_scoring

# # --- Horizon 12 ---
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/mngu0/ \
#   --data_path ema_norm_1000.csv \
#   --model_id mngu0_mlp_96_12 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --enc_in 36 \
#   --dec_in 36 \
#   --c_out 36 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --d_state 32 \
#   --temporal_e_layers 2 \
#   --d_conv_temporal 4 \
#   --d_conv_variate 4 \
#   --expand_temporal 2 \
#   --embed_mlp_expand 2 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --use_norm 0 \
#   --loss MSE \
#   --exp_name speech \
#   --itr 1 \
#   --per_variate_scoring

# --- Horizon 48 ---
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_mlp_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 32 \
  --temporal_e_layers 2 \
  --d_conv_temporal 4 \
  --d_conv_variate 4 \
  --expand_temporal 2 \
  --embed_mlp_expand 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_norm 0 \
  --loss MSE \
  --exp_name speech \
  --itr 1 \
  --per_variate_scoring

# # --- Horizon 96 ---
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/mngu0/ \
#   --data_path ema_norm_1000.csv \
#   --model_id mngu0_mlp_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 36 \
#   --dec_in 36 \
#   --c_out 36 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --d_state 32 \
#   --temporal_e_layers 2 \
#   --d_conv_temporal 4 \
#   --d_conv_variate 4 \
#   --expand_temporal 2 \
#   --embed_mlp_expand 2 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --use_norm 0 \
#   --loss MSE \
#   --exp_name speech \
#   --itr 1 \
#   --per_variate_scoring

# # --- Horizon 192 ---
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/mngu0/ \
#   --data_path ema_norm_1000.csv \
#   --model_id mngu0_mlp_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --enc_in 36 \
#   --dec_in 36 \
#   --c_out 36 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --d_state 32 \
#   --temporal_e_layers 2 \
#   --d_conv_temporal 4 \
#   --d_conv_variate 4 \
#   --expand_temporal 2 \
#   --embed_mlp_expand 2 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --use_norm 0 \
#   --loss MSE \
#   --exp_name speech \
#   --itr 1 \
#   --per_variate_scoring

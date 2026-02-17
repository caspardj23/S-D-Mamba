export CUDA_VISIBLE_DEVICES=0
# Example script for using R2 loss instead of MSE loss
# To use R2 loss, add "--loss R2" to the run.py arguments

model_name=S_Mamba

# Training with R2 loss
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_ema_norm_1000_96_6_r2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
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
  --itr 1 \
  --per_variate_scoring \
  --loss R2

# Testing with R2 loss
python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1000.csv \
  --model_id mngu0_ema_norm_1000_96_12_r2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
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
  --itr 1 \
  --per_variate_scoring \
  --loss R2

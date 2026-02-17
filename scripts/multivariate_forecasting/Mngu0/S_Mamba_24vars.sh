export CUDA_VISIBLE_DEVICES=0
# """Copy from traffic forecasting."""

model_name=S_Mamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100_24vars.csv \
  --model_id mngu0_ema_norm_100_24vars_96_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --e_layers 4 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --target 23 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100_24vars.csv \
  --model_id mngu0_ema_norm_100_24vars_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --target 23 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100_24vars.csv \
  --model_id mngu0_ema_norm_100_24vars_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 4 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --target 23 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100_24vars.csv \
  --model_id mngu0_ema_norm_100_24vars_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --target 23 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1
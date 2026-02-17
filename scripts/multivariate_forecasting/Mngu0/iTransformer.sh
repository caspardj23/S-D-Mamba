export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_6 \
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
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_12 \
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
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_48 \
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
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_96 \
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
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 512 \
  --itr 1
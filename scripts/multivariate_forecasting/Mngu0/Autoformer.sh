export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_100.csv \
  --model_id mngu0_ema_norm_100_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --target 35 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3
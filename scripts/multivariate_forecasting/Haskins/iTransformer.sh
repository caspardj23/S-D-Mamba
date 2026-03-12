export CUDA_VISIBLE_DEVICES=0
# Haskins EMA forecasting — sentence-aware data (ema_8_pos_xz.csv)
# 16 variates (posX + posZ), 100 Hz, sentence-level train/val/test split.

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz.csv \
  --model_id haskins_ema_7_pos_xz_96_6 \
  --model $model_name \
  --data haskins_forecast \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --e_layers 2 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --target 15 \
  --des 'Exp' \
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5 \
  --batch_size 64 \
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz.csv \
  --model_id haskins_ema_7_pos_xz_96_12 \
  --model $model_name \
  --data haskins_forecast \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --target 15 \
  --des 'Exp' \
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5 \
  --batch_size 64 \
  --d_ff 256 \
  --itr 1

# python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_7_pos_xz.csv \
#   --model_id haskins_ema_7_pos_xz_96_24 \
#   --model $model_name \
#   --data haskins_forecast \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --enc_in 16 \
#   --dec_in 16 \
#   --c_out 16 \
#   --target 15 \
#   --des 'Exp' \
#   --d_model 256 \
#   --learning_rate 0.0005 \
#   --train_epochs 5 \
#   --batch_size 64 \
#   --d_ff 256 \
#   --itr 1 \
#   --per_variate_scoring

# python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_7_pos_xz.csv \
#   --model_id haskins_ema_7_pos_xz_96_48 \
#   --model $model_name \
#   --data haskins_forecast \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --enc_in 16 \
#   --dec_in 16 \
#   --c_out 16 \
#   --target 15 \
#   --des 'Exp' \
#   --d_model 256 \
#   --learning_rate 0.0005 \
#   --train_epochs 5 \
#   --batch_size 64 \
#   --d_ff 256 \
#   --itr 1 \
#   --per_variate_scoring

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz.csv \
  --model_id haskins_ema_7_pos_xz_96_96 \
  --model $model_name \
  --data haskins_forecast \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --target 15 \
  --des 'Exp' \
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5 \
  --batch_size 64 \
  --d_ff 256 \
  --itr 1 \
  --per_variate_scoring
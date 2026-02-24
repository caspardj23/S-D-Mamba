export CUDA_VISIBLE_DEVICES=0
# """Copy from traffic forecasting."""

model_name=S_Mamba

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_6.csv \
#   --model_id haskins_ema_6_96_6 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 6 \
#   --e_layers 4 \
#   --enc_in 48 \
#   --dec_in 48 \
#   --c_out 48 \
#   --target 47 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --per_variate_scoring

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_6.csv \
#   --model_id haskins_ema_6_96_12 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 12 \
#   --e_layers 4 \
#   --enc_in 48 \
#   --dec_in 48 \
#   --c_out 48 \
#   --target 47 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --per_variate_scoring

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_6.csv \
#   --model_id haskins_ema_6_96_48 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 4 \
#   --enc_in 48 \
#   --dec_in 48 \
#   --c_out 48 \
#   --target 47 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --per_variate_scoring

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_6.csv \
#   --model_id haskins_ema_6_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 4 \
#   --enc_in 48 \
#   --dec_in 48 \
#   --c_out 48 \
#   --target 47 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --per_variate_scoring

#   python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/haskins/ \
#   --data_path ema_6.csv \
#   --model_id haskins_ema_6_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 4 \
#   --enc_in 48 \
#   --dec_in 48 \
#   --c_out 48 \
#   --target 47 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 64 \
#   --learning_rate 0.0001 \
#   --itr 1 \
#   --per_variate_scoring

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_ema_6_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 48 \
  --dec_in 48 \
  --c_out 48 \
  --target 47 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --itr 1 \
  --per_variate_scoring

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_ema_6_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 48 \
  --dec_in 48 \
  --c_out 48 \
  --target 47 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --itr 1 \
  --per_variate_scoring
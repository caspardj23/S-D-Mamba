export CUDA_VISIBLE_DEVICES=0
"""Copy from traffic forecasting."""

model_name=S_Mamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_96_6_1500s \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --e_layers 4 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_96_12_1500s \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/librivoxspeech/ \
#   --data_path msg_this_side_paradise.csv \
#   --model_id msg_librivoxspeech_96_24 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 4 \
#   --enc_in 80 \
#   --dec_in 80 \
#   --c_out 80 \
#   --target 79 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.002 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/librivoxspeech/ \
#   --data_path msg_this_side_paradise.csv \
#   --model_id msg_librivoxspeech_96_48 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 4 \
#   --enc_in 80 \
#   --dec_in 80 \
#   --c_out 80 \
#   --target 79 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.0008 \
#   --itr 1
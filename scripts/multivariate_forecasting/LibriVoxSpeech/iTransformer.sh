export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_1500s_96_6 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 6 \
  --e_layers 2 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_1500s_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 256 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_1500s_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 512 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/librivoxspeech/ \
  --data_path msg_this_side_paradise_1500s.csv \
  --model_id msg_librivoxspeech_1500s_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 80 \
  --dec_in 80 \
  --c_out 80 \
  --target 79 \
  --des 'Exp' \
  --d_model 512 \
  --learning_rate 0.0005 \
  --train_epochs 5\
  --d_ff 512 \
  --itr 1
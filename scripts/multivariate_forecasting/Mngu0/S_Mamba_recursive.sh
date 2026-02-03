export CUDA_VISIBLE_DEVICES=0
# """Copy from traffic forecasting."""

model_name=S_Mamba

python -u run.py \
  --is_training 3 \
  --root_path ./dataset/mngu0/ \
  --data_path new_test_data.csv \
  --recursive_cycles 10 \
  --model_id mngu0_recursive_test \
  --model S_Mamba \
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
  --batch_size 32 # Note: data loader will force batch_size=1 for this mode
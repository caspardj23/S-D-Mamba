export CUDA_VISIBLE_DEVICES=0
# """Copy from traffic forecasting."""

model_name=S_Mamba

python -u run.py \
  --is_training 3 \
  --root_path ./dataset/mngu0/ \
  --data_path ema_norm_1001_to_1200_12vars.csv \
  --recursive_cycles 10 \
  --model_id mngu0_recursive_test \
  --checkpoint_model_id mngu0_ema_norm_1000_12vars_96_12 \
  --model S_Mamba \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --target 11 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --stride 96 \
  --batch_size 32 # Note: data loader will force batch_size=1 for this mode
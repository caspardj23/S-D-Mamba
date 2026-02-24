#!/bin/bash
# ==============================================================================
# MAE Pre-Training on Haskins EMA Data
#
# Self-supervised pre-training using masked frame prediction.
# Block masking with configurable mask_ratio and block_size.
#
# All parameters can be overridden via environment variables, e.g.:
#   TRAIN_EPOCHS=30 MASK_RATIO=0.5 bash S_Mamba_MAE_pretrain.sh
#
# After pre-training, the encoder checkpoint is saved at:
#   ./checkpoints/<setting>/encoder_checkpoint.pth
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variable ---
TRAIN_EPOCHS=${TRAIN_EPOCHS:-30}

model_name=S_Mamba_MAE

echo "============================================"
echo "MAE Pre-Training on Haskins EMA"
echo "  epochs=${TRAIN_EPOCHS}"
echo "  mask_ratio=0.4, block_size=8"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_mae_pretrain \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 384 \
  --pred_len 384 \
  --e_layers 4 \
  --enc_in 48 \
  --dec_in 48 \
  --c_out 48 \
  --target 47 \
  --des 'MAE_Pretrain' \
  --d_model 256 \
  --d_ff 512 \
  --d_state 32 \
  --d_conv_temporal 4 \
  --expand_temporal 2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs $TRAIN_EPOCHS \
  --patience 10 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_pretrain \
  --mask_ratio 0.4 \
  --block_size 8 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --use_cosine_scheduler \
  --dropout 0.1 \
  --itr 1

echo "============================================"
echo "Pre-training complete."
echo "Encoder checkpoint saved in ./checkpoints/"
echo "============================================"

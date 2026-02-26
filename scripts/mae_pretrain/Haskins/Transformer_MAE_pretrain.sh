#!/bin/bash
# ==============================================================================
# Transformer MAE Pre-Training on Haskins EMA Data
#
# Self-supervised pre-training using masked frame prediction with a
# Transformer encoder (replaces BiMamba). Attention naturally handles
# masked positions without state corruption.
#
# All parameters can be overridden via environment variables, e.g.:
#   TRAIN_EPOCHS=50 D_MODEL=256 bash Transformer_MAE_pretrain.sh
#
# After pre-training, the encoder checkpoint is saved at:
#   ./checkpoints/<setting>/encoder_checkpoint.pth
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
TRAIN_EPOCHS=${TRAIN_EPOCHS:-50}
D_MODEL=${D_MODEL:-128}
N_HEADS=${N_HEADS:-4}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-512}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
MASK_RATIO=${MASK_RATIO:-0.4}
BLOCK_SIZE=${BLOCK_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
SEQ_LEN=${SEQ_LEN:-384}
PATIENCE=${PATIENCE:-15}

model_name=Transformer_MAE

echo "============================================"
echo "Transformer MAE Pre-Training on Haskins EMA"
echo "  epochs=${TRAIN_EPOCHS}"
echo "  d_model=${D_MODEL}, n_heads=${N_HEADS}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, dropout=${DROPOUT}"
echo "  lr=${LR}, weight_decay=${WEIGHT_DECAY}"
echo "  mask_ratio=${MASK_RATIO}, block_size=${BLOCK_SIZE}"
echo "  batch_size=${BATCH_SIZE}, seq_len=${SEQ_LEN}"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_transformer_mae_pretrain \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $SEQ_LEN \
  --e_layers $E_LAYERS \
  --n_heads $N_HEADS \
  --enc_in 48 \
  --dec_in 48 \
  --c_out 48 \
  --target 47 \
  --des 'TransformerMAE_Pretrain' \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --use_norm 0 \
  --loss MSE \
  --exp_name transformer_mae_pretrain \
  --mask_ratio $MASK_RATIO \
  --block_size $BLOCK_SIZE \
  --warmup_epochs 5 \
  --weight_decay $WEIGHT_DECAY \
  --max_grad_norm 1.0 \
  --use_cosine_scheduler \
  --dropout $DROPOUT \
  --itr 1

echo "============================================"
echo "Pre-training complete."
echo "Encoder checkpoint saved in ./checkpoints/"
echo "============================================"

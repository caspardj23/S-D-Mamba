#!/bin/bash
# ==============================================================================
# Transformer MAE Fine-Tuning on Haskins EMA Data
#
# Fine-tunes a pre-trained Transformer MAE encoder for forecasting.
# Supports three strategies: full, freeze, partial.
#
# Usage:
#   # Full fine-tune (default)
#   bash Transformer_MAE_finetune.sh
#
#   # Frozen encoder
#   STRATEGY=freeze bash Transformer_MAE_finetune.sh
#
#   # Partial (unfreeze last 1 layer)
#   STRATEGY=partial UNFREEZE=1 bash Transformer_MAE_finetune.sh
#
# ENCODER_CKPT must point to the pre-trained encoder checkpoint.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
PRED_LEN=${PRED_LEN:-12}
SEQ_LEN=${SEQ_LEN:-384}
D_MODEL=${D_MODEL:-128}
N_HEADS=${N_HEADS:-4}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-512}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-0.0001}
LR_ENCODER=${LR_ENCODER:-0.00001}
STRATEGY=${STRATEGY:-full}
UNFREEZE=${UNFREEZE:-2}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-30}
PATIENCE=${PATIENCE:-10}
BATCH_SIZE=${BATCH_SIZE:-64}

# Path to pre-trained encoder checkpoint — UPDATE THIS after pre-training
ENCODER_CKPT=${ENCODER_CKPT:-"./checkpoints/FILL_IN_SETTING/encoder_checkpoint.pth"}

model_name=Transformer_MAE_Finetune

echo "============================================"
echo "Transformer MAE Fine-Tuning on Haskins EMA"
echo "  pred_len=${PRED_LEN}, strategy=${STRATEGY}"
echo "  encoder_ckpt=${ENCODER_CKPT}"
echo "  lr=${LR}, lr_encoder=${LR_ENCODER}"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_transformer_mae_ft_${STRATEGY}_pl${PRED_LEN} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 0 \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --n_heads $N_HEADS \
  --enc_in 48 \
  --dec_in 48 \
  --c_out 48 \
  --target 47 \
  --des 'TransformerMAE_Finetune' \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --lr_encoder $LR_ENCODER \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --use_norm 0 \
  --loss MSE \
  --exp_name transformer_mae_finetune \
  --pretrain_checkpoint $ENCODER_CKPT \
  --finetune_strategy $STRATEGY \
  --unfreeze_layers $UNFREEZE \
  --max_grad_norm 1.0 \
  --dropout $DROPOUT \
  --itr 1

echo "============================================"
echo "Fine-tuning complete."
echo "============================================"

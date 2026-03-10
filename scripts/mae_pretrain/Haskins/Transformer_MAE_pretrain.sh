#!/bin/bash
# ==============================================================================
# Transformer MAE Pre-Training on Haskins EMA Data (v4)
#
# Self-supervised pre-training using masked frame prediction with a
# Transformer encoder (replaces BiMamba). Attention naturally handles
# masked positions without state corruption.
#
# v4 changes:
#   - 16 variates (posX + posZ) via ema_8_pos_xz.csv
#   - 8 speakers (F01-F04, M01-M04)
#   - Sentence-aware splitting and windowing
#   - seq_len=160, stride=80
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
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MASK_RATIO=${MASK_RATIO:-0.4}
BLOCK_SIZE=${BLOCK_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
SEQ_LEN=${SEQ_LEN:-160}
PATIENCE=${PATIENCE:-50}
MAE_STRIDE=${MAE_STRIDE:-80}
ENC_IN=${ENC_IN:-16}
ALPHA_MASK=${ALPHA_MASK:-1.0}
BETA_NEXT=${BETA_NEXT:-0.0}
GAMMA_SPECTRAL=${GAMMA_SPECTRAL:-0.0}

model_name=Transformer_MAE

echo "============================================"
echo "Transformer MAE Pre-Training on Haskins EMA (v4)"
echo "  variates=${ENC_IN} (posX + posZ)"
echo "  epochs=${TRAIN_EPOCHS}"
echo "  d_model=${D_MODEL}, n_heads=${N_HEADS}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, dropout=${DROPOUT}"
echo "  lr=${LR}, weight_decay=${WEIGHT_DECAY}"
echo "  mask_ratio=${MASK_RATIO}, block_size=${BLOCK_SIZE}"
echo "  batch_size=${BATCH_SIZE}, seq_len=${SEQ_LEN}"
echo "  mae_stride=${MAE_STRIDE}"
echo "  alpha_mask=${ALPHA_MASK}, beta_next=${BETA_NEXT}, gamma_spectral=${GAMMA_SPECTRAL}"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_8_pos_xz.csv \
  --model_id haskins_transformer_mae_pretrain_v4_${TRAIN_EPOCHS}epochs \
  --model $model_name \
  --data haskins_mae \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $SEQ_LEN \
  --e_layers $E_LAYERS \
  --n_heads $N_HEADS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 15 \
  --des 'TransformerMAE_Pretrain_v4' \
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
  --warmup_epochs 3 \
  --weight_decay $WEIGHT_DECAY \
  --max_grad_norm 1.0 \
  --use_cosine_scheduler \
  --dropout $DROPOUT \
  --mae_stride $MAE_STRIDE \
  --alpha_mask $ALPHA_MASK \
  --beta_next $BETA_NEXT \
  --gamma_spectral $GAMMA_SPECTRAL \
  --itr 1 \
  --per_variate_scoring \

echo "============================================"
echo "Pre-training complete."
echo "Encoder checkpoint saved in ./checkpoints/"
echo "============================================"

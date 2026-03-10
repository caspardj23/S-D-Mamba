#!/bin/bash
# ==============================================================================
# S-Mamba MAE Pre-Training on Haskins EMA Data (v4)
#
# Self-supervised pre-training using masked frame prediction.
# Block masking with configurable mask_ratio and block_size.
#
# v4 changes:
#   - 16 variates (posX + posZ only) via ema_7_pos_xz.csv
#   - 8 speakers (F01-F04, M01-M04)
#   - Sentence-aware splitting and windowing (no cross-sentence windows)
#   - seq_len=160 (fits all sentences, min=171 frames)
#   - stride=80 (half seq_len)
#
# All parameters can be overridden via environment variables, e.g.:
#   TRAIN_EPOCHS=50 MASK_RATIO=0.4 bash S_Mamba_MAE_pretrain.sh
#
# After pre-training, the encoder checkpoint is saved at:
#   ./checkpoints/<setting>/encoder_checkpoint.pth
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
TRAIN_EPOCHS=${TRAIN_EPOCHS:-50}
D_MODEL=${D_MODEL:-128}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-256}
D_STATE=${D_STATE:-32}
D_CONV_TEMPORAL=${D_CONV_TEMPORAL:-4}
EXPAND_TEMPORAL=${EXPAND_TEMPORAL:-2}
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
ABLATION_LABEL=${LABEL:-}
IS_TRAINING=${IS_TRAINING:-1}

model_name=S_Mamba_MAE

echo "============================================"
echo "S-Mamba MAE Pre-Training on Haskins EMA (v4)"
echo "  variates=${ENC_IN} (posX + posZ)"
echo "  epochs=${TRAIN_EPOCHS}"
echo "  d_model=${D_MODEL}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, d_state=${D_STATE}"
echo "  d_conv_temporal=${D_CONV_TEMPORAL}, expand_temporal=${EXPAND_TEMPORAL}"
echo "  lr=${LR}, weight_decay=${WEIGHT_DECAY}, dropout=${DROPOUT}"
echo "  mask_ratio=${MASK_RATIO}, block_size=${BLOCK_SIZE}"
echo "  batch_size=${BATCH_SIZE}, seq_len=${SEQ_LEN}"
echo "  mae_stride=${MAE_STRIDE}"
echo "  alpha_mask=${ALPHA_MASK}, beta_next=${BETA_NEXT}, gamma_spectral=${GAMMA_SPECTRAL}"
echo "============================================"

python -u run.py \
  --is_training $IS_TRAINING \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz.csv \
  --model_id haskins_mae_pretrain_v5_${TRAIN_EPOCHS}epochs \
  --model $model_name \
  --data haskins_mae \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $SEQ_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 15 \
  --des "MAE_Pretrain_v5${ABLATION_LABEL:+_${ABLATION_LABEL}}" \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --d_state $D_STATE \
  --d_conv_temporal $D_CONV_TEMPORAL \
  --expand_temporal $EXPAND_TEMPORAL \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_pretrain \
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

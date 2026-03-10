#!/bin/bash
# ==============================================================================
# S-Mamba MAE Fine-Tuning on Haskins EMA Data (v4)
#
# Fine-tunes a pre-trained S-Mamba MAE encoder for forecasting.
# Supports three strategies: full, freeze, partial.
#
# v4 changes:
#   - 16 variates (posX + posZ) via ema_8_pos_xz.csv
#   - 8 speakers (F01-F04, M01-M04)
#   - Sentence-aware splitting and windowing
#   - seq_len=160, stride=80
#
# Usage:
#   # Full fine-tune (default)
#   ENCODER_CKPT=<path> bash S_Mamba_MAE_finetune.sh
#
#   # Frozen encoder
#   ENCODER_CKPT=<path> STRATEGY=freeze bash S_Mamba_MAE_finetune.sh
#
#   # Partial (unfreeze last 2 layers)
#   ENCODER_CKPT=<path> STRATEGY=partial UNFREEZE=2 bash S_Mamba_MAE_finetune.sh
#
#   # Scratch baseline (no pre-training)
#   STRATEGY=full bash S_Mamba_MAE_finetune.sh
#
# All parameters can be overridden via environment variables, e.g.:
#   TRAIN_EPOCHS=50 PRED_LEN=48 ENCODER_CKPT=<path> bash S_Mamba_MAE_finetune.sh
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
PRED_LEN=${PRED_LEN:-48}
SEQ_LEN=${SEQ_LEN:-160}
D_MODEL=${D_MODEL:-128}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-256}
D_STATE=${D_STATE:-32}
D_CONV_TEMPORAL=${D_CONV_TEMPORAL:-4}
EXPAND_TEMPORAL=${EXPAND_TEMPORAL:-2}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-0.0001}
LR_ENCODER=${LR_ENCODER:-0.00001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
STRATEGY=${STRATEGY:-full}
UNFREEZE=${UNFREEZE:-2}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-30}
PATIENCE=${PATIENCE:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
MAE_STRIDE=${MAE_STRIDE:-80}
ENC_IN=${ENC_IN:-16}
GAMMA_SPECTRAL=${GAMMA_SPECTRAL:-0.0}

# Path to pre-trained encoder checkpoint
ENCODER_CKPT=${ENCODER_CKPT:-""}

model_name=S_Mamba_MAE_Finetune

echo "============================================"
echo "S-Mamba MAE Fine-Tuning on Haskins EMA (v4)"
echo "  variates=${ENC_IN} (posX + posZ)"
echo "  pred_len=${PRED_LEN}, strategy=${STRATEGY}"
echo "  d_model=${D_MODEL}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, d_state=${D_STATE}"
echo "  lr=${LR}, lr_encoder=${LR_ENCODER}"
echo "  dropout=${DROPOUT}, weight_decay=${WEIGHT_DECAY}"
echo "  mae_stride=${MAE_STRIDE}"
if [ -n "${ENCODER_CKPT}" ]; then
echo "  encoder_ckpt=${ENCODER_CKPT}"
else
echo "  encoder_ckpt=NONE (scratch baseline)"
fi
echo "============================================"

# Build pre-train checkpoint args
PRETRAIN_ARGS=""
if [ -n "${ENCODER_CKPT}" ]; then
    if [ -f "${ENCODER_CKPT}" ]; then
        PRETRAIN_ARGS="--pretrain_checkpoint ${ENCODER_CKPT}"
    else
        echo "WARNING: Encoder checkpoint not found at: ${ENCODER_CKPT}"
        echo "Searching for any S_Mamba encoder checkpoint..."
        FOUND=$(find ./checkpoints -name "encoder_checkpoint.pth" -path "*S_Mamba_MAE*" | head -1)
        if [ -n "${FOUND}" ]; then
            ENCODER_CKPT="${FOUND}"
            PRETRAIN_ARGS="--pretrain_checkpoint ${ENCODER_CKPT}"
            echo "Using: ${ENCODER_CKPT}"
        else
            echo "No encoder checkpoint found. Running as scratch baseline."
        fi
    fi
fi

# Construct model_id
if [ -n "${PRETRAIN_ARGS}" ]; then
    MODEL_ID="haskins_mae_ft_v4_${STRATEGY}_pl${PRED_LEN}"
    DES="MAE_FT_v4_${STRATEGY}"
else
    MODEL_ID="haskins_mae_ft_v4_scratch_pl${PRED_LEN}"
    DES="MAE_FT_v4_scratch"
fi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_8_pos_xz.csv \
  --model_id $MODEL_ID \
  --model $model_name \
  --data haskins_mae \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 0 \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 15 \
  --des $DES \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --d_state $D_STATE \
  --d_conv_temporal $D_CONV_TEMPORAL \
  --expand_temporal $EXPAND_TEMPORAL \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --lr_encoder $LR_ENCODER \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --finetune_strategy $STRATEGY \
  --unfreeze_layers $UNFREEZE \
  --max_grad_norm 1.0 \
  --use_cosine_scheduler \
  --weight_decay $WEIGHT_DECAY \
  --dropout $DROPOUT \
  --mae_stride $MAE_STRIDE \
  --gamma_spectral $GAMMA_SPECTRAL \
  --itr 1 \
  --per_variate_scoring \
  $PRETRAIN_ARGS

echo "============================================"
echo "Fine-tuning complete."
echo "============================================"

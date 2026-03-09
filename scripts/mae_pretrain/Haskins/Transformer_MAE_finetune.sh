#!/bin/bash
# ==============================================================================
# Transformer MAE Fine-Tuning on Haskins EMA Data (v2)
#
# Fine-tunes a pre-trained Transformer MAE encoder for forecasting.
# Supports three strategies: full, freeze, partial.
#
# v2 changes (aligned with pre-training v2):
#   - 24 variates (position only) via ema_6_pos.csv
#   - Speaker-interleaved train/val/test split (Dataset_Haskins_MAE)
#   - Configurable stride to avoid stride-1 redundancy
#   - Stronger regularization (dropout=0.2, weight_decay=0.01)
#   - Per-variate scoring for speech-specific evaluation
#
# Usage:
#   # Full fine-tune (default)
#   ENCODER_CKPT=<path> bash Transformer_MAE_finetune.sh
#
#   # Frozen encoder
#   ENCODER_CKPT=<path> STRATEGY=freeze bash Transformer_MAE_finetune.sh
#
#   # Partial (unfreeze last 2 layers)
#   ENCODER_CKPT=<path> STRATEGY=partial UNFREEZE=2 bash Transformer_MAE_finetune.sh
#
#   # Scratch baseline (no pre-training)
#   STRATEGY=full bash Transformer_MAE_finetune.sh
#
# All parameters can be overridden via environment variables, e.g.:
#   TRAIN_EPOCHS=50 PRED_LEN=96 ENCODER_CKPT=<path> bash Transformer_MAE_finetune.sh
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
PRED_LEN=${PRED_LEN:-48}
SEQ_LEN=${SEQ_LEN:-384}
D_MODEL=${D_MODEL:-128}
N_HEADS=${N_HEADS:-4}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-512}
DROPOUT=${DROPOUT:-0.2}
LR=${LR:-0.0001}
LR_ENCODER=${LR_ENCODER:-0.00001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
STRATEGY=${STRATEGY:-full}
UNFREEZE=${UNFREEZE:-2}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-30}
PATIENCE=${PATIENCE:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
MAE_STRIDE=${MAE_STRIDE:-192}
ENC_IN=${ENC_IN:-24}
GAMMA_SPECTRAL=${GAMMA_SPECTRAL:-0.0}

# Path to pre-trained encoder checkpoint
ENCODER_CKPT=${ENCODER_CKPT:-""}

# Optional suffix for ablation runs (e.g., "lr3e4", "100ep", "192d")
MODEL_SUFFIX=${MODEL_SUFFIX:-""}

model_name=Transformer_MAE_Finetune

echo "============================================"
echo "Transformer MAE Fine-Tuning on Haskins EMA (v2)"
echo "  variates=${ENC_IN} (position only)"
echo "  pred_len=${PRED_LEN}, strategy=${STRATEGY}"
echo "  d_model=${D_MODEL}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, n_heads=${N_HEADS}"
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
        echo "Searching for any Transformer encoder checkpoint..."
        FOUND=$(find ./checkpoints -name "encoder_checkpoint.pth" -path "*Transformer_MAE*" | head -1)
        if [ -n "${FOUND}" ]; then
            ENCODER_CKPT="${FOUND}"
            PRETRAIN_ARGS="--pretrain_checkpoint ${ENCODER_CKPT}"
            echo "Using: ${ENCODER_CKPT}"
        else
            echo "No encoder checkpoint found. Running as scratch baseline."
        fi
    fi
fi

# Construct model_id (with optional suffix for ablation disambiguation)
SUFFIX_STR=""
if [ -n "${MODEL_SUFFIX}" ]; then
    SUFFIX_STR="_${MODEL_SUFFIX}"
fi

if [ -n "${PRETRAIN_ARGS}" ]; then
    MODEL_ID="haskins_transformer_mae_ft_v2_${STRATEGY}_pl${PRED_LEN}${SUFFIX_STR}"
    DES="TransformerMAE_FT_v2_${STRATEGY}${SUFFIX_STR}"
else
    MODEL_ID="haskins_transformer_mae_ft_v2_scratch_pl${PRED_LEN}${SUFFIX_STR}"
    DES="TransformerMAE_FT_v2_scratch${SUFFIX_STR}"
fi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6_pos.csv \
  --model_id $MODEL_ID \
  --model $model_name \
  --data haskins_mae \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 0 \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --n_heads $N_HEADS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 23 \
  --des $DES \
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

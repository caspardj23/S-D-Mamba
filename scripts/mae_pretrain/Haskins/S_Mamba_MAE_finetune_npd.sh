#!/bin/bash
# ==============================================================================
# S-Mamba MAE Fine-Tuning with NextPatchDecoder on Haskins EMA Data
#
# Uses context-only encoding + pre-trained NextPatchDecoder instead of the
# mask-token approach. This avoids the OOD suffix-mask problem that degraded
# performance in the original fine-tuning pipeline.
#
# Usage:
#   # Full fine-tune (default)
#   ENCODER_CKPT=<path> bash S_Mamba_MAE_finetune_npd.sh
#
#   # Frozen encoder + NPD
#   ENCODER_CKPT=<path> STRATEGY=freeze bash S_Mamba_MAE_finetune_npd.sh
#
#   # With velocity loss
#   ENCODER_CKPT=<path> DELTA_VELOCITY=0.1 bash S_Mamba_MAE_finetune_npd.sh
#
# All parameters can be overridden via environment variables.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
PRED_LEN=${PRED_LEN:-48}
SEQ_LEN=${SEQ_LEN:-160}
D_MODEL=${D_MODEL:-192}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-384}
D_STATE=${D_STATE:-32}
D_CONV_TEMPORAL=${D_CONV_TEMPORAL:-4}
EXPAND_TEMPORAL=${EXPAND_TEMPORAL:-4}
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
DELTA_VELOCITY=${DELTA_VELOCITY:-0.1}
AR_PATCH_SIZE=${AR_PATCH_SIZE:-24}
AR_THRESHOLD=${AR_THRESHOLD:-48}

# Path to pre-trained encoder checkpoint
ENCODER_CKPT=${ENCODER_CKPT:-""}

model_name=S_Mamba_MAE_Finetune

echo "============================================"
echo "S-Mamba NextPatch Fine-Tuning on Haskins EMA"
echo "  variates=${ENC_IN} (posX + posZ)"
echo "  pred_len=${PRED_LEN}, strategy=${STRATEGY}"
echo "  d_model=${D_MODEL}, e_layers=${E_LAYERS}"
echo "  d_ff=${D_FF}, d_state=${D_STATE}"
echo "  lr=${LR}, lr_encoder=${LR_ENCODER}"
echo "  dropout=${DROPOUT}, weight_decay=${WEIGHT_DECAY}"
echo "  mae_stride=${MAE_STRIDE}"
echo "  delta_velocity=${DELTA_VELOCITY}"
echo "  ar_patch_size=${AR_PATCH_SIZE}, ar_threshold=${AR_THRESHOLD}"
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
    MODEL_ID="haskins_mae_ft_npd_${STRATEGY}_pl${PRED_LEN}"
    DES="MAE_FT_NPD_${STRATEGY}"
else
    MODEL_ID="haskins_mae_ft_npd_scratch_pl${PRED_LEN}"
    DES="MAE_FT_NPD_scratch"
fi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz.csv \
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
  --finetune_mode nextpatch \
  --finetune_strategy $STRATEGY \
  --unfreeze_layers $UNFREEZE \
  --max_grad_norm 1.0 \
  --use_cosine_scheduler \
  --weight_decay $WEIGHT_DECAY \
  --dropout $DROPOUT \
  --mae_stride $MAE_STRIDE \
  --gamma_spectral $GAMMA_SPECTRAL \
  --delta_velocity $DELTA_VELOCITY \
  --ar_patch_size $AR_PATCH_SIZE \
  --ar_threshold $AR_THRESHOLD \
  --itr 1 \
  --per_variate_scoring \
  $PRETRAIN_ARGS

echo "============================================"
echo "NextPatch fine-tuning complete."
echo "============================================"

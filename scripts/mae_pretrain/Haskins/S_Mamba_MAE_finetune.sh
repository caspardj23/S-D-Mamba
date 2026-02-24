#!/bin/bash
# ==============================================================================
# Fine-Tune Pre-Trained MAE Encoder for Forecasting on Haskins EMA
#
# Loads the pre-trained encoder from MAE pre-training and attaches a 
# forecasting head. Tests three fine-tuning strategies:
#   1. freeze:  Frozen encoder, only train head
#   2. partial: Unfreeze last 2 encoder layers + head
#   3. full:    All params trainable with differential LR
#
# Also runs a "scratch" baseline (no pre-training) for comparison.
#
# Prerequisites:
#   Run S_Mamba_MAE_pretrain.sh first to generate encoder_checkpoint.pth
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configuration ---
# Update this path to point to your best pre-trained MAE checkpoint
# The exact setting name depends on the pretrain run's model_id + config.
# Find it with: ls ./checkpoints/ | grep mae_pretrain
PRETRAIN_CKPT_DIR="./checkpoints"
# This will be set per-run below; you may need to update the exact setting name.
# Use the helper at the bottom of this file to find the right path.

# Common args
SEQ_LEN=96
PRED_LEN=48
ENC_IN=48
D_MODEL=256
E_LAYERS=4
D_STATE=32
BATCH_SIZE=32
LR=0.0001
EPOCHS=30

# Helper: construct the pre-training setting name to find the checkpoint
# Format: {model_id}_{model}_{data}_{features}_ft{seq_len}_sl{label_len}_pl{pred_len}_...
# For the default pretrain config:
PRETRAIN_SETTING="haskins_mae_pretrain_S_Mamba_MAE_custom_M_ft96_sl48_pl96_dm256_nh8_el4_dl1_df512_fc1_ebtimeF_dtTrue_MAE_Pretrain_projection_0"
PRETRAIN_CKPT="${PRETRAIN_CKPT_DIR}/${PRETRAIN_SETTING}/encoder_checkpoint.pth"

echo "============================================"
echo "Looking for pre-trained checkpoint at:"
echo "  ${PRETRAIN_CKPT}"
echo "============================================"

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "WARNING: Pre-trained checkpoint not found!"
    echo "Searching for any encoder checkpoint..."
    FOUND=$(find ${PRETRAIN_CKPT_DIR} -name "encoder_checkpoint.pth" | head -1)
    if [ -n "${FOUND}" ]; then
        PRETRAIN_CKPT="${FOUND}"
        echo "Using: ${PRETRAIN_CKPT}"
    else
        echo "No encoder checkpoint found. Run pre-training first!"
        echo "Running scratch baseline only..."
        PRETRAIN_CKPT=""
    fi
fi

# ==============================================================================
# Experiment 1: Scratch Baseline (no pre-training)
# ==============================================================================
echo ""
echo "============================================"
echo "Experiment 1: Scratch Baseline (no pre-training)"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_mae_ft_scratch_${PRED_LEN} \
  --model S_Mamba_MAE_Finetune \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 47 \
  --des 'Scratch' \
  --d_model $D_MODEL \
  --d_ff 512 \
  --d_state $D_STATE \
  --d_conv_temporal 4 \
  --expand_temporal 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 5 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --finetune_strategy full \
  --max_grad_norm 1.0 \
  --dropout 0.1 \
  --itr 1

# ==============================================================================
# Skip pre-trained experiments if no checkpoint found
# ==============================================================================
if [ -z "${PRETRAIN_CKPT}" ]; then
    echo "Skipping pre-trained experiments (no checkpoint found)."
    exit 0
fi

# ==============================================================================
# Experiment 2: Freeze Encoder (only train forecasting head)
# ==============================================================================
echo ""
echo "============================================"
echo "Experiment 2: Freeze Encoder"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_mae_ft_freeze_${PRED_LEN} \
  --model S_Mamba_MAE_Finetune \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 47 \
  --des 'Freeze' \
  --d_model $D_MODEL \
  --d_ff 512 \
  --d_state $D_STATE \
  --d_conv_temporal 4 \
  --expand_temporal 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 5 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --pretrain_checkpoint ${PRETRAIN_CKPT} \
  --finetune_strategy freeze \
  --max_grad_norm 1.0 \
  --dropout 0.1 \
  --itr 1

# ==============================================================================
# Experiment 3: Partial Fine-tuning (unfreeze last 2 layers)
# ==============================================================================
echo ""
echo "============================================"
echo "Experiment 3: Partial Fine-tuning (last 2 layers)"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_mae_ft_partial_${PRED_LEN} \
  --model S_Mamba_MAE_Finetune \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 47 \
  --des 'Partial' \
  --d_model $D_MODEL \
  --d_ff 512 \
  --d_state $D_STATE \
  --d_conv_temporal 4 \
  --expand_temporal 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $EPOCHS \
  --patience 5 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --pretrain_checkpoint ${PRETRAIN_CKPT} \
  --finetune_strategy partial \
  --unfreeze_layers 2 \
  --max_grad_norm 1.0 \
  --dropout 0.1 \
  --itr 1

# ==============================================================================
# Experiment 4: Full Fine-tuning (differential LR)
# ==============================================================================
echo ""
echo "============================================"
echo "Experiment 4: Full Fine-tuning (differential LR)"
echo "============================================"

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_6.csv \
  --model_id haskins_mae_ft_full_${PRED_LEN} \
  --model S_Mamba_MAE_Finetune \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 47 \
  --des 'FullFT' \
  --d_model $D_MODEL \
  --d_ff 512 \
  --d_state $D_STATE \
  --d_conv_temporal 4 \
  --expand_temporal 2 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --lr_encoder 0.00001 \
  --train_epochs $EPOCHS \
  --patience 5 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --pretrain_checkpoint ${PRETRAIN_CKPT} \
  --finetune_strategy full \
  --max_grad_norm 1.0 \
  --dropout 0.1 \
  --itr 1

echo ""
echo "============================================"
echo "All fine-tuning experiments complete!"
echo "Results in result_mae_finetune.txt"
echo "============================================"

#!/bin/bash
# ==============================================================================
# MAE Pre-Training Ablation Study on Haskins EMA
#
# Sweeps core MAE hyperparameters at T=48:
#   A. Scratch baseline (no pre-training)
#   B. mask_ratio: {0.3, 0.4, 0.5}        (block_size=8 fixed)
#   C. block_size: {4, 8, 16, 32}          (mask_ratio=0.4 fixed)
#   D. pre-train epochs: {20, 50, 100}     (best mask config)
#   E. fine-tune strategy: freeze vs partial vs full
#   F. seq_len: {192, 384}                 (context length ablation)
#
# Each pre-training run is followed by a fine-tuning run at T=48.
# Results are logged to result_mae_pretrain.txt and result_mae_finetune.txt.
#
# Usage:
#   bash scripts/mae_pretrain/Haskins/S_Mamba_MAE_ablation.sh
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# Common configuration
ROOT_PATH="./dataset/haskins/"
DATA_PATH="ema_6.csv"
DATA="custom"
FEATURES="M"
SEQ_LEN=384
PRED_LEN=48
ENC_IN=48
D_MODEL=256
D_FF=512
D_STATE=32
D_CONV=4
EXPAND=2
E_LAYERS=4
FT_EPOCHS=30
FT_LR=0.0001
FT_BATCH=32
DROPOUT=0.1
CKPT_DIR="./checkpoints"

# ==============================================================================
# Helper function: pre-train and then fine-tune
# ==============================================================================
pretrain_and_finetune() {
    local PT_ID=$1         # pre-train model_id
    local PT_EPOCHS=$2     # pre-train epochs
    local MASK_RATIO=$3    # mask ratio
    local BLOCK_SIZE=$4    # block size
    local FT_STRATEGY=$5   # "freeze" | "partial" | "full"
    local FT_ID=$6         # fine-tune model_id
    local FT_DES=$7        # experiment description

    echo ""
    echo "======================================================"
    echo "Pre-training: ${PT_ID}"
    echo "  mask_ratio=${MASK_RATIO}, block_size=${BLOCK_SIZE}, epochs=${PT_EPOCHS}"
    echo "======================================================"

    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id $PT_ID \
      --model S_Mamba_MAE \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $SEQ_LEN \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des 'MAE_Ablation' \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size 64 \
      --learning_rate 0.001 \
      --train_epochs $PT_EPOCHS \
      --patience 15 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_pretrain \
      --mask_ratio $MASK_RATIO \
      --block_size $BLOCK_SIZE \
      --warmup_epochs 5 \
      --weight_decay 1e-4 \
      --max_grad_norm 1.0 \
      --use_cosine_scheduler \
      --dropout $DROPOUT \
      --itr 1

    # Find the encoder checkpoint
    local ENCODER_CKPT=$(find ${CKPT_DIR} -path "*${PT_ID}*" -name "encoder_checkpoint.pth" | head -1)
    
    if [ -z "${ENCODER_CKPT}" ]; then
        echo "WARNING: Encoder checkpoint not found for ${PT_ID}"
        echo "Skipping fine-tuning for this config."
        return 1
    fi

    echo ""
    echo "======================================================"
    echo "Fine-tuning: ${FT_ID}"
    echo "  strategy=${FT_STRATEGY}, checkpoint=${ENCODER_CKPT}"
    echo "======================================================"

    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id $FT_ID \
      --model S_Mamba_MAE_Finetune \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des "${FT_DES}" \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size $FT_BATCH \
      --learning_rate $FT_LR \
      --lr_encoder 0.00001 \
      --train_epochs $FT_EPOCHS \
      --patience 5 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_finetune \
      --pretrain_checkpoint "${ENCODER_CKPT}" \
      --finetune_strategy $FT_STRATEGY \
      --unfreeze_layers 2 \
      --max_grad_norm 1.0 \
      --dropout $DROPOUT \
      --itr 1
}

# ==============================================================================
# Ablation A: Scratch Baseline (no pre-training)
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation A: Scratch Baseline (no pre-training)"
echo "###################################################################"

python -u run.py \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --model_id haskins_ablation_scratch_48 \
  --model S_Mamba_MAE_Finetune \
  --data $DATA \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --target 47 \
  --des 'Scratch_Baseline' \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --d_state $D_STATE \
  --d_conv_temporal $D_CONV \
  --expand_temporal $EXPAND \
  --batch_size $FT_BATCH \
  --learning_rate $FT_LR \
  --train_epochs $FT_EPOCHS \
  --patience 5 \
  --use_norm 0 \
  --loss MSE \
  --exp_name mae_finetune \
  --finetune_strategy full \
  --max_grad_norm 1.0 \
  --dropout $DROPOUT \
  --itr 1

# ==============================================================================
# Ablation B: Mask Ratio Sweep {0.3, 0.4, 0.5}
# Using full fine-tuning, block_size=8, 50 pre-train epochs
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation B: Mask Ratio Sweep"
echo "###################################################################"

for MASK_RATIO in 0.3 0.4 0.5; do
    RATIO_TAG=$(echo $MASK_RATIO | tr '.' 'p')
    pretrain_and_finetune \
        "haskins_ablation_mr${RATIO_TAG}_pt" \
        50 \
        $MASK_RATIO \
        8 \
        "full" \
        "haskins_ablation_mr${RATIO_TAG}_ft_48" \
        "MaskRatio_${RATIO_TAG}"
done

# ==============================================================================
# Ablation C: Block Size Sweep {4, 8, 16, 32}
# Using full fine-tuning, mask_ratio=0.4, 50 pre-train epochs
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation C: Block Size Sweep"
echo "###################################################################"

for BLOCK_SIZE in 4 8 16 32; do
    pretrain_and_finetune \
        "haskins_ablation_bs${BLOCK_SIZE}_pt" \
        50 \
        0.4 \
        $BLOCK_SIZE \
        "full" \
        "haskins_ablation_bs${BLOCK_SIZE}_ft_48" \
        "BlockSize_${BLOCK_SIZE}"
done

# ==============================================================================
# Ablation D: Pre-Training Epochs {20, 50, 100}
# Using full fine-tuning, mask_ratio=0.4, block_size=8
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation D: Pre-Training Epochs Sweep"
echo "###################################################################"

for PT_EPOCHS in 20 50 100; do
    pretrain_and_finetune \
        "haskins_ablation_ep${PT_EPOCHS}_pt" \
        $PT_EPOCHS \
        0.4 \
        8 \
        "full" \
        "haskins_ablation_ep${PT_EPOCHS}_ft_48" \
        "PTEpochs_${PT_EPOCHS}"
done

# ==============================================================================
# Ablation E: Fine-Tuning Strategy Comparison
# Using best pre-training config (mask_ratio=0.4, block_size=8, 50 epochs)
# Compare: freeze vs partial vs full
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation E: Fine-Tuning Strategy Comparison"
echo "###################################################################"

# Pre-train once with default config (may already exist from Ablation B)
PT_ID="haskins_ablation_strategy_pt"
pretrain_and_finetune \
    $PT_ID \
    50 \
    0.4 \
    8 \
    "freeze" \
    "haskins_ablation_strategy_freeze_48" \
    "Strategy_Freeze"

# Find the checkpoint from the above pre-training
STRATEGY_CKPT=$(find ${CKPT_DIR} -path "*${PT_ID}*" -name "encoder_checkpoint.pth" | head -1)

if [ -n "${STRATEGY_CKPT}" ]; then
    # Partial fine-tuning
    echo ""
    echo "Fine-tuning with PARTIAL strategy..."
    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id haskins_ablation_strategy_partial_48 \
      --model S_Mamba_MAE_Finetune \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des 'Strategy_Partial' \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size $FT_BATCH \
      --learning_rate $FT_LR \
      --train_epochs $FT_EPOCHS \
      --patience 5 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_finetune \
      --pretrain_checkpoint "${STRATEGY_CKPT}" \
      --finetune_strategy partial \
      --unfreeze_layers 2 \
      --max_grad_norm 1.0 \
      --dropout $DROPOUT \
      --itr 1

    # Full fine-tuning
    echo ""
    echo "Fine-tuning with FULL strategy..."
    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id haskins_ablation_strategy_full_48 \
      --model S_Mamba_MAE_Finetune \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des 'Strategy_Full' \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size $FT_BATCH \
      --learning_rate $FT_LR \
      --lr_encoder 0.00001 \
      --train_epochs $FT_EPOCHS \
      --patience 5 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_finetune \
      --pretrain_checkpoint "${STRATEGY_CKPT}" \
      --finetune_strategy full \
      --max_grad_norm 1.0 \
      --dropout $DROPOUT \
      --itr 1
fi

# ==============================================================================
# Ablation F: Context Length (seq_len) Sweep {192, 384}
# Pre-train + fine-tune at different seq_len values.
# Using full fine-tuning, mask_ratio=0.4, block_size=8, 50 PT epochs.
# The default SEQ_LEN=384 was already used in Ablation B (mr0p4).
# Here we run seq_len=192 to compare.
# ==============================================================================
echo ""
echo "###################################################################"
echo "# Ablation F: Context Length (seq_len) Sweep"
echo "###################################################################"

for SL in 192; do
    echo ""
    echo "======================================================"
    echo "Pre-training with seq_len=${SL}"
    echo "======================================================"

    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id "haskins_ablation_sl${SL}_pt" \
      --model S_Mamba_MAE \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SL \
      --pred_len $SL \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des 'MAE_Ablation' \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size 64 \
      --learning_rate 0.001 \
      --train_epochs 50 \
      --patience 15 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_pretrain \
      --mask_ratio 0.4 \
      --block_size 8 \
      --warmup_epochs 5 \
      --weight_decay 1e-4 \
      --max_grad_norm 1.0 \
      --use_cosine_scheduler \
      --dropout $DROPOUT \
      --itr 1

    ENCODER_CKPT_SL=$(find ${CKPT_DIR} -path "*haskins_ablation_sl${SL}_pt*" -name "encoder_checkpoint.pth" | head -1)

    if [ -z "${ENCODER_CKPT_SL}" ]; then
        echo "WARNING: Encoder checkpoint not found for seq_len=${SL}"
        continue
    fi

    echo ""
    echo "======================================================"
    echo "Fine-tuning with seq_len=${SL}, pred_len=${PRED_LEN}"
    echo "======================================================"

    python -u run.py \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --model_id "haskins_ablation_sl${SL}_ft_${PRED_LEN}" \
      --model S_Mamba_MAE_Finetune \
      --data $DATA \
      --features $FEATURES \
      --seq_len $SL \
      --pred_len $PRED_LEN \
      --e_layers $E_LAYERS \
      --enc_in $ENC_IN \
      --dec_in $ENC_IN \
      --c_out $ENC_IN \
      --target 47 \
      --des "SeqLen_${SL}" \
      --d_model $D_MODEL \
      --d_ff $D_FF \
      --d_state $D_STATE \
      --d_conv_temporal $D_CONV \
      --expand_temporal $EXPAND \
      --batch_size $FT_BATCH \
      --learning_rate $FT_LR \
      --lr_encoder 0.00001 \
      --train_epochs $FT_EPOCHS \
      --patience 5 \
      --use_norm 0 \
      --loss MSE \
      --exp_name mae_finetune \
      --pretrain_checkpoint "${ENCODER_CKPT_SL}" \
      --finetune_strategy full \
      --max_grad_norm 1.0 \
      --dropout $DROPOUT \
      --itr 1
done

# The seq_len=384 result comes from Ablation B (mr0p4_ft_48), so no need to re-run.
echo "NOTE: seq_len=384 result is from Ablation B (mask_ratio=0.4)."

echo ""
echo "###################################################################"
echo "# Ablation study complete!"
echo "#"
echo "# Results stored in:"
echo "#   result_mae_pretrain.txt   (pre-training reconstruction metrics)"
echo "#   result_mae_finetune.txt   (downstream forecasting metrics)"
echo "#"
echo "# Summary of experiments:"
echo "#   A: Scratch baseline (no pre-training)"
echo "#   B: Mask ratio sweep   {0.3, 0.4, 0.5}"
echo "#   C: Block size sweep   {4, 8, 16, 32}"
echo "#   D: PT epochs sweep    {20, 50, 100}"
echo "#   E: FT strategy        {freeze, partial, full}"
echo "#   F: seq_len sweep      {192, 384}"
echo "###################################################################"

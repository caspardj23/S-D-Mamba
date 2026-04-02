#!/bin/bash
# ==============================================================================
# S-Mamba MAE Representation Probing on Haskins EMA Data
#
# Freezes a pre-trained MAE encoder and trains a lightweight classifier
# to evaluate whether the representations encode speech-relevant structure.
#
# Probe tasks:
#   - speaker: Speaker identification (8 classes)
#   - phoneme: Phoneme classification (~40 classes) — reads phone column from CSV
#   - manner:  Manner of articulation (6 classes) — derived from phone column
#
# Probe types:
#   - linear: Single linear layer (tests linear separability)
#   - mlp:    Two-layer MLP (tests with minimal nonlinearity)
#
# Compares three representation sources:
#   1. Pre-trained encoder (frozen)
#   2. Random encoder (same architecture, random weights, frozen)
#   3. Raw EMA features (no encoder)
#
# All parameters can be overridden via environment variables.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0

# --- Configurable via environment variables ---
PROBE_TASK=${PROBE_TASK:-speaker}
PROBE_TYPE=${PROBE_TYPE:-linear}
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-}

# Encoder config (must match pre-training)
D_MODEL=${D_MODEL:-128}
E_LAYERS=${E_LAYERS:-3}
D_FF=${D_FF:-256}
D_STATE=${D_STATE:-32}
D_CONV_TEMPORAL=${D_CONV_TEMPORAL:-4}
EXPAND_TEMPORAL=${EXPAND_TEMPORAL:-2}
DROPOUT=${DROPOUT:-0.2}

# Data config
SEQ_LEN=${SEQ_LEN:-160}
MAE_STRIDE=${MAE_STRIDE:-80}
ENC_IN=${ENC_IN:-16}
BATCH_SIZE=${BATCH_SIZE:-64}

# Training config
TRAIN_EPOCHS=${TRAIN_EPOCHS:-50}
LR=${LR:-0.001}
PATIENCE=${PATIENCE:-15}

LABEL=${LABEL:-}

echo "============================================"
echo "S-Mamba MAE Probing on Haskins EMA"
echo "  probe_task=${PROBE_TASK}"
echo "  probe_type=${PROBE_TYPE}"
echo "  checkpoint=${PRETRAIN_CHECKPOINT}"
echo "  d_model=${D_MODEL}, e_layers=${E_LAYERS}"
echo "  seq_len=${SEQ_LEN}, enc_in=${ENC_IN}"
echo "  epochs=${TRAIN_EPOCHS}, batch_size=${BATCH_SIZE}"
echo "============================================"

# Build checkpoint argument
CKPT_ARG=""
if [ -n "$PRETRAIN_CHECKPOINT" ]; then
  CKPT_ARG="--pretrain_checkpoint $PRETRAIN_CHECKPOINT"
fi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/haskins/ \
  --data_path ema_7_pos_xz_phone.csv \
  --model_id "probe_${PROBE_TASK}_${PROBE_TYPE}${LABEL:+_${LABEL}}" \
  --model S_Mamba_MAE \
  --data haskins_mae \
  --features M \
  --seq_len $SEQ_LEN \
  --pred_len $SEQ_LEN \
  --e_layers $E_LAYERS \
  --enc_in $ENC_IN \
  --dec_in $ENC_IN \
  --c_out $ENC_IN \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --d_state $D_STATE \
  --d_conv_temporal $D_CONV_TEMPORAL \
  --expand_temporal $EXPAND_TEMPORAL \
  --dropout $DROPOUT \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --use_norm 0 \
  --exp_name probe \
  --probe_task $PROBE_TASK \
  --probe_type $PROBE_TYPE \
  --mae_stride $MAE_STRIDE \
  $CKPT_ARG \

echo "============================================"
echo "Probing complete. Results in ./probe_results/"
echo "============================================"

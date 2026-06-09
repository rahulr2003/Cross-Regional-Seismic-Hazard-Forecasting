#!/bin/bash

# Variables
BATCH_SIZE=18
EPOCHS=250
N_SEEDS=100
HIDDEN_DIM=256
DROPOUT=0.2
LR=1e-3

# Rename log file to include variables
mv logs/${SLURM_JOB_ID}.txt \
   logs/${SLURM_JOB_ID}_b${BATCH_SIZE}_e${EPOCHS}_s${N_SEEDS}_lr${LR}_dropout${DROPOUT}.txt

python train.py \
    --data_dir data/ \
    --output_dir train_logs/output_b${BATCH_SIZE}_e${EPOCHS}_n${N_SEEDS}_lr${LR}_dropout${DROPOUT}_v2/ \
    --n_seeds $N_SEEDS \
    --n_epochs $EPOCHS \
    --batch_t $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR \
    --dropout $DROPOUT \
    --lambda_con 2.0 \
    --phase1_epochs 45 \
    --seed 1
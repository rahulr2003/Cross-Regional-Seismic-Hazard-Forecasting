#!/bin/bash
CHECKPOINTS_DIR="train_logs/output_b18_e250_n100_lr1e-3_dropout0.2_v2/"
OUTPUT_BASE="output_mcpc_similar_to_58_euclidean"

python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint ${CHECKPOINTS_DIR}/pretrained_seed3.pt \
    --output_dir ${OUTPUT_BASE}/seed3/ \
    --n_cycles 150 \
    --n_seeds 1 \
    --n_source 6 \
    --n_predict 3 \
    --ft_epochs 50 \
    --ft_patience 10 \
    --ft_lr 1e-4

python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint ${CHECKPOINTS_DIR}/pretrained_seed5.pt \
    --output_dir ${OUTPUT_BASE}/seed5/ \
    --n_cycles 150 \
    --n_seeds 1 \
    --n_source 6 \
    --n_predict 3 \
    --ft_epochs 50 \
    --ft_patience 10 \
    --ft_lr 1e-4

python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint ${CHECKPOINTS_DIR}/pretrained_seed7.pt \
    --output_dir ${OUTPUT_BASE}/seed7/ \
    --n_cycles 150 \
    --n_seeds 1 \
    --n_source 6 \
    --n_predict 3 \
    --ft_epochs 50 \
    --ft_patience 10 \
    --ft_lr 1e-4

python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint ${CHECKPOINTS_DIR}/pretrained_seed58.pt \
    --output_dir ${OUTPUT_BASE}/seed58/ \
    --n_cycles 150 \
    --n_seeds 1 \
    --n_source 6 \
    --n_predict 3 \
    --ft_epochs 50 \
    --ft_patience 10 \
    --ft_lr 1e-4

python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint ${CHECKPOINTS_DIR}/pretrained_seed60.pt \
    --output_dir ${OUTPUT_BASE}/seed60/ \
    --n_cycles 150 \
    --n_seeds 1 \
    --n_source 6 \
    --n_predict 3 \
    --ft_epochs 50 \
    --ft_patience 10 \
    --ft_lr 1e-4
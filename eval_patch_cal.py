#!/usr/bin/env python3
"""
eval_patch_cal.py — MCPC evaluation with base rate calibration.

Three conditions evaluated per cycle:
    1. Frozen-prior transfer (ours):
           R-prior + fine-tune F + calibrate base rate to P → predict P
    2. Naive transfer (baseline):
           Global prior + no fine-tune + calibrate base rate to P → predict P
    3. In-distribution (upper bound):
           Global prior + fine-tune P + calibrate base rate to P → predict P

Base rate calibration: resets log_base_rate to match P's empirical
training period positive rate before evaluation. Uses training period
statistics only — no leakage from test period.

Usage:
    python eval_patch_cal.py \
        --data_dir data/ \
        --checkpoint train_logs/output_b18_e150_n10/pretrained_seed42.pt \
        --output_dir output_mcpc_v4/seed42/ \
        --n_cycles 200 \
        --n_seeds 3 \
        --n_source 6 \
        --n_predict 3 \
        --min_pos_rate 0.001
"""

import argparse
import json
import sys
import copy
import math
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PATCHES, ARCH, TARGET,
    TRAIN_END, VAL_END, TEST_END,
    PATCH_REGIMES,
)
from src.data import (
    load_graphs, load_targets,
    compute_class_weights,
    get_batch_targets, get_sample_weights,
    compute_all_metrics, build_prior_for_cycle,
)
from src.model  import SeismicHazardGNN
from src.losses import FocalLoss

import warnings
warnings.filterwarnings('ignore')


# Argument parsing

def parse_args():
    p = argparse.ArgumentParser(
        description='MCPC Transfer Evaluation with Base Rate Calibration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir',     type=str,   required=True)
    p.add_argument('--checkpoint',   type=str,   required=True)
    p.add_argument('--output_dir',   type=str,   required=True)
    p.add_argument('--n_cycles',     type=int,   default=200)
    p.add_argument('--n_seeds',      type=int,   default=3)
    p.add_argument('--n_source',     type=int,   default=6)
    p.add_argument('--n_predict',    type=int,   default=3)
    p.add_argument('--ft_epochs',    type=int,   default=50)
    p.add_argument('--ft_patience',  type=int,   default=10)
    p.add_argument('--ft_lr',        type=float, default=1e-4)
    p.add_argument('--min_pos_rate', type=float, default=0.001)
    p.add_argument('--radius_km',    type=int,   default=TARGET['radius_km'])
    p.add_argument('--mw_thresh',    type=float, default=TARGET['mw_thresh'])
    return p.parse_args()


# Graph prior update

def apply_prior_to_graph(graph, prior_probs, k=6):
    """
    Deep copy graph and replace x_prior with R-derived prior.
    prior_probs: (n_lat, n_lon, k) numpy array
    """
    g     = copy.deepcopy(graph)
    n_lat = g.grid_shape[0].item()
    n_lon = g.grid_shape[1].item()
    prior_flat = torch.tensor(
        prior_probs.reshape(-1, k)[g.valid_idx.cpu().numpy()],
        dtype=torch.float32,
    ).to(g.x_static.device)
    g.x_prior = torch.nan_to_num(prior_flat, nan=1.0 / k)
    return g


# Base rate helpers

def get_patch_base_rate(patch, targets, graphs):
    """
    Compute empirical positive rate from training period only.
    No leakage — test period statistics never used.
    """
    data      = graphs[patch]
    n_lat     = data.grid_shape[0].item()
    n_lon     = data.grid_shape[1].item()
    t_flat    = targets[patch].reshape(n_lat * n_lon, 300)
    valid_idx = data.valid_idx.cpu()          # ensure CPU for indexing
    t_valid   = t_flat[valid_idx]             # (N_valid, 300)
    pos_rate  = t_valid[:, :TRAIN_END].float().mean().item()
    return pos_rate


def calibrate_base_rate(model, patch, targets, graphs):
    """
    Return a copy of model with log_base_rate reset to match
    P's empirical training period positive rate.
    Eliminates miscalibration introduced by fine-tuning on F.
    """
    model_cal = copy.deepcopy(model)
    pos_rate  = get_patch_base_rate(patch, targets, graphs)

    if 0.0 < pos_rate < 1.0:
        new_base = math.log(pos_rate / (1.0 - pos_rate + 1e-8))
        with torch.no_grad():
            model_cal.head.log_base_rate.fill_(new_base)

    return model_cal


# Geological distance

def hellinger_distance(prior_a, prior_b):
    dist_a = np.nanmean(prior_a.reshape(-1, prior_a.shape[-1]), axis=0)
    dist_b = np.nanmean(prior_b.reshape(-1, prior_b.shape[-1]), axis=0)
    dist_a = dist_a / (dist_a.sum() + 1e-8)
    dist_b = dist_b / (dist_b.sum() + 1e-8)
    return float(np.sqrt(0.5 * np.sum(
        (np.sqrt(dist_a) - np.sqrt(dist_b)) ** 2
    )))


def same_regime(patch_a, patch_b):
    return PATCH_REGIMES.get(patch_a) == PATCH_REGIMES.get(patch_b)


# Fine-tuning

def finetune_on_patch(model, patch, graphs, targets,
                      class_weights, device, cfg, seed=0):
    """
    Fine-tune adaptive components on patch F.
    Geological encoder + temporal backbone stay frozen.
    Returns fine-tuned model (deep copy).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model_ft = copy.deepcopy(model)
    model_ft.freeze_backbone()

    data  = graphs[patch].to(device)
    focal = FocalLoss(gamma=2.0, alpha=0.25)

    opt = AdamW(
        model_ft.get_adaptive_params(),
        lr=cfg['ft_lr'],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        opt, T_max=cfg['ft_epochs'], eta_min=1e-6
    )

    w_neg, w_pos = class_weights.get(patch, (1.0, 1.0))
    t_size       = 24
    best_val     = float('inf')
    patience     = 0
    best_state   = None

    for epoch in range(cfg['ft_epochs']):

        # Train step
        model_ft.train()
        model_ft.encoder.eval()
        model_ft.backbone.eval()

        for _ in range(max(1, TRAIN_END // t_size)):
            t_start = np.random.randint(0, max(1, TRAIN_END - t_size))
            t_end   = min(t_start + t_size, TRAIN_END)

            probs = model_ft(data, t_start=t_start, t_end=t_end)
            tgts  = get_batch_targets(
                patch, graphs, targets, t_start, t_end, device
            )
            sw   = get_sample_weights(tgts, w_neg, w_pos)
            loss = focal(probs.flatten(), tgts.flatten(), sw)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model_ft.get_adaptive_params(), 1.0
            )
            opt.step()

        scheduler.step()

        # Validation step
        model_ft.eval()
        with torch.no_grad():
            probs_val = model_ft(data, t_start=TRAIN_END, t_end=VAL_END)
            tgts_val  = get_batch_targets(
                patch, graphs, targets, TRAIN_END, VAL_END, device
            )
            sw_val   = get_sample_weights(tgts_val, w_neg, w_pos)
            val_loss = focal(
                probs_val.flatten(), tgts_val.flatten(), sw_val
            ).item()

        if val_loss < best_val:
            best_val   = val_loss
            patience   = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model_ft.state_dict().items()
            }
        else:
            patience += 1

        if patience >= cfg['ft_patience']:
            break

    if best_state:
        model_ft.load_state_dict(best_state)

    return model_ft


# Evaluation

def evaluate_on_patch(model, patch, graphs, targets, device):
    """Evaluate model on test period of patch P."""
    data = graphs[patch].to(device)
    model.eval()

    with torch.no_grad():
        probs = model(data, t_start=VAL_END, t_end=TEST_END)
        tgts  = get_batch_targets(
            patch, graphs, targets, VAL_END, TEST_END, device
        )

    metrics = compute_all_metrics(probs.flatten(), tgts.flatten())
    return metrics, probs.cpu(), tgts.cpu()


# In-distribution baseline

def indistribution_eval(model, patch, graphs, targets,
                        class_weights, device, cfg):
    """
    Fine-tune on P, calibrate base rate to P, evaluate on P test period.
    Upper bound — best achievable performance per patch.
    """
    model_ft  = finetune_on_patch(
        model, patch, graphs, targets,
        class_weights, device, cfg, seed=0,
    )
    model_cal = calibrate_base_rate(model_ft, patch, targets, graphs)
    metrics, _, _ = evaluate_on_patch(
        model_cal, patch, graphs, targets, device
    )
    return metrics


# MCPC main loop

def run_mcpc(args, model, graphs, targets,
             class_weights, device, feature_dir):

    np.random.seed(42)

    ft_cfg = {
        'ft_lr':       args.ft_lr,
        'ft_epochs':   args.ft_epochs,
        'ft_patience': args.ft_patience,
    }

    CONTINUOUS_FEATURES = [
        'vs30', 'elevation', 'slope', 'roughness',
        'sediment_km', 'crustal_km', 'dist_fault_km',
        'fault_density', 'heat_flow',
    ]
    with open(Path(feature_dir) / "feature_names.json") as f:
        feature_names = json.load(f)
    cont_idx = [feature_names.index(fn) for fn in CONTINUOUS_FEATURES]

    # Eligibility filter
    print("Computing positive rates for eligibility filter...", flush=True)
    eligible         = []
    patch_base_rates = {}

    for patch in PATCHES:
        if patch not in graphs:
            continue
        rate = get_patch_base_rate(patch, targets, graphs)
        patch_base_rates[patch] = rate
        if rate >= args.min_pos_rate:
            eligible.append(patch)
            print(f"  {patch:<25} train_pos_rate={rate:.4f} ✓", flush=True)
        else:
            print(f"  {patch:<25} train_pos_rate={rate:.4f} ✗ (excluded)",
                  flush=True)

    print(f"\nEligible patches: {len(eligible)}", flush=True)

    min_needed = args.n_source + 1 + args.n_predict
    if len(eligible) < min_needed:
        raise ValueError(
            f"Only {len(eligible)} eligible patches but need "
            f"n_source({args.n_source}) + 1 + n_predict({args.n_predict})"
            f" = {min_needed}"
        )

    # In-distribution baselines
    print("\nRunning in-distribution baselines...", flush=True)
    indist_metrics = {}
    for patch in eligible:
        m = indistribution_eval(
            model, patch, graphs, targets,
            class_weights, device, ft_cfg,
        )
        indist_metrics[patch] = m
        print(
            f"  {patch:<25} AUC={m['auc_roc']:.4f}  "
            f"BSS={m['bss']:.4f}  "
            f"pos_rate={m['pos_rate']:.3f}  "
            f"base_rate={patch_base_rates[patch]:.4f}",
            flush=True,
        )

    mean_indist = np.nanmean([
        m['auc_roc'] for m in indist_metrics.values()
        if not np.isnan(m['auc_roc'])
    ])
    print(f"\nMean in-distribution AUC: {mean_indist:.4f}", flush=True)

    # MCPC cycles
    print(f"\nRunning {args.n_cycles} MCPC cycles...", flush=True)
    print(
        f"  R={args.n_source} source  F=1 fine-tune  "
        f"P={args.n_predict} predict",
        flush=True,
    )
    print(
        f"  Base rate calibration: ON (P training period stats)\n",
        flush=True,
    )

    results = []

    for cycle in range(args.n_cycles):

        shuffled  = np.random.permutation(eligible)
        R_patches = list(shuffled[:args.n_source])
        F_patch   = str(shuffled[args.n_source])
        P_patches = list(shuffled[
            args.n_source + 1: args.n_source + 1 + args.n_predict
        ])

        if len(P_patches) < args.n_predict:
            continue

        # Build leakage-free prior from R only
        get_prior, _, _, _, _ = build_prior_for_cycle(
            R_patches   = R_patches,
            feature_dir = feature_dir,
            cont_idx    = cont_idx,
            k=6, n_pca=4,
        )

        # R-prior graphs for F and P (deep copy, replace x_prior)
        graphs_r = {}
        for patch in [F_patch] + P_patches:
            graphs_r[patch] = apply_prior_to_graph(
                graphs[patch], get_prior(patch)
            )
        for patch in R_patches:
            graphs_r[patch] = graphs[patch]

        # Fine-tune on F — once per cycle, shared across all P patches
        ft_models = []
        for seed in range(args.n_seeds):
            model_ft = finetune_on_patch(
                model, F_patch,
                graphs_r, targets,
                class_weights, device,
                ft_cfg, seed=seed,
            )
            ft_models.append(model_ft)

        # Evaluate on each P patch
        for P_patch in P_patches:

            prior_F      = get_prior(F_patch)
            prior_P      = get_prior(str(P_patch))
            d_H          = hellinger_distance(prior_F, prior_P)
            regime_match = same_regime(F_patch, str(P_patch))
            p_base_rate  = patch_base_rates.get(str(P_patch), float('nan'))
            f_base_rate  = patch_base_rates.get(F_patch, float('nan'))

            cycle_results = {
                'cycle':        cycle,
                'F_patch':      F_patch,
                'P_patch':      str(P_patch),
                'R_patches':    [str(r) for r in R_patches],
                'regime_match': bool(regime_match),
                'd_H':          float(d_H),
                'F_regime':     PATCH_REGIMES.get(F_patch, 'Unknown'),
                'P_regime':     PATCH_REGIMES.get(str(P_patch), 'Unknown'),
                'P_base_rate':  float(p_base_rate),
                'F_base_rate':  float(f_base_rate),
                'base_rate_mismatch': float(abs(f_base_rate - p_base_rate)),
            }

            # Condition 1: frozen-prior transfer + calibration
            seed_metrics = []
            for model_ft in ft_models:
                # Calibrate base rate to P before evaluating
                model_cal = calibrate_base_rate(
                    model_ft, str(P_patch), targets, graphs_r
                )
                m, _, _ = evaluate_on_patch(
                    model_cal, str(P_patch),
                    graphs_r, targets, device,
                )
                seed_metrics.append(m)

            for metric in ['auc_roc', 'auc_pr', 'bss', 'brier']:
                vals = [sm[metric] for sm in seed_metrics
                        if not np.isnan(sm[metric])]
                cycle_results[f'transfer_{metric}'] = (
                    float(np.mean(vals)) if vals else float('nan')
                )
                cycle_results[f'transfer_{metric}_std'] = (
                    float(np.std(vals)) if len(vals) > 1 else float('nan')
                )

            # Condition 2: naive + calibration
            # Global prior, no fine-tuning, calibrate base rate to P
            model_naive = calibrate_base_rate(
                model, str(P_patch), targets, graphs
            )
            m_naive, _, _ = evaluate_on_patch(
                model_naive, str(P_patch),
                graphs, targets, device,
            )
            cycle_results['naive_auc_roc'] = float(m_naive['auc_roc'])
            cycle_results['naive_auc_pr']  = float(m_naive['auc_pr'])
            cycle_results['naive_bss']     = float(m_naive['bss'])

            # In-distribution reference for P
            p_indist = indist_metrics.get(str(P_patch), {})
            cycle_results['indist_auc_roc'] = float(
                p_indist.get('auc_roc', float('nan'))
            )

            # Degradation metrics
            indist_auc   = cycle_results['indist_auc_roc']
            transfer_auc = cycle_results['transfer_auc_roc']
            naive_auc    = cycle_results['naive_auc_roc']

            if not np.isnan(indist_auc) and indist_auc > 0:
                cycle_results['transfer_degradation'] = float(
                    (indist_auc - transfer_auc) / indist_auc
                )
                cycle_results['naive_degradation'] = float(
                    (indist_auc - naive_auc) / indist_auc
                )
            else:
                cycle_results['transfer_degradation'] = float('nan')
                cycle_results['naive_degradation']    = float('nan')

            cycle_results['gap_vs_naive'] = float(transfer_auc - naive_auc)

            results.append(cycle_results)

        # Progress every 10 cycles
        if (cycle + 1) % 10 == 0:
            valid = [r for r in results
                     if not np.isnan(r['transfer_auc_roc'])]
            if valid:
                mean_t   = np.mean([r['transfer_auc_roc'] for r in valid])
                mean_n   = np.mean([r['naive_auc_roc'] for r in valid
                                    if not np.isnan(r['naive_auc_roc'])])
                mean_gap = np.mean([r['gap_vs_naive'] for r in valid])
                mean_td  = np.mean([
                    r['transfer_degradation'] for r in valid
                    if not np.isnan(r['transfer_degradation'])
                ])
                mean_nd  = np.mean([
                    r['naive_degradation'] for r in valid
                    if not np.isnan(r['naive_degradation'])
                ])
                print(
                    f"Cycle {cycle+1:>4}/{args.n_cycles} | "
                    f"transfer={mean_t:.4f}  "
                    f"naive={mean_n:.4f}  "
                    f"gap={mean_gap:+.4f}  "
                    f"t_deg={mean_td*100:.1f}%  "
                    f"n_deg={mean_nd*100:.1f}%  "
                    f"n={len(valid)}",
                    flush=True,
                )

    return results, indist_metrics


# Summary

def print_summary(results, indist_metrics):
    valid = [r for r in results
             if not np.isnan(r['transfer_auc_roc'])]

    if not valid:
        print("No valid results", flush=True)
        return

    print(f"\n{'='*75}", flush=True)
    print("MCPC EVALUATION SUMMARY (with base rate calibration)", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"Total valid cycles: {len(valid)}", flush=True)

    t_aucs = [r['transfer_auc_roc'] for r in valid]
    n_aucs = [r['naive_auc_roc']    for r in valid
              if not np.isnan(r['naive_auc_roc'])]
    t_degs = [r['transfer_degradation'] for r in valid
              if not np.isnan(r['transfer_degradation'])]
    n_degs = [r['naive_degradation']    for r in valid
              if not np.isnan(r['naive_degradation'])]
    gaps   = [r['gap_vs_naive'] for r in valid]

    mean_indist = np.nanmean([
        m['auc_roc'] for m in indist_metrics.values()
        if not np.isnan(m['auc_roc'])
    ])

    print(
        f"\n{'Condition':<45} {'AUC':>8} {'±std':>7} {'Degradation':>12}",
        flush=True,
    )
    print("-" * 75, flush=True)
    print(
        f"  {'In-distribution (upper bound)':<43} "
        f"{mean_indist:>8.4f} {'—':>7} {'0.0%':>12}",
        flush=True,
    )
    print(
        f"  {'Frozen-prior + FT on F + calibrate to P':<43} "
        f"{np.mean(t_aucs):>8.4f} {np.std(t_aucs):>7.4f} "
        f"{np.mean(t_degs)*100:>11.1f}%",
        flush=True,
    )
    print(
        f"  {'Naive (global prior + calibrate to P)':<43} "
        f"{np.mean(n_aucs):>8.4f} {np.std(n_aucs):>7.4f} "
        f"{np.mean(n_degs)*100:>11.1f}%",
        flush=True,
    )

    print(
        f"\nGap (ours vs naive):  {np.mean(gaps):+.4f} "
        f"± {np.std(gaps):.4f}  median={np.median(gaps):+.4f}",
        flush=True,
    )
    print(
        f"CV (transfer AUC):    "
        f"{np.std(t_aucs)/np.mean(t_aucs)*100:.2f}%",
        flush=True,
    )

    # Same vs cross regime
    same  = [r for r in valid if r['regime_match']]
    cross = [r for r in valid if not r['regime_match']]
    print(
        f"\n{'Regime pair':<35} {'N':>6} {'Transfer':>10} "
        f"{'Naive':>8} {'Gap':>8} {'T_deg%':>8}",
        flush=True,
    )
    print("-" * 80, flush=True)
    for label, subset in [("Same-regime", same), ("Cross-regime", cross)]:
        if not subset:
            continue
        t = np.mean([r['transfer_auc_roc'] for r in subset])
        n = np.mean([r['naive_auc_roc']    for r in subset
                     if not np.isnan(r['naive_auc_roc'])])
        g = np.mean([r['gap_vs_naive']      for r in subset])
        d = np.mean([r['transfer_degradation'] for r in subset
                     if not np.isnan(r['transfer_degradation'])])
        print(
            f"  {label:<33} {len(subset):>6} {t:>10.4f} "
            f"{n:>8.4f} {g:>+8.4f} {d*100:>8.1f}%",
            flush=True,
        )

    # Per-patch breakdown
    print(f"\nPer-patch (as P):", flush=True)
    print(
        f"  {'Patch':<25} {'N':>6} {'Transfer':>10} {'Naive':>8} "
        f"{'Gap':>8} {'Deg%':>7} {'Indist':>8} {'BaseRate':>9}",
        flush=True,
    )
    print("  " + "-" * 85, flush=True)
    for patch in sorted(set(r['P_patch'] for r in valid)):
        sub = [r for r in valid if r['P_patch'] == patch]
        t   = np.nanmean([r['transfer_auc_roc'] for r in sub])
        n   = np.nanmean([r['naive_auc_roc']    for r in sub])
        g   = np.nanmean([r['gap_vs_naive']      for r in sub])
        d   = np.nanmean([r['transfer_degradation'] for r in sub])
        ind = indist_metrics.get(patch, {}).get('auc_roc', float('nan'))
        br  = sub[0].get('P_base_rate', float('nan')) if sub else float('nan')
        print(
            f"  {patch:<25} {len(sub):>6} {t:>10.4f} {n:>8.4f} "
            f"{g:>+8.4f} {d*100:>7.1f}% {ind:>8.4f} {br:>9.4f}",
            flush=True,
        )

    # Base rate mismatch analysis
    mismatches = [r['base_rate_mismatch'] for r in valid
                  if not np.isnan(r.get('base_rate_mismatch', float('nan')))]
    if mismatches:
        print(f"\nBase rate mismatch analysis:", flush=True)
        q25 = np.quantile(mismatches, 0.25)
        q75 = np.quantile(mismatches, 0.75)
        low_mm  = [r for r in valid
                   if r.get('base_rate_mismatch', float('nan')) <= q25]
        high_mm = [r for r in valid
                   if r.get('base_rate_mismatch', float('nan')) >= q75]
        if low_mm:
            print(
                f"  Low mismatch  (≤{q25:.3f}): "
                f"mean gap={np.mean([r['gap_vs_naive'] for r in low_mm]):+.4f}  "
                f"n={len(low_mm)}",
                flush=True,
            )
        if high_mm:
            print(
                f"  High mismatch (≥{q75:.3f}): "
                f"mean gap={np.mean([r['gap_vs_naive'] for r in high_mm]):+.4f}  "
                f"n={len(high_mm)}",
                flush=True,
            )


# Main

def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {args.data_dir}...", flush=True)
    data_dir      = Path(args.data_dir)
    graphs        = load_graphs(data_dir / "graphs")
    targets       = load_targets(
        data_dir / "targets",
        radius_km = args.radius_km,
        mw_thresh = args.mw_thresh,
    )
    class_weights = compute_class_weights(graphs, targets)
    feature_dir   = str(data_dir / "feature_tensors")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}", flush=True)
    model = SeismicHazardGNN.from_config(ARCH).to(device)
    state = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded", flush=True)
    print(f"Model on: {next(model.parameters()).device}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    # Run MCPC
    results, indist_metrics = run_mcpc(
        args          = args,
        model         = model,
        graphs        = graphs,
        targets       = targets,
        class_weights = class_weights,
        device        = device,
        feature_dir   = feature_dir,
    )

    print_summary(results, indist_metrics)

    # Save results
    out = {
        'args':           vars(args),
        'results':        results,
        'indist_metrics': {
            k: {mk: float(mv) for mk, mv in v.items()
                if isinstance(mv, (int, float))}
            for k, v in indist_metrics.items()
        },
    }
    with open(out_dir / "mcpc_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {out_dir}/mcpc_results.json", flush=True)


if __name__ == "__main__":
    main()
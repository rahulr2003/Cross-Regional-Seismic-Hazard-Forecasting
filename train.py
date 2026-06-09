"""
train.py — Entry point for seismic hazard GNN pre-training.

Usage:
    # Single seed
    python train.py --data_dir /path/to/data --output_dir /path/to/output

    # Multiple seeds
    python train.py --data_dir /path/to/data --output_dir /path/to/output --n_seeds 3

    # Custom hyperparameters
    python train.py --data_dir /path/to/data --output_dir /path/to/output \
                    --n_epochs 150 --lr 1e-4 --batch_t 32

    # Different target definition
    python train.py --data_dir /path/to/data --output_dir /path/to/output \
                    --radius_km 30 --mw_thresh 3.5
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Add project root to path so src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config  import ARCH, TRAIN, TARGET, PATCHES
from src.data    import load_all_data, compute_class_weights, print_class_weights
from src.trainer import pretrain, save_checkpoint


# Argument parsing

def parse_args():
    p = argparse.ArgumentParser(
        description='Seismic Hazard GNN — Pre-training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument('--data_dir',    type=str, required=True,
                   help='Path to data directory containing graphs/ and targets/')
    p.add_argument('--output_dir',  type=str, required=True,
                   help='Path to save model checkpoints and history')

    # Seeds
    p.add_argument('--seed',        type=int, default=42,
                   help='Base random seed')
    p.add_argument('--n_seeds',     type=int, default=TRAIN['n_seeds'],
                   help='Number of seeds to run')

    # Training hyperparameters
    p.add_argument('--n_epochs',    type=int,   default=TRAIN['n_epochs'])
    p.add_argument('--lr',          type=float, default=TRAIN['lr'])
    p.add_argument('--weight_decay',type=float, default=TRAIN['weight_decay'])
    p.add_argument('--patience',    type=int,   default=TRAIN['patience'])
    p.add_argument('--batch_t',     type=int,   default=TRAIN['batch_t_size'],
                   help='Time steps per training batch')
    p.add_argument('--lambda_adv',  type=float, default=TRAIN['lambda_adv'],
                   help='Adversarial loss weight')
    p.add_argument('--lambda_con',  type=float, default=TRAIN['lambda_con'],
                   help='Contrastive loss weight')
    p.add_argument('--max_weight',  type=float, default=TRAIN['max_weight'],
                   help='Class weight cap')
    p.add_argument('--clip_grad',   type=float, default=TRAIN['clip_grad'],
                   help='Gradient clipping norm')

    # Architecture overrides
    p.add_argument('--hidden_dim',  type=int,   default=ARCH['hidden_dim'])
    p.add_argument('--embed_dim',   type=int,   default=ARCH['node_embed_dim'])
    p.add_argument('--geo_dim',     type=int,   default=ARCH['geo_embed_dim'])
    p.add_argument('--n_heads',     type=int,   default=ARCH['n_heads'])
    p.add_argument('--dropout',     type=float, default=ARCH['dropout'])

    # Target definition
    p.add_argument('--radius_km',   type=int,   default=TARGET['radius_km'],
                   help='Spatial radius for target variable (km)')
    p.add_argument('--mw_thresh',   type=float, default=TARGET['mw_thresh'],
                   help='Magnitude threshold for target variable')

    return p.parse_args()


# Build config namespace

def build_config(args):
    """
    Merge parsed args with defaults into a structured config namespace.
    cfg.arch  — architecture hyperparameters
    cfg.train — training hyperparameters
    """
    arch = dict(
        static_dim     = ARCH['static_dim'],
        prior_dim      = ARCH['prior_dim'],
        temporal_dim   = ARCH['temporal_dim'],
        geo_embed_dim  = args.geo_dim,
        hidden_dim     = args.hidden_dim,
        node_embed_dim = args.embed_dim,
        n_heads        = args.n_heads,
        dropout        = args.dropout,
    )

    train = dict(
        n_epochs      = args.n_epochs,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        patience      = args.patience,
        batch_t_size  = args.batch_t,
        lambda_adv    = args.lambda_adv,
        lambda_con    = args.lambda_con,
        alpha_max     = TRAIN['alpha_max'],
        clip_grad     = args.clip_grad,
        max_weight    = args.max_weight,
        finetune_lr       = 1e-4,
        finetune_epochs   = 50,
        finetune_patience = 10,
    )

    return SimpleNamespace(arch=arch, train=train)


# Device setup

def setup_device():
    """Detect and report available hardware."""
    if torch.cuda.is_available():
        device = "cuda" #torch.device("cuda")
        # props  = torch.cuda.get_device_properties(0)
        # print(f"GPU:   {props.name}",flush=True)
        # print(f"VRAM:  {props.total_memory / 1e9:.1f} GB",flush=True)
        # print(f"CUDA:  {torch.version.cuda}",flush=True)
    else:
        device = torch.device("cpu")
        print("GPU:   not available — using CPU",flush=True)

    print(f"Device: {device}\n",flush=True)
    return device


# Summary reporting

def print_run_summary(all_histories, output_dir):
    """Print per-seed training summary and save combined history."""
    print(f"\n{'='*60}",flush=True)
    print("TRAINING COMPLETE",flush=True)
    print(f"{'='*60}",flush=True)
    print(f"{'Seed':<10} {'Best val_loss':>14} {'Best epoch':>11} "
          f"{'Best val_auc':>13} {'Total epochs':>13}",flush=True)
    print("-" * 60,flush=True)

    for seed_key, hist in all_histories.items():
        best = min(hist, key=lambda x: x['val_loss'])
        last = hist[-1]
        print(
            f"  {seed_key:<8} "
            f"{best['val_loss']:>14.4f} "
            f"{best['epoch']:>11} "
            f"{best['val_auc']:>13.4f} "
            f"{last['epoch']:>13}",flush=True
        )

    # Save combined history
    out = Path(output_dir)
    with open(out / "all_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"\nHistories saved to {out}/all_histories.json",flush=True)


# Main

def main():
    args   = parse_args()
    cfg    = build_config(args)
    device = "cuda" #setup_device()

    # Print run configuration
    print("Configuration:")
    print(f"  data_dir:    {args.data_dir}",flush=True)
    print(f"  output_dir:  {args.output_dir}",flush=True)
    print(f"  seeds:       {[args.seed + i for i in range(args.n_seeds)]}",flush=True)
    print(f"  target:      r={args.radius_km}km  Mw≥{args.mw_thresh}",flush=True)
    print(f"  n_epochs:    {cfg.train['n_epochs']}",flush=True)
    print(f"  lr:          {cfg.train['lr']}",flush=True)
    print(f"  batch_t:     {cfg.train['batch_t_size']} steps",flush=True)
    print(f"  lambda_adv:  {cfg.train['lambda_adv']}",flush=True)
    print(f"  lambda_con:  {cfg.train['lambda_con']}",flush=True)
    print("",flush=True)

    # Load data
    print("Loading data...")
    graphs, targets = load_all_data(
        args.data_dir,
        radius_km = args.radius_km,
        mw_thresh = args.mw_thresh,
    )

    # Class weights
    class_weights = compute_class_weights(
        graphs, targets,
        max_weight = args.max_weight,
    )
    print_class_weights(class_weights)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_to_save = dict(
        arch   = cfg.arch,
        train  = cfg.train,
        target = dict(radius_km=args.radius_km, mw_thresh=args.mw_thresh),
        seeds  = [args.seed + i for i in range(args.n_seeds)],
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"\nConfig saved to {out_dir}/config.json",flush=True)

    # Run seeds
    all_histories = {}

    for i in range(args.n_seeds):
        seed = args.seed + i

        # Memory before
        mem_before = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory before seed {seed}: {mem_before:.3f} GB")

        model, history = pretrain(
            cfg           = cfg,
            graphs        = graphs,
            targets       = targets,
            class_weights = class_weights,
            device        = device,
            seed          = seed,
        )

        # Save checkpoint
        save_checkpoint(
            model      = model,
            history    = history,
            output_dir = args.output_dir,
            name       = f"pretrained_seed{seed}",
        )

        # Memory before cleanup
        mem_peak = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after training seed {seed}: {mem_peak:.3f} GB")

        # Explicit cleanup between seeds
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f"GPU memory after seed {seed}: "
          f"{torch.cuda.memory_allocated()/1e9:.3f} GB allocated",flush=True)

        all_histories[f"seed{seed}"] = history

    # Summary
    print_run_summary(all_histories, args.output_dir)


if __name__ == "__main__":
    main()
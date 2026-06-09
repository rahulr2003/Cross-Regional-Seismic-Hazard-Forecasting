# src/trainer.py
# Training and validation loops for seismic hazard GNN.
#
# Two training modes:
#   pretrain()  — global pre-training on all source patches
#                 all parameters trainable
#                 three-component loss: focal + adversarial + contrastive
#
#   finetune()  — fine-tuning on a single patch F
#                 only adaptive components update (3,141 params)
#                 focal loss only

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from .config  import (
    PATCHES, EXCLUDE_FROM_TRAIN,
    TRAIN_END, VAL_END, TEST_END,
)
from .model   import SeismicHazardGNN
from .losses  import (
    FocalLoss, PatchDiscriminator,
    ContrastiveGeologicalLoss, compute_combined_loss,
)
from .data    import (
    get_batch_targets, get_cluster_labels,
    get_sample_weights, compute_all_metrics,
)


# Pre-training

def pretrain(cfg, graphs, targets, class_weights, device, seed=42):
    """
    Pre-train SeismicHazardGNN on all source patches simultaneously.

    Loss:
        L = L_focal + lambda_adv * L_adversarial + lambda_con * L_contrastive

    Adversarial loss encourages backbone to produce patch-invariant
    (regime-invariant) node embeddings via gradient reversal.

    Contrastive loss encourages backbone to encode geological cluster
    structure consistently with the frozen GMM prior.

    Parameters
    ----------
    cfg          : SimpleNamespace with .arch and .train dicts
    graphs       : dict patch_name → PyG Data
    targets      : dict patch_name → tensor (n_lat, n_lon, 300)
    class_weights: dict patch_name → (w_neg, w_pos)
    device       : torch.device
    seed         : int

    Returns
    -------
    model   : trained SeismicHazardGNN (best checkpoint restored)
    history : list of per-epoch log dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}",flush=True)
    print(f"Pre-training  seed={seed}  device={device}",flush=True)
    print(f"{'='*60}",flush=True)

    # Initialise
    model = SeismicHazardGNN.from_config(cfg.arch).to(device)
    model.unfreeze_all()
    model.print_param_summary()

    disc = PatchDiscriminator(
        embed_dim  = cfg.arch['node_embed_dim'],
        n_patches  = len(PATCHES),
        hidden_dim = 32,
    ).to(device)

    focal       = FocalLoss(gamma=2.0, alpha=0.25)
    contrastive = ContrastiveGeologicalLoss(temperature=0.07)

    opt_model = AdamW(
        model.parameters(),
        lr           = cfg.train['lr'],
        weight_decay = cfg.train['weight_decay'],
    )
    opt_disc = AdamW(
        disc.parameters(),
        lr           = cfg.train['lr'],
        weight_decay = cfg.train['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        opt_model,
        T_max  = cfg.train['n_epochs'],
        eta_min = 1e-6,
    )

    train_patches = [
        p for p in PATCHES
        if p in graphs and p not in EXCLUDE_FROM_TRAIN
    ]
    print(f"\nTraining patches ({len(train_patches)}): {train_patches}",flush=True)
    print(f"Excluded:                          {EXCLUDE_FROM_TRAIN}",flush=True)
    print(f"Validated on all {len([p for p in PATCHES if p in graphs])} patches\n",flush=True)

    best_val   = float('inf')
    patience   = 0
    best_state = None
    history    = []

    for epoch in range(cfg.train['n_epochs']):

        # Gradient reversal annealing (Ganin 2016 schedule)
        # alpha ramps from 0 → alpha_max over training
        progress = epoch / cfg.train['n_epochs']
        alpha    = cfg.train['alpha_max'] * (
            2 / (1 + np.exp(-10 * progress)) - 1
        )

        # Train
        model.train()
        disc.train()
        ep = {'focal': [], 'adv': [], 'con': [], 'total': []}

        for pi, patch in enumerate(
            np.random.permutation(train_patches)
        ):
            data = graphs[patch].to(device)

            # Random time window within training period
            n_train = TRAIN_END
            t_size  = cfg.train['batch_t_size']
            t_start = np.random.randint(0, max(1, n_train - t_size))
            t_end   = min(t_start + t_size, n_train)

            # Forward pass → probabilities
            probs = model(data, t_start=t_start, t_end=t_end)
            tgts  = get_batch_targets(
                patch, graphs, targets, t_start, t_end, device
            )

            # Sample weights from class balancing
            w_neg, w_pos = class_weights.get(patch, (1.0, 1.0))
            sw = get_sample_weights(tgts, w_neg, w_pos)

            # Node embeddings for auxiliary losses
            # Recompute through backbone to get gradients
            geo_embed   = model.encoder(data.x_static, data.x_prior)
            edge_scaled = model.edge_weight_scaler(data.edge_attr)
            node_embed  = model.backbone(
                data.x_temporal[:, t_start:t_end, :],
                geo_embed,
                data.edge_index,
                edge_scaled,
                n_steps    = t_end - t_start,
                return_all_steps = False,
            )  # (N, node_embed_dim) — final step embedding

            # Discriminator logits with gradient reversal
            disc_logits  = disc(node_embed, alpha=alpha)
            patch_labels = torch.full(
                (node_embed.shape[0],),
                pi % len(PATCHES),
                dtype = torch.long,
                device = device,
            )

            # Cluster labels for contrastive loss
            cluster_lbls = get_cluster_labels(patch, graphs, device)

            # Combined loss
            loss, loss_dict = compute_combined_loss(
                probs        = probs.flatten(),
                targets      = tgts.flatten(),
                sample_weights = sw,
                node_embed   = node_embed,
                disc_logits  = disc_logits,
                patch_labels = patch_labels,
                cluster_labels = cluster_lbls,
                focal        = focal,
                contrastive  = contrastive,
                lambda_adv   = cfg.train['lambda_adv'],
                lambda_con   = cfg.train['lambda_con'],
            )

            opt_model.zero_grad()
            opt_disc.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm = cfg.train['clip_grad'],
            )
            opt_model.step()
            opt_disc.step()

            for k, v in loss_dict.items():
                ep[k].append(v)

        scheduler.step()

        # Validate on ALL patches including sparse ones
        val_metrics = _validate(
            model, graphs, targets, class_weights,
            focal, device,
            t_start = TRAIN_END,
            t_end   = VAL_END,
        )

        mean_vl  = val_metrics['mean_val_loss']
        mean_auc = val_metrics['mean_val_auc']

        # Discriminator accuracy monitoring
        disc_acc = _discriminator_accuracy(
            model, disc, graphs, train_patches, device,
            t_step=0
        )

        # Early stopping on validation loss
        if mean_vl < best_val:
            best_val   = mean_vl
            patience   = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience += 1

        log = dict(
            epoch       = epoch + 1,
            seed        = seed,
            loss_focal  = float(np.mean(ep['focal'])),
            loss_adv    = float(np.mean(ep['adv'])),
            loss_con    = float(np.mean(ep['con'])),
            loss_total  = float(np.mean(ep['total'])),
            val_loss    = float(mean_vl),
            val_auc     = float(mean_auc),
            disc_acc    = float(disc_acc),
            alpha       = float(alpha),
            lr          = float(scheduler.get_last_lr()[0]),
            patience    = patience,
        )
        history.append(log)

        # Logging every 5 epochs and first 3
        if (epoch + 1) % 5 == 0 or epoch < 3:
            print(
                f"Ep {epoch+1:>3} | "
                f"focal={log['loss_focal']:.4f}  "
                f"adv={log['loss_adv']:.4f}  "
                f"con={log['loss_con']:.4f} | "
                f"val_loss={mean_vl:.4f}  "
                f"val_auc={mean_auc:.4f} | "
                f"disc_acc={disc_acc:.3f}  "
                f"α={alpha:.3f}  "
                f"p={patience}/{cfg.train['patience']}",flush=True
            )

        if patience >= cfg.train['patience']:
            print(f"\nEarly stopping at epoch {epoch + 1}",flush=True)
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint  (val_loss={best_val:.4f})",flush=True)

    return model, history

def pretrain(cfg, graphs, targets, class_weights, device, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}",flush=True)
    print(f"Pre-training  seed={seed}  device={device}",flush=True)
    print(f"Two-phase: Phase 1 (con+adv only) → Phase 2 (full loss)",flush=True)
    print(f"{'='*60}",flush=True)

    model = SeismicHazardGNN.from_config(cfg.arch).to(device)
    model.unfreeze_all()
    model.print_param_summary()

    disc = PatchDiscriminator(
        embed_dim  = cfg.arch['node_embed_dim'],
        n_patches  = len(PATCHES),
        hidden_dim = 32,
    ).to(device)

    focal       = FocalLoss(gamma=2.0, alpha=0.25)
    contrastive = ContrastiveGeologicalLoss()

    opt_model = AdamW(
        model.parameters(),
        lr           = cfg.train['lr'],
        weight_decay = cfg.train['weight_decay'],
    )
    opt_disc = AdamW(
        disc.parameters(),
        lr           = cfg.train['lr'],
        weight_decay = cfg.train['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        opt_model,
        T_max   = cfg.train['n_epochs'],
        eta_min = 1e-6,
    )

    train_patches = [
        p for p in PATCHES
        if p in graphs and p not in EXCLUDE_FROM_TRAIN
    ]

    # Phase boundaries
    phase1_epochs = cfg.train.get('phase1_epochs', 60)
    n_epochs      = cfg.train['n_epochs']

    print(f"\nPhase 1: epochs 1–{phase1_epochs}  "
          f"(contrastive + adversarial only)",flush=True)
    print(f"Phase 2: epochs {phase1_epochs+1}–{n_epochs}  "
          f"(full loss — focal added gradually)",flush=True)
    print(f"Training patches ({len(train_patches)}): {train_patches}\n",flush=True)

    best_val   = float('inf')
    patience   = 0
    best_state = None
    history    = []

    for epoch in range(n_epochs):

        progress = epoch / n_epochs
        alpha    = cfg.train['alpha_max'] * (
            2 / (1 + np.exp(-10 * progress)) - 1
        )

        # Focal loss weight ramps from 0 → 1 over phase 2
        if epoch < phase1_epochs:
            focal_weight = 0.0
        else:
            focal_weight = min(
                1.0,
                (epoch - phase1_epochs) /
                max(1, phase1_epochs * 0.5)
            )

        # Train
        model.train()
        disc.train()
        ep = {'focal': [], 'adv': [], 'con': [], 'total': []}

        for pi, patch in enumerate(
            np.random.permutation(train_patches)
        ):
            data    = graphs[patch].to(device)
            n_train = TRAIN_END
            t_size  = cfg.train['batch_t_size']
            t_start = np.random.randint(0, max(1, n_train - t_size))
            t_end   = min(t_start + t_size, n_train)

            # Node embeddings — always needed for con + adv
            geo_embed   = model.encoder(data.x_static, data.x_prior)
            edge_scaled = model.edge_weight_scaler(data.edge_attr)
            node_embed  = model.backbone(
                data.x_temporal[:, t_start:t_end, :],
                geo_embed,
                data.edge_index,
                edge_scaled,
                n_steps          = t_end - t_start,
                return_all_steps = False,
            )

            # Contrastive loss — both phases
            cluster_lbls = get_cluster_labels(patch, graphs, device)
            loss_con     = contrastive(node_embed, cluster_lbls)
            if torch.isnan(loss_con):
                loss_con = torch.tensor(0.0, device=device)

            # Adversarial loss — both phases
            disc_logits  = disc(node_embed, alpha=alpha)
            patch_labels = torch.full(
                (node_embed.shape[0],),
                pi % len(PATCHES),
                dtype  = torch.long,
                device = device,
            )
            loss_adv = F.cross_entropy(disc_logits, patch_labels)

            # Focal loss — phase 2 only, ramped in gradually
            if focal_weight > 0:
                probs   = model(data, t_start=t_start, t_end=t_end)
                tgts    = get_batch_targets(
                    patch, graphs, targets, t_start, t_end, device
                )
                w_neg, w_pos = class_weights.get(patch, (1.0, 1.0))
                sw           = get_sample_weights(tgts, w_neg, w_pos)
                loss_focal   = focal(
                    probs.flatten(), tgts.flatten(), sw
                )
            else:
                loss_focal = torch.tensor(0.0, device=device)

            # Combined loss
            loss = (
                focal_weight       * loss_focal
                + cfg.train['lambda_adv'] * loss_adv
                + cfg.train['lambda_con'] * loss_con
            )

            opt_model.zero_grad()
            opt_disc.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm = cfg.train['clip_grad'],
            )
            opt_model.step()
            opt_disc.step()

            ep['focal'].append(loss_focal.item())
            ep['adv'].append(loss_adv.item())
            ep['con'].append(loss_con.item())
            ep['total'].append(loss.item())

        scheduler.step()

        # Only compute val AUC during phase 2 — meaningless in phase 1
        if epoch >= phase1_epochs:
            val_metrics = _validate(
                model, graphs, targets, class_weights,
                focal, device,
                t_start = TRAIN_END,
                t_end   = VAL_END,
            )
            mean_vl  = val_metrics['mean_val_loss']
            mean_auc = val_metrics['mean_val_auc']
        else:
            mean_vl  = float('inf')
            mean_auc = float('nan')

        disc_acc = _discriminator_accuracy(
            model, disc, graphs, train_patches, device, t_step=0
        )

        # Early stopping only in phase 2
        if epoch >= phase1_epochs and mean_vl < best_val:
            best_val   = mean_vl
            patience   = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        elif epoch >= phase1_epochs:
            patience += 1

        phase = "P1" if epoch < phase1_epochs else "P2"
        log   = dict(
            epoch        = epoch + 1,
            phase        = phase,
            seed         = seed,
            loss_focal   = float(np.mean(ep['focal'])),
            loss_adv     = float(np.mean(ep['adv'])),
            loss_con     = float(np.mean(ep['con'])),
            loss_total   = float(np.mean(ep['total'])),
            val_loss     = float(mean_vl),
            val_auc      = float(mean_auc),
            disc_acc     = float(disc_acc),
            focal_weight = float(focal_weight),
            alpha        = float(alpha),
            lr           = float(scheduler.get_last_lr()[0]),
            patience     = patience,
        )
        history.append(log)

        if (epoch + 1) % 5 == 0 or epoch < 3:
            print(
                f"[{phase}] Ep {epoch+1:>3} | "
                f"con={log['loss_con']:.4f}  "
                f"adv={log['loss_adv']:.4f}  "
                f"focal={log['loss_focal']:.4f}  "
                f"fw={focal_weight:.2f} | "
                f"val_auc={mean_auc:.4f}  "
                f"disc={disc_acc:.3f}  "
                f"p={patience}/{cfg.train['patience']}",flush=True
            )

        if (epoch >= phase1_epochs and
                patience >= cfg.train['patience']):
            print(f"\nEarly stopping at epoch {epoch + 1}",flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint (val_loss={best_val:.4f})",flush=True)

    return model, history

# Fine-tuning

def finetune(model, patch, graphs, targets, class_weights,
             device, cfg, seed=0):
    """
    Fine-tune adaptive components on a single patch F.

    Frozen:   encoder + backbone  (194,528 params)
    Trainable: edge_weight_scaler + head + calibration  (3,141 params)

    Uses focal loss only — no adversarial or contrastive losses
    since we're adapting to a specific patch, not learning invariances.

    Two-phase training:
        Phase 1 — train head + edge_weight_scaler on training period
        Phase 2 — fit calibration layer on validation period

    Parameters
    ----------
    model        : pretrained SeismicHazardGNN
    patch        : str — patch name (F)
    graphs       : dict patch_name → PyG Data
    targets      : dict patch_name → tensor
    class_weights: dict patch_name → (w_neg, w_pos)
    device       : torch.device
    cfg          : SimpleNamespace with .train dict
    seed         : int

    Returns
    -------
    model   : fine-tuned model (best checkpoint on val loss)
    history : list of per-epoch log dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nFine-tuning on {patch}  seed={seed}",flush=True)

    if patch not in graphs:
        raise ValueError(f"Patch {patch} not found in graphs")

    # Freeze backbone — only adaptive params update
    model.freeze_backbone()
    counts = model.param_counts()
    print(f"  Trainable: {counts['trainable']:,}  "
          f"Frozen: {counts['frozen']:,}",flush=True)

    data  = graphs[patch].to(device)
    focal = FocalLoss(gamma=2.0, alpha=0.25)

    # Phase 1: train head + edge scaler
    opt = AdamW(
        model.get_adaptive_params(),
        lr           = cfg.train.get('finetune_lr', 1e-4),
        weight_decay = cfg.train['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        opt,
        T_max   = cfg.train.get('finetune_epochs', 50),
        eta_min = 1e-6,
    )

    best_val   = float('inf')
    patience   = 0
    best_state = None
    history    = []

    n_train  = TRAIN_END
    n_val    = VAL_END - TRAIN_END
    t_size   = cfg.train['batch_t_size']
    w_neg, w_pos = class_weights.get(patch, (1.0, 1.0))

    for epoch in range(cfg.train.get('finetune_epochs', 50)):

        # Train
        model.train()
        model.encoder.eval()   # keep BatchNorm in eval mode when frozen
        model.backbone.eval()

        ep_losses = []
        for _ in range(max(1, n_train // t_size)):
            t_start = np.random.randint(0, max(1, n_train - t_size))
            t_end   = min(t_start + t_size, n_train)

            probs = model(data, t_start=t_start, t_end=t_end)
            tgts  = get_batch_targets(
                patch, graphs, targets, t_start, t_end, device
            )
            sw   = get_sample_weights(tgts, w_neg, w_pos)
            loss = focal(probs.flatten(), tgts.flatten(), sw)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.get_adaptive_params(),
                max_norm = cfg.train['clip_grad'],
            )
            opt.step()
            ep_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            probs_val = model(
                data, t_start=TRAIN_END, t_end=VAL_END
            )
            tgts_val = get_batch_targets(
                patch, graphs, targets, TRAIN_END, VAL_END, device
            )
            sw_val  = get_sample_weights(tgts_val, w_neg, w_pos)
            val_loss = focal(
                probs_val.flatten(), tgts_val.flatten(), sw_val
            ).item()
            metrics = compute_all_metrics(
                probs_val.flatten(), tgts_val.flatten()
            )

        if val_loss < best_val:
            best_val   = val_loss
            patience   = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience += 1

        log = dict(
            epoch      = epoch + 1,
            patch      = patch,
            train_loss = float(np.mean(ep_losses)),
            val_loss   = float(val_loss),
            val_auc    = float(metrics['auc_roc']),
            val_auc_pr = float(metrics['auc_pr']),
            val_bss    = float(metrics['bss']),
            patience   = patience,
        )
        history.append(log)

        patience_limit = cfg.train.get('finetune_patience', 10)
        if patience >= patience_limit:
            print(f"  Early stopping at epoch {epoch + 1}",flush=True)
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2: calibrate on validation period
    model = _calibrate(
        model, patch, graphs, targets, device
    )

    print(
        f"  Done  val_loss={best_val:.4f}  "
        f"val_auc={history[-1]['val_auc']:.4f}  "
        f"T={model.calibration.temperature:.3f}",flush=True
    )

    return model, history


# Evaluation

def evaluate(model, patch, graphs, targets, device,
             t_start=None, t_end=None):
    """
    Evaluate model on one patch for a time window.
    Defaults to test period (VAL_END → TEST_END).

    Returns: metrics dict from compute_all_metrics
    """
    t_s = t_start if t_start is not None else VAL_END
    t_e = t_end   if t_end   is not None else TEST_END

    data = graphs[patch].to(device)
    model.eval()

    with torch.no_grad():
        probs = model(data, t_start=t_s, t_end=t_e)
        tgts  = get_batch_targets(
            patch, graphs, targets, t_s, t_e, device
        )

    metrics         = compute_all_metrics(
        probs.flatten(), tgts.flatten()
    )
    metrics['patch'] = patch
    metrics['t_start'] = t_s
    metrics['t_end']   = t_e

    return metrics, probs, tgts


def evaluate_all_patches(model, graphs, targets, device,
                         t_start=None, t_end=None):
    """
    Evaluate on all available patches.
    Returns list of metric dicts.
    """
    all_metrics = []
    for patch in PATCHES:
        if patch not in graphs:
            continue
        metrics, _, _ = evaluate(
            model, patch, graphs, targets, device,
            t_start=t_start, t_end=t_end,
        )
        all_metrics.append(metrics)
        print(
            f"  {patch:<25} "
            f"auc_roc={metrics['auc_roc']:.4f}  "
            f"auc_pr={metrics['auc_pr']:.4f}  "
            f"bss={metrics['bss']:.4f}  "
            f"pos_rate={metrics['pos_rate']:.3f}",flush=True
        )
    return all_metrics


# Checkpoint utilities

def save_checkpoint(model, history, output_dir, name):
    """Save model state dict and training history."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        out / f"{name}.pt",
    )
    with open(out / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {out}/{name}.pt",flush=True)


def load_checkpoint(model, checkpoint_path, device):
    """Load model state dict from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded: {checkpoint_path}",flush=True)
    return model


# Private helpers

def _validate(model, graphs, targets, class_weights,
              focal, device, t_start, t_end):
    """
    Run validation on all available patches.
    Returns dict with mean_val_loss and mean_val_auc.
    """
    model.eval()
    val_losses, val_aucs = [], []

    with torch.no_grad():
        for patch in PATCHES:
            if patch not in graphs:
                continue
            data  = graphs[patch].to(device)
            probs = model(data, t_start=t_start, t_end=t_end)
            tgts  = get_batch_targets(
                patch, graphs, targets, t_start, t_end, device
            )
            w_neg, w_pos = class_weights.get(patch, (1.0, 1.0))
            sw   = get_sample_weights(tgts, w_neg, w_pos)
            vl   = focal(probs.flatten(), tgts.flatten(), sw).item()
            metrics = compute_all_metrics(
                probs.flatten(), tgts.flatten()
            )
            val_losses.append(vl)
            if not np.isnan(metrics['auc_roc']):
                val_aucs.append(metrics['auc_roc'])

    return {
        'mean_val_loss': float(np.mean(val_losses)),
        'mean_val_auc':  float(np.mean(val_aucs)) if val_aucs
                         else float('nan'),
        'per_patch':     val_losses,
    }


def _calibrate(model, patch, graphs, targets, device):
    """
    Phase 2 fine-tuning: fit calibration temperature on val period.
    Only CalibrationLayer updates — head and edge scaler are frozen.
    """
    data  = graphs[patch].to(device)
    focal = FocalLoss(gamma=2.0, alpha=0.25)

    opt = AdamW(
        model.calibration.parameters(),
        lr=1e-2,
    )

    model.eval()
    # Keep calibration layer in train mode for parameter update
    model.calibration.train()

    for _ in range(20):  # short calibration loop
        with torch.no_grad():
            logits = model(
                data,
                t_start      = TRAIN_END,
                t_end        = VAL_END,
                return_logits = True,
            )
            tgts = get_batch_targets(
                patch, graphs, targets, TRAIN_END, VAL_END, device
            )

        probs = model.calibration(logits.flatten().detach())
        loss  = focal(probs, tgts.flatten())

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.calibration.eval()
    return model


def _discriminator_accuracy(model, disc, graphs,
                             train_patches, device, t_step=0):
    """
    Compute patch discriminator accuracy without gradient reversal.
    Target: ~1/n_patches ≈ 0.083 (random chance).
    Used for monitoring domain invariance during pre-training.
    """
    model.eval()
    disc.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for pi, patch in enumerate(train_patches):
            data = graphs[patch].to(device)
            geo  = model.encoder(data.x_static, data.x_prior)
            ea   = model.edge_weight_scaler(data.edge_attr)
            emb  = model.backbone(
                data.x_temporal[:, t_step:t_step+1, :],
                geo, data.edge_index, ea,
                n_steps=1, return_all_steps=False,
            )
            logits = disc.classifier(emb)
            preds  = logits.argmax(dim=-1)
            labels = torch.full(
                (emb.shape[0],), pi % len(PATCHES),
                dtype=torch.long, device=device,
            )
            correct += (preds == labels).sum().item()
            total   += emb.shape[0]

    model.train()
    disc.train()
    return correct / total if total > 0 else float('nan')
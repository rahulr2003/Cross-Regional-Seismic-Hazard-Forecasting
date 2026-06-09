# src/model.py
# Full seismic hazard GNN architecture — all components in one file.
#
# Component hierarchy:
#
#   FROZEN (pre-trained globally, never updated on F or P):
#       GeologicalEncoder     static features + GMM prior → geo embedding
#       TemporalGNNBackbone   temporal features over graph → node embedding
#
#   ADAPTIVE (fine-tuned on patch F, 3,141 parameters total):
#       EdgeWeightScaler      learns spatial vs temporal edge importance (2 params)
#       PredictionHead        node embedding → hazard probability (3,138 params)
#       CalibrationLayer      Platt scaling temperature (1 param)
#
#   FULL SYSTEM:
#       SeismicHazardGNN      wires all components, manages freeze/unfreeze

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


# 1. Geological Encoder (FROZEN)

class GeologicalEncoder(nn.Module):
    """
    Maps static geological features + frozen GMM prior probabilities
    to a geological context embedding.

    Input:
        x_static (N, 12) — normalised static features
        x_prior  (N, 6)  — GMM cluster probabilities
    Output:
        embedding (N, embed_dim)

    FROZEN after global pre-training.
    Encodes geological regime — a property of geology, not seismic history.
    Never updated during fine-tuning on new patches.
    """
    def __init__(self, static_dim=12, prior_dim=6,
                 hidden_dims=None, embed_dim=32):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.embed_dim = embed_dim
        layers, in_dim = [], static_dim + prior_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.1),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_static, x_prior):
        # Replace NaN with neutral values before encoding
        x_static = torch.nan_to_num(x_static, nan=0.0)
        x_prior  = torch.nan_to_num(x_prior,  nan=1.0/6)  # uniform prior
        x        = torch.cat([x_static, x_prior], dim=-1)
        return self.mlp(x)

    def freeze(self):
        """Freeze all parameters after pre-training."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        """Unfreeze for pre-training or ablation studies."""
        for p in self.parameters():
            p.requires_grad = True
        self.train()


# 2. Temporal GNN Backbone (FROZEN)

class TemporalGNNBackbone(nn.Module):
    """
    Processes temporal seismic features over the patch graph,
    conditioned on the geological encoder embedding at each time step.

    Architecture per time step:
        [temporal_feat | geo_embed] → input_proj → hidden_dim
        GraphTransformer layer 1 (with residual)
        GraphTransformer layer 2 (with residual)
        GRU cell (propagates temporal state)
    Output projection: node_embed_dim

    Key design choices:
        - aggr='add'  in TransformerConv for deterministic aggregation
        - Learnable h0 broadcast to all nodes for equivariant init
        - geo_embed concatenated at every time step (frozen conditioning)
        - nan_to_num on temporal features (many features are sparse/NaN)

    FROZEN after global pre-training.
    """
    def __init__(self, temporal_dim=17, geo_embed_dim=32,
                 hidden_dim=128, node_embed_dim=64,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.node_embed_dim = node_embed_dim

        # Input projection: temporal + geo → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(temporal_dim + geo_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Graph Transformer layer 1
        self.conv1 = TransformerConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // n_heads,
            heads        = n_heads,
            dropout      = dropout,
            edge_dim     = 2,
            concat       = True,
            aggr         = 'add',
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Graph Transformer layer 2
        self.conv2 = TransformerConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // n_heads,
            heads        = n_heads,
            dropout      = dropout,
            edge_dim     = 2,
            concat       = True,
            aggr         = 'add',
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # GRU for temporal state propagation
        self.gru = nn.GRUCell(
            input_size  = hidden_dim,
            hidden_size = node_embed_dim,
        )

        # Learnable initial hidden state — broadcast to all nodes
        # Ensures permutation-equivariant initialisation
        self.h0 = nn.Parameter(torch.zeros(1, node_embed_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.LayerNorm(node_embed_dim),
            nn.GELU(),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x_temporal, geo_embed, edge_index, edge_attr,
                n_steps=None, return_all_steps=False):
        """
        x_temporal:       (N, T, temporal_dim)
        geo_embed:        (N, geo_embed_dim)   — frozen encoder output
        edge_index:       (2, E)
        edge_attr:        (E, 2)               — [weight, edge_type]
        n_steps:          int | None           — process first n_steps only
        return_all_steps: bool

        Returns:
            (N, node_embed_dim)        if return_all_steps=False
            (N, T, node_embed_dim)     if return_all_steps=True
        """
        N, T, _ = x_temporal.shape
        T       = min(T, n_steps) if n_steps else T

        # Replace NaN with 0 — sparse features treated as absent
        x_t = torch.nan_to_num(x_temporal, nan=0.0)

        # Initialise GRU state — same for all nodes (equivariant)
        h = self.h0.expand(N, -1).contiguous()

        steps = []
        for t in range(T):
            # Concatenate temporal features with geological context
            x_in  = torch.cat([x_t[:, t, :], geo_embed], dim=-1)
            x_h   = self.input_proj(x_in)

            # Graph Transformer with residual connections
            x_c1  = self.norm1(
                self.conv1(x_h, edge_index, edge_attr) + x_h
            )
            x_c1  = self.drop(x_c1)
            x_c2  = self.norm2(
                self.conv2(x_c1, edge_index, edge_attr) + x_c1
            )
            x_c2  = self.drop(x_c2)

            # Temporal state update
            h = self.gru(x_c2, h)

            if return_all_steps:
                steps.append(h.unsqueeze(1))

        if return_all_steps:
            all_h = torch.cat(steps, dim=1)        # (N, T, D)
            flat  = all_h.reshape(-1, self.node_embed_dim)
            return self.output_proj(flat).reshape(N, T, self.node_embed_dim)

        return self.output_proj(h)                 # (N, D)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()


# 3. Edge Weight Scaler (ADAPTIVE)

class EdgeWeightScaler(nn.Module):
    """
    Learns the relative importance of spatial vs temporal edges
    during message passing.

    Two scalar parameters:
        spatial_weight  — scales inverse-distance spatial edges
        temporal_weight — scales Pearson correlation temporal edges

    ADAPTIVE: fine-tuned on each new patch F.
    Allows the model to learn that in sparse patches (e.g. Kutch)
    spatial edges matter more because temporal edges are absent,
    while in dense patches temporal correlation edges carry more signal.
    """
    def __init__(self):
        super().__init__()
        self.spatial_weight  = nn.Parameter(torch.tensor(1.0))
        self.temporal_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, edge_attr):
        """
        edge_attr: (E, 2) — [weight, edge_type]
                   edge_type: 0=spatial, 1=temporal
        Returns:   (E, 2) — weight column scaled by learned type weight
        """
        weights      = torch.stack([self.spatial_weight,
                                     self.temporal_weight])
        edge_type    = edge_attr[:, 1].long()
        scale        = weights[edge_type]           # (E,)
        scaled       = edge_attr.clone()
        scaled[:, 0] = edge_attr[:, 0] * scale
        return scaled


# 4. Prediction Head (ADAPTIVE

class PredictionHead(nn.Module):
    """
    Maps [node_embedding | geo_embedding] → hazard probability.

    Includes a learnable log_base_rate that absorbs the patch-specific
    background seismicity rate, allowing the MLP weights to focus on
    relative spatial and temporal risk rather than absolute probability.

    Initialised at log_base_rate = -3.0 → base prob ≈ 0.047,
    which is near the positive rate of the active patches.

    ADAPTIVE: fine-tuned on each new patch F.
    """
    def __init__(self, node_embed_dim=64, geo_embed_dim=32,
                 hidden_dim=32, dropout=0.3):
        super().__init__()

        self.log_base_rate = nn.Parameter(torch.tensor(-3.0))

        self.mlp = nn.Sequential(
            nn.Linear(node_embed_dim + geo_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embed, geo_embed):
        """Returns probabilities in [0, 1]."""
        return torch.sigmoid(self.get_logits(node_embed, geo_embed))

    def get_logits(self, node_embed, geo_embed):
        """Returns raw logits (before sigmoid) — used for calibration."""
        x = torch.cat([node_embed, geo_embed], dim=-1)
        return self.mlp(x).squeeze(-1) + self.log_base_rate

    @property
    def base_prob(self):
        """Current base probability from log_base_rate."""
        return torch.sigmoid(self.log_base_rate).item()


# 5. Calibration Layer (ADAPTIVE)

class CalibrationLayer(nn.Module):
    """
    Platt scaling with a single learnable temperature parameter.

    Calibrated probabilities = sigmoid(logits / T)

    T > 1: softer probabilities (model was overconfident)
    T < 1: sharper probabilities (model was underconfident)
    T = 1: no calibration (identity)

    Temperature constrained > 0 via softplus parameterisation.
    Fine-tuned on validation portion of patch F.

    ADAPTIVE: 1 parameter total.
    """
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        """
        logits: (N,) or (N, T) raw logits before sigmoid
        Returns: calibrated probabilities, same shape
        """
        T = F.softplus(self.log_temperature) + 1e-6
        return torch.sigmoid(logits / T)

    @property
    def temperature(self):
        """Current temperature value."""
        return (F.softplus(self.log_temperature) + 1e-6).item()


# 6. Full System

class SeismicHazardGNN(nn.Module):
    """
    Full seismic hazard prediction system.

    Forward pass:
        1. Encode geological context             [FROZEN]
        2. Scale edge weights by type            [ADAPTIVE]
        3. Process temporal dynamics over graph  [FROZEN]
        4. Predict hazard probability            [ADAPTIVE]
        5. Calibrate output probabilities        [ADAPTIVE]

    Pre-training mode  (unfreeze_all):
        All 197,669 parameters are trainable.
        Uses adversarial + contrastive auxiliary losses.

    Fine-tuning mode   (freeze_backbone):
        Only 3,141 adaptive parameters update.
        Backbone frozen — transfers pre-trained dynamics.

    Parameter breakdown:
        Frozen:   GeologicalEncoder  13,152
                  TemporalGNNBackbone 181,376
                  ─────────────────  194,528  (98.4%)

        Adaptive: EdgeWeightScaler        2
                  PredictionHead      3,138
                  CalibrationLayer        1
                  ─────────────────    3,141   (1.6%)
    """
    def __init__(self, static_dim=12, prior_dim=6, temporal_dim=17,
                 geo_embed_dim=32, hidden_dim=128, node_embed_dim=64,
                 n_heads=4, dropout=0.1):
        super().__init__()

        # Frozen components
        self.encoder  = GeologicalEncoder(
            static_dim, prior_dim, [128, 64], geo_embed_dim
        )
        self.backbone = TemporalGNNBackbone(
            temporal_dim, geo_embed_dim, hidden_dim,
            node_embed_dim, n_heads, dropout
        )

        # Adaptive components
        self.edge_weight_scaler = EdgeWeightScaler()
        self.head               = PredictionHead(
            node_embed_dim, geo_embed_dim, 32, dropout
        )
        self.calibration        = CalibrationLayer()

    def forward(self, data, t_start=None, t_end=None,
                return_logits=False):
        """
        data:          PyG Data object for one patch
        t_start:       int | None — start time step (inclusive)
        t_end:         int | None — end time step (exclusive)
        return_logits: bool — return raw logits instead of probabilities

        Returns: (N, T) tensor — probabilities or logits
        """
        T_full  = data.x_temporal.shape[1]
        t_s     = t_start if t_start is not None else 0
        t_e     = t_end   if t_end   is not None else T_full
        n_steps = t_e - t_s

        # Step 1 — geological context (frozen)
        geo = self.encoder(data.x_static, data.x_prior)  # (N, geo_dim)

        # Step 2 — scale edge weights (adaptive)
        ea  = self.edge_weight_scaler(data.edge_attr)     # (E, 2)

        # Step 3 — temporal dynamics over graph (frozen)
        embeds = self.backbone(
            data.x_temporal[:, t_s:t_e, :],
            geo, data.edge_index, ea,
            n_steps=n_steps, return_all_steps=True,
        )  # (N, T, node_embed_dim)

        # Step 4 — predict (adaptive)
        N, T, D  = embeds.shape
        geo_exp  = geo.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1)
        logits   = self.head.get_logits(
            embeds.reshape(N * T, D), geo_exp
        )  # (N*T,)

        if return_logits:
            return logits.reshape(N, T)

        # Step 5 — calibrate (adaptive)
        return self.calibration(logits).reshape(N, T)

    # Mode switching

    def freeze_backbone(self):
        """
        Switch to fine-tuning mode.
        Freezes encoder + backbone. Only adaptive components update.
        Call before fine-tuning on a new patch F.
        """
        self.encoder.freeze()
        self.backbone.freeze()

    def unfreeze_all(self):
        """
        Switch to pre-training mode.
        All parameters are trainable.
        Call before global pre-training.
        """
        self.encoder.unfreeze()
        self.backbone.unfreeze()
        for p in self.parameters():
            p.requires_grad = True

    # Parameter utilities

    def get_adaptive_params(self):
        """
        Return only adaptive component parameters.
        Used to construct the fine-tuning optimiser.
        """
        return (
            list(self.edge_weight_scaler.parameters()) +
            list(self.head.parameters()) +
            list(self.calibration.parameters())
        )

    def param_counts(self):
        """Return total, trainable, and frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            'total':     total,
            'trainable': trainable,
            'frozen':    total - trainable,
        }

    def print_param_summary(self):
        """Print parameter breakdown by component."""
        components = {
            'GeologicalEncoder  [FROZEN]':   self.encoder,
            'TemporalGNNBackbone [FROZEN]':  self.backbone,
            'EdgeWeightScaler  [ADAPTIVE]':  self.edge_weight_scaler,
            'PredictionHead    [ADAPTIVE]':  self.head,
            'CalibrationLayer  [ADAPTIVE]':  self.calibration,
        }
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'Component':<40} {'Params':>10} {'Trainable':>10}")
        print("-" * 62)
        for name, module in components.items():
            n      = sum(p.numel() for p in module.parameters())
            t      = sum(
                p.numel() for p in module.parameters()
                if p.requires_grad
            )
            print(f"  {name:<38} {n:>10,} {t:>10,}")
        print("-" * 62)
        print(f"  {'TOTAL':<38} {total:>10,}")

    def gradient_audit(self):
        """
        Verify frozen/adaptive gradient split.
        Call after freeze_backbone() and a backward pass.
        Returns True if audit passes.
        """
        frozen_with_grad = [
            name for name, p in self.named_parameters()
            if not p.requires_grad and p.grad is not None
        ]
        adaptive_no_grad = [
            name for name, p in self.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        ok = (len(frozen_with_grad) == 0)
        if not ok:
            print(f"  AUDIT FAIL: frozen params with gradients: "
                  f"{frozen_with_grad}")
        if adaptive_no_grad:
            print(f"  WARNING: adaptive params without gradients: "
                  f"{adaptive_no_grad}")
        return ok

    @classmethod
    def from_config(cls, cfg):
        """Instantiate from a config dictionary."""
        return cls(
            static_dim     = cfg.get('static_dim',     12),
            prior_dim      = cfg.get('prior_dim',       6),
            temporal_dim   = cfg.get('temporal_dim',   17),
            geo_embed_dim  = cfg.get('geo_embed_dim',  32),
            hidden_dim     = cfg.get('hidden_dim',    128),
            node_embed_dim = cfg.get('node_embed_dim',  64),
            n_heads        = cfg.get('n_heads',          4),
            dropout        = cfg.get('dropout',        0.1),
        )
# src/losses.py
# Loss functions for seismic hazard GNN training.
#
# Three components used during pre-training:
#   1. FocalLoss - primary prediction loss, handles class imbalance
#   2. PatchDiscriminator — adversarial patch confusion (domain invariance)
#   3. ContrastiveGeologicalLoss — geological consistency (cluster structure)
#
# Combined loss:
#   L = L_focal + lambda_adv * L_adversarial + lambda_con * L_contrastive

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Focal Loss

class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary classification.

        FL(p) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 down-weights easy negatives — the vast majority of
    cell-time-step combinations in sparse patches.
    alpha=0.25 down-weights the negative class contribution.

    Supports optional per-sample weights for additional
    patch-level class balancing on top of focal weighting.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, probs, targets, weights=None):
        """
        probs:   (N,) predicted probabilities in [0, 1]
        targets: (N,) binary targets {0, 1}
        weights: (N,) optional per-sample weights

        Returns: scalar loss
        """
        probs   = torch.clamp(probs, 1e-7, 1 - 1e-7)
        targets = targets.float()

        # p_t: probability of the true class
        p_t     = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t: per-class alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: down-weights easy examples
        focal_w = alpha_t * (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = -(
            targets * torch.log(probs) +
            (1 - targets) * torch.log(1 - probs)
        )

        loss = focal_w * bce

        if weights is not None:
            loss = loss * weights

        return loss.mean()


# 2. Gradient Reversal + Patch Discriminator

class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training.

    Forward pass:  identity (x → x)
    Backward pass: negate gradients scaled by alpha (-alpha * grad)

    The backbone is trained to minimise prediction loss while
    simultaneously fooling the patch discriminator via this layer.
    When the discriminator tries to identify which patch a node
    embedding came from, the gradient reversal pushes the backbone
    to produce patch-invariant (regime-invariant) representations.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class PatchDiscriminator(nn.Module):
    """
    Classifies which patch a node embedding came from.
    Trained adversarially against the backbone.

    Target behaviour after successful training:
        Accuracy → 1/n_patches ≈ 8.3% (random chance for 12 patches)

    If accuracy stays above ~25%, the backbone has not learned
    sufficiently patch-invariant representations — increase lambda_adv.

    Architecture: simple 2-layer MLP (no BatchNorm — we want it
    to be a weak classifier so the backbone can fool it easily).
    """
    def __init__(self, embed_dim=64, n_patches=12, hidden_dim=32):
        super().__init__()
        self.n_patches  = n_patches
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_patches),
        )

    def forward(self, node_embed, alpha=1.0):
        """
        node_embed: (N, embed_dim) — backbone output
        alpha:      float — gradient reversal strength (annealed during training)

        Returns: (N, n_patches) logits
        """
        x = GradientReversal.apply(node_embed, alpha)
        return self.classifier(x)

    def accuracy(self, node_embed, patch_labels, alpha=0.0):
        """
        Compute discriminator accuracy (no gradient reversal).
        Used for monitoring — target is ~1/n_patches.
        """
        with torch.no_grad():
            logits = self.classifier(node_embed)
            preds  = logits.argmax(dim=-1)
            return (preds == patch_labels).float().mean().item()


# 3. Contrastive Geological Consistency Loss

class ContrastiveGeologicalLoss(nn.Module):
    """
    SupCon-style contrastive loss on GMM geological cluster assignments.

    Pulls together embeddings from cells in the same GMM cluster.
    Pushes apart embeddings from cells in different clusters.

    This encourages the backbone to encode geological regime structure
    in a way that is consistent with the frozen prior — a key requirement
    for the prior to be useful at transfer time.

    Numerically stable implementation:
        - Log-sum-exp stabilisation prevents overflow/underflow
        - Diagonal masked to 0.0 before multiplication
          prevents 0 * -inf = NaN (IEEE 754 issue)

    Temperature=0.07 following SimCLR/SupCon conventions.
    Subsampling to max_samples prevents O(N^2) memory cost on
    large patches (Sumatra has 1,246 valid cells).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, node_embed, cluster_labels, max_samples=256):
        """
        node_embed:     (N, embed_dim) — backbone output
        cluster_labels: (N,) integer GMM cluster assignments
        max_samples:    int — subsample for memory efficiency

        Returns: scalar loss (0.0 if no positive pairs exist)
        """
        N = node_embed.shape[0]

        # Subsample for memory efficiency
        if N > max_samples:
            idx            = torch.randperm(N, device=node_embed.device)
            idx            = idx[:max_samples]
            node_embed     = node_embed[idx]
            cluster_labels = cluster_labels[idx]
            N              = max_samples

        # L2 normalise embeddings onto unit hypersphere
        z = F.normalize(node_embed, dim=-1)  # (N, D)

        # Scaled pairwise cosine similarity matrix
        sim       = torch.mm(z, z.T) / self.temperature  # (N, N)
        diag_mask = torch.eye(N, device=z.device).bool()

        # Positive pair mask: same cluster, excluding self-similarity
        labels_eq = (
            cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1)
        ).float().masked_fill(diag_mask, 0.0)  # (N, N)

        # Numerically stable log-sum-exp over non-self pairs
        sim_masked  = sim.masked_fill(diag_mask, float('-inf'))
        sim_max     = sim_masked.max(dim=1, keepdim=True).values.clamp(-1e9)
        exp_sim     = torch.exp(sim_masked - sim_max)
        log_sum_exp = (
            torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8) + sim_max
        )

        # Log probability of each pair
        # Mask diagonal to 0.0 BEFORE multiplication
        # Critical: prevents 0 * -inf = NaN in IEEE 754
        log_prob = (sim_masked - log_sum_exp).masked_fill(diag_mask, 0.0)

        # Average negative log-prob over positive pairs
        n_pos = labels_eq.sum(dim=1)
        has_p = n_pos > 0

        if has_p.sum() == 0:
            # No positive pairs — return zero loss with gradient
            return torch.tensor(
                0.0, device=z.device, requires_grad=True
            )

        loss = -(labels_eq * log_prob).sum(dim=1)
        return (loss[has_p] / (n_pos[has_p] + 1e-8)).mean()

class ContrastiveGeologicalLoss(nn.Module): # this is actually ClusterAlignmentLoss
    """
    Directly minimises within-cluster variance of backbone embeddings.
    Simpler than SupCon — directly targets what we want.
    Same-cluster embeddings should be similar in embedding space.
    """
    def __init__(self):
        super().__init__()

    def forward(self, node_embed, cluster_labels, n_clusters=6,temperature=0.5):
        loss     = torch.tensor(0.0, device=node_embed.device,
                                requires_grad=True)
        n_active = 0

        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.sum() < 5:
                continue
            emb_c  = node_embed[mask]
            centre = emb_c.mean(dim=0, keepdim=True).detach()
            loss   = loss + ((emb_c - centre) ** 2).mean()
            n_active += 1

        if n_active == 0:
            return torch.tensor(
                0.0, device=node_embed.device, requires_grad=True
            )
        return loss / n_active


# Keep ContrastiveGeologicalLoss for reference but use
# ClusterAlignmentLoss as the default
# ContrastiveGeologicalLoss = ClusterAlignmentLoss

# Combined loss helper

def compute_combined_loss(probs, targets, sample_weights,
                          node_embed, disc_logits, patch_labels,
                          cluster_labels,
                          focal, contrastive,
                          lambda_adv=0.1, lambda_con=0.05):
    """
    Compute the three-component combined training loss.

    L = L_focal + lambda_adv * L_adv + lambda_con * L_con

    Parameters
    ----------
    probs:          (N*T,) predicted probabilities
    targets:        (N*T,) binary targets
    sample_weights: (N*T,) per-sample class weights
    node_embed:     (N, D) backbone node embeddings
    disc_logits:    (N, n_patches) discriminator output
    patch_labels:   (N,) integer patch ID labels
    cluster_labels: (N,) integer GMM cluster labels
    focal:          FocalLoss instance
    contrastive:    ContrastiveGeologicalLoss instance
    lambda_adv:     float adversarial loss weight
    lambda_con:     float contrastive loss weight

    Returns
    -------
    loss:       scalar total loss
    loss_dict:  dict with individual components for logging
    """
    loss_focal = focal(probs, targets, sample_weights)
    loss_adv   = F.cross_entropy(disc_logits, patch_labels)
    loss_con   = contrastive(node_embed, cluster_labels)

    # Guard against NaN in contrastive (e.g. all-different clusters)
    if torch.isnan(loss_con):
        loss_con = torch.tensor(0.0, device=node_embed.device)

    loss = loss_focal + lambda_adv * loss_adv + lambda_con * loss_con

    return loss, {
        'focal': loss_focal.item(),
        'adv':   loss_adv.item(),
        'con':   loss_con.item(),
        'total': loss.item(),
    }
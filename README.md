# Cross-Regional Seismic Hazard Forecasting with Frozen Geological Prior GNNs

[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org)
[![Zenodo](https://img.shields.io/badge/Zenodo-pending-blue.svg)](https://zenodo.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

Official code for the paper:

> **Frozen Geological Prior Graph Neural Networks for Cross-Regional Seismic Hazard Forecasting**
> Rahul Ravi, University of Nottingham
> *Computers & Geosciences* (under review) | arXiv: pending

---

## Overview

This repository contains the full implementation of the frozen geological prior GNN architecture, the GeoSeisML-12 benchmark dataset loaders, and the Monte Carlo Patch Cycling (MCPC) evaluation framework for cross-regional seismic hazard forecasting.

**The problem:** Machine learning models for seismic hazard forecasting are trained and evaluated within single geographic regions, encoding region-specific patterns that do not generalise to new tectonic settings. Data-sparse hazardous regions, where reliable hazard estimates are most needed, lack sufficient local catalogs for region-specific model training.

**Our approach:** A frozen geological prior GNN that explicitly disentangles static geological structure from dynamic seismicity patterns. By freezing 98.4% of model parameters after global pre-training, only 3,141 adaptive parameters are updated during transfer to a new region — enabling robust cross-regional adaptation with negligible performance loss.

**Key result:** Mean transfer degradation of 0.20% relative to in-distribution performance across 1,299 randomised MCPC evaluations spanning 8 tectonic regimes on 4 continents.

---

## GeoSeisML-12 Dataset

GeoSeisML-12 is the first multi-regime ML-ready seismic hazard benchmark, comprising:

- **12 patches** × 300×300 km across **8 tectonic regimes** on 4 continents
- **7 static geological layers** → 12 features per grid cell (Vs30, crustal thickness, fault density, heat flow, terrain, stress orientation)
- **17 temporal seismicity features** × 300 monthly timesteps (2000–2024)
- **20 binary target definitions** (5 radii × 4 Mw thresholds)
- **Primary target:** Mw ≥ 3.0 within 50 km within 30 days
- Heterogeneous graph objects with embedded chronological train/val/test splits
- Leakage-free GMM-PCA frozen geological prior (k=6 clusters)

**Download:** [Zenodo DOI pending]

### Patches

| ID | Patch | Country | Regime | Pos. Rate |
|----|-------|---------|--------|-----------|
| P01 | Kanto | Japan | Subduction interface | 21.3% |
| P02 | Tohoku | Japan | Outer-rise / trench | 18.9% |
| P03 | Central Chile | Chile | Megathrust | 12.8% |
| P04 | Central Turkey | Turkey | Strike-slip (EAF) | 1.5% |
| P05 | Central Nepal | Nepal | Continental collision | 1.7% |
| P06 | N. Island NZ | New Zealand | Subduction + transform | 13.8% |
| P07 | S. Sumatra | Indonesia | Complex multi-plate | 8.1% |
| P08 | Kutch | India | Intraplate reactivated | 0.6% |
| P09 | Longmenshan | China | Thrust belt | 3.8% |
| P10 | W. Australia | Australia | Stable craton (Archean) | 0.3% |
| P11 | S. Norway | Norway | Post-glacial rebound | 0.05% |
| P12 | Ordos | China | Stable craton (N. China) | 0.7% |

---

## Architecture

```
Static Geological Features (N×12) ──┐
                                     ├──► Geological Encoder [FROZEN · 13,152 params]
Geological Prior Probabilities (N×6) ┘         │
                                            z_geo (N×32)
                                                │ injected at every timestep
Temporal Seismicity Features (N×T×17) ──► Edge Weight Scaler [ADAPTIVE · 2 params]
                                                │
                                                ▼
                                     Temporal GNN Backbone [FROZEN · 181,376 params]
                                     Graph Transformer ×2 + GRU
                                                │
                                         node_embed (N×64)
                                                │
                                     concat(node_embed, z_geo) → (N×96)
                                                │
                                     Prediction Head [ADAPTIVE · 3,138 params]
                                                │
                                     Calibration Layer [ADAPTIVE · 1 param]
                                                │
                                          P(hazard) (N×T)

Total: 197,669 params  |  Frozen: 194,528 (98.4%)  |  Adaptive: 3,141 (1.6%)
```

---

## Installation

```bash
git clone https://github.com/rahulr2003/Cross-Regional-Seismic-Hazard-Forecasting.git
cd Cross-Regional-Seismic-Hazard-Forecasting
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
scipy>=1.10.0
matplotlib>=3.7.0
```

---

## Repository Structure

```
├── src/
│   ├── config.py          # Architecture, training, and target configuration
│   ├── data.py            # Dataset loading, feature engineering, graph construction
│   ├── model.py           # Full GNN architecture (encoder, backbone, head)
│   ├── losses.py          # Focal, contrastive, and adversarial losses
│   └── trainer.py         # Two-phase pre-training procedure
├── train.py               # Pre-training entry point
├── eval_patch_cal.py      # MCPC evaluation with base rate calibration
├── data/
│   ├── graphs/            # PyG Data objects (12 patches)
│   ├── targets/           # Binary target arrays (20 definitions)
│   ├── feature_tensors/   # Static geological feature tensors
│   └── frozen_prior/      # Global GMM-PCA prior components
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Usage

### Pre-training

```bash
python train.py \
    --data_dir data/ \
    --output_dir train_logs/run1/ \
    --n_epochs 250 \
    --hidden_dim 128 \
    --lr 1e-3 \
    --dropout 0.2 \
    --batch_t 18
```

### MCPC Evaluation

```bash
python eval_patch_cal.py \
    --data_dir data/ \
    --checkpoint train_logs/run1/pretrained_seed58.pt \
    --output_dir output_mcpc/seed58/ \
    --n_cycles 100 \
    --n_source 6 \
    --n_predict 3 \
    --min_pos_rate 0.001
```

### Results Analysis

```bash
python analyse_mcpc_results.py \
    --log_file MCPC_Results.txt \
    --output_dir mcpc_analysis/
```

---
## Results

MCPC evaluation across 5 checkpoints × 100 cycles × 3 prediction patches = 1,299 total evaluations:

| Condition | AUC | Degradation |
|-----------|-----|-------------|
| In-distribution (upper bound) | 0.7563 ± 0.0042 | — |
| FP Transfer (ours) | 0.7387 ± 0.0469 | 0.20% |
| Naive Transfer (baseline) | 0.7392 ± 0.0461 | 0.21% |

- **Max degradation:** 1.0% across all evaluated checkpoints
- **Positive gaps (FP > naive):** 3/5 checkpoints
- **Transfer > in-distribution:** 1/5 checkpoints (seed58)

---

## Citation

If you use this code, dataset, or methodology in your research, please cite:

```bibtex
@article{ravi2026geoseisml,
  title   = {Frozen Geological Prior Graph Neural Networks for 
             Cross-Regional Seismic Hazard Forecasting},
  author  = {Ravi, Rahul},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Under review. Preprint: arXiv:pending}
}
```

---

## Data Sources

Static geological layers were assembled from the following open datasets:

| Layer | Source | DOI / URL |
|-------|--------|-----------|
| Vs30 | USGS Global Vs30 Mosaic | [Heath et al., 2020] |
| Crustal & Sediment Thickness | CRUST1.0 | [Laske et al., 2013] |
| Terrain (elevation, slope, roughness) | NASADEM | [Crippen et al., 2016] |
| Heat Flow | IHFC Global Heat Flow Database 2024 | 10.5880/fidgeo.2024.014 |
| Tectonic Stress | World Stress Map 2025 | 10.5880/WSM.2025.001 |
| Active Faults | GEM Global Active Faults | [Styron & Pagani, 2020] |
| Earthquake Catalogs | USGS ComCat | earthquake.usgs.gov |

---

## Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.

---

## Contact

Rahul Ravi — University of Nottingham
GitHub: [@rahulr2003](https://github.com/rahulr2003)

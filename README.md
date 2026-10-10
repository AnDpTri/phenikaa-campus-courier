# 🏆 Top13-Phenikaa-AI-Hackathon-2026

[![Hackathon](https://img.shields.io/badge/Phenikaa%20AI%20Hackathon%202026-Top%2013-gold?style=for-the-badge&logo=target)](https://github.com/AnDpTri)
[![Peak Score](https://img.shields.io/badge/Peak%20Score-71.17%25-brightgreen?style=for-the-badge&logo=speedtest)](https://github.com/AnDpTri)
[![Team](https://img.shields.io/badge/Team-Devil%20May%20Cry-red?style=for-the-badge&logo=playstation)](https://github.com/AnDpTri)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://github.com/AnDpTri)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-orange?style=for-the-badge&logo=pytorch)](https://github.com/AnDpTri)

> **Official Solution by Team Devil May Cry for the Phenikaa Campus Courier 2026 (COURIER2) Challenge.**  
> **Official Public/Private Scores:** **71.17%** (Submission `#1208`) & **70.83%** (Submission `#1201`).

---

## 📖 Overview

This repository contains the complete clean-room, production-grade AI pipeline developed by **Team Devil May Cry** for the **Phenikaa Campus Courier 2026** competition. 

The task requires solving autonomous multimodal multi-robot delivery routing under complex constraints:
* **Inputs:** Raw aerial campus map images + Vietnamese natural language delivery orders.
* **Outputs:** Optimal discrete navigational actions (`0: UP`, `1: DOWN`, `2: LEFT`, `3: RIGHT`) for **10 specialized robot classes** (motorbikes, vans, legged drones, emergency rovers, fragile-cargo carriers, etc.).

---

## 🏛️ System Architecture

The solution follows a strict, decoupled 3-tier architecture with explicit schema contracts:

```
Campus Image ──> [ Computer Vision Pipeline ] ──> SceneGraph
                                                      │
Mission Text ──> [ Hybrid NLP Parser        ] ──> Mission ──> [ Multi-Robot Solver ] ──> Action (0..3)
```

### 1. Computer Vision Pipeline (`src/courier/cv/`)
* **Lattice & Landmark Extraction:** Custom deep detector resolving campus lattice nodes, road types, and landmark bounding regions.
* **Directional Edge Classifier:** Classifies road connectivity, slope/stairs, and one-way directional constraints.
* **Weather Classifier:** Gradient-boosted model identifying weather factors that modulate transit velocity.
* **Hardware Acceleration:** Native PyTorch inference optimized for CUDA GPUs.

### 2. Hybrid Vietnamese NLP Parser (`src/courier/nlp/`)
* **Campus Idiom Normalizer:** Domain-specific spatial grammar resolving intricate Vietnamese expressions (*"nơi tít phía nam nhất"*, *"tòa nhà ở sinh viên"*, *"trạm xá"*, *"điểm chăm sóc sức khỏe"*).
* **Neural BiGRU Semantic Fusion (`neural_parser_v5_v9`):** Multi-task neural network predicting delivery `Goal`, checkpoint `Via`, `Urgent` urgency flags, and `Fragile` handling tags.
* **Map Grounding:** Maps semantic landmarks to graph coordinates extracted from CV.

### 3. Multi-Robot Strategy Solver (`src/courier/solver/`)
* **Per-Robot Strategy Model:** 10 independent machine learning models trained on 10 distinct physical robot specifications.
* **Ultra Ensemble Solver:** Multi-seed HistGradientBoosting ranking candidate graph outgoing edges with strict legal move masking (no one-way violations, no stair climbing for wheeled robots).
* **Personality Tie-Breaking Calibration:** Differentiates ambiguous paths when nominal shortest-path costs are tied.

---

## 📊 Benchmark & Score Progression

| Milestone | Architecture / Configuration | Public Test Score |
| :--- | :--- | :---: |
| **Initial Baseline** | Rule Parser + Uniform Shortest Path | `52.39%` |
| **Stage 1 (V1 Solver)** | Neural CV 512 + Candidate Strategy Ranker | `65.50%` |
| **Stage 2 (NLP V4)** | Fine-tuned 200k Curated Synthetic NLP | `70.39%` |
| **Stage 3 (Ultra Solver)** | 5-Seed Ensemble Solver on Full 2.300 Scenes | `70.83%` |
| **Stage 4 (Peak Submission)** | V5+V9 Ensemble NLP + Learned Strategy Policy | **`71.17%` 🔥** |

---

## 📂 Repository Structure

```text
src/courier/
  common/       # Contracts and dataset I/O (Scene, Mission, SceneGraph)
  cv/           # Vision extraction (Detector, Nets, Classifiers)
  nlp/          # Natural language understanding and spatial resolution
  solver/       # Multi-robot pathfinding and learned action policy
scripts/
  cv/           # CV training and evaluation benchmarks
  nlp/          # NLP synthetic data generators and neural training
  solver/       # Strategy model training and oracle graph solvers
  create_submission.py # Production end-to-end submission generator
artifacts/      # Serialized neural weights (.pt) and ML models (.joblib)
tests/          # Unit and integration test suites
```

---

## 🚀 Getting Started & Reproducibility

### 1. Installation

Python 3.12 is recommended:

```powershell
# Clone the repository
git clone https://github.com/AnDpTri/Top13-Phenikaa-AI-Hackathon-2026.git
cd Top13-Phenikaa-AI-Hackathon-2026

# Install in editable mode
py -3.12 -m pip install -e ".[solver,cv]"
$env:PYTHONPATH = "src"
```

### 2. Run Verification Tests

```powershell
py -3.12 -m unittest discover -s tests -v
```

### 3. Generate Submission Predictions

```powershell
py -3.12 scripts/create_submission.py `
  --detector-device cuda `
  --strategy-artifact artifacts/solver/candidate_strategy_ultra.joblib `
  --nlp-model artifacts/nlp/neural_parser_v5_sf200_scratch_via095.pt `
  --out results/submission_reproduced/predictions.json `
  --diagnostics-dir results/submission_reproduced/diagnostics
```

---

## 📜 License

Licensed under the [MIT License](LICENSE). Developed by **Team Devil May Cry** for the Phenikaa Campus Courier 2026 Challenge.

# Weight Space: Mechanistic Interpretability & Closed-Form Weight Compilation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

An empirical mechanistic interpretability toolkit for discovering monosemantic feature directions using **Top-$K$ Sparse Autoencoders (SAEs)**, analyzing coordinate bases in residual activation space, and permanently compiling latent steering vectors directly into transformer MLP down-projection weights via **closed-form rank-1 updates**.

---

## Key Findings

### 1. Top-$K$ Sparse Autoencoder Feature Isolation
Trained on Layer 12 residual stream activations of `Qwen/Qwen2.5-0.5B-Instruct` ($d_{\text{model}} = 896$, expansion $d_{\text{sae}} = 4096$, $K = 32$), achieving $99.23\%$ activation sparsity. Discovered pure monosemantic features including French language direction (Latent #3460).

### 2. Destructive Interference in Physical Coordinate Projections
Decomposing feature directions into the model's standard physical coordinate basis reveals that steering **only** the top 5 positive coordinates (`#490, #507, #266, #579, #570`) causes the model to collapse into an orthogonal software developer persona (generating C++ and Python code).

Negative coordinates (`#100, #131, #43, #850, #194`) act as inhibitory destructive interference brakes. Balancing 10 positive with 10 negative coordinates fully cancels the software engineering interference and restores fluent French generation.

### 3. Zero-Overhead Closed-Form Weight Compilation
Latent directions can be permanently folded into the MLP down-projection matrix $W_{\text{down}}$ using a closed-form rank-1 outer product:

```math
\Delta W = \alpha \, \mathbf{d} \, \mathbf{u}^T
```

This eliminates runtime forward hooks, introduces zero inference latency, and preserves general capabilities.

---

## Mathematical Formulation

### 1. Top-$K$ Sparse Autoencoder

Given residual representation $\mathbf{x} \in \mathbb{R}^{d}$, the encoder maps centered activations to an overcomplete latent space $\mathbb{R}^{M}$ ($M > d$):

```math
\mathbf{z} = \operatorname{TopK}\left(\operatorname{ReLU}\left(W_{\text{enc}}(\mathbf{x} - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}}\right), K\right)
```

The decoder reconstructs activations:

```math
\hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}
```

subject to unit column norm constraints $\|W_{\text{dec}}[:, i]\|_2 = 1$.

### 2. Physical Coordinate Decomposition

For an isolated feature unit direction $\mathbf{d} = W_{\text{dec}}[:, c] \in \mathbb{R}^{d}$, we decompose its components along the standard physical basis:

```math
\mathbf{d} = \sum_{i=1}^{d} d_i \mathbf{e}_i = \sum_{i \in \mathcal{P}} d_i \mathbf{e}_i + \sum_{j \in \mathcal{N}} d_j \mathbf{e}_j
```

where $\mathcal{P} = \{i \mid d_i > 0\}$ and $\mathcal{N} = \{j \mid d_j < 0\}$. Steering without $\mathcal{N}$ breaks geometric alignment and triggers off-target projections.

### 3. Closed-Form Parameter Compilation

For an MLP down-projection layer $\mathbf{y} = W_{\text{down}} \mathbf{h}_{\text{inter}}$, injecting an output offset proportional to $\mathbf{d}$ corresponds to:

```math
W_{\text{down}}^{\text{new}} = W_{\text{down}} + \alpha \, \mathbf{d} \, \mathbf{u}^T
```

where:

```math
\mathbf{u} = \frac{\mathbf{1}}{\sqrt{d_{\text{inter}}}} \in \mathbb{R}^{d_{\text{inter}}}
```

---

## Repository Structure

```
weight_space/
├── weight_space/              # Core library
│   ├── models/                # TopKSAE architecture
│   │   └── sae.py
│   ├── data/                  # Domain corpora and evaluation prompts
│   │   └── prompts.py
│   ├── extraction.py          # Residual stream hook capture utilities
│   ├── train.py               # SAE training loop with unit-norm projection
│   ├── decomposition.py       # Differential scoring & coordinate decomposition
│   ├── steering.py            # SteeringHook context manager and generation
│   └── weight_edit.py         # Rank-1 parameter compilation & orthogonal ablation
├── experiments/               # Reproducible experiment pipelines
│   ├── balanced_steering.py   # Destructive interference & balanced coordinate tests
│   ├── compile_to_weights.py  # Weight compilation into W_down
│   └── train_and_discover.py  # End-to-end SAE training from scratch
├── scripts/                   # Evaluation harnesses
│   ├── run_pos_neg_remote.py  # GPU / Colab evaluation script
│   └── test_pos_neg_local.py  # Local CPU / MPS sanity test
├── tests/                     # Unit test suite
│   ├── test_sae.py
│   ├── test_steering.py
│   └── test_weight_edit.py
├── docs/                      # Interactive research report
│   └── index.html             # Distill-style article & visualizer
├── results/
│   └── eval_traces.md         # Full generation transcripts and loss traces
├── reproduce.py               # Top-level reproduction CLI
└── pyproject.toml             # Package configuration
```

---

## Installation

```bash
git clone https://github.com/sirohikartik/weight_space.git
cd weight_space
pip install -e .
```

To run the unit tests:
```bash
pytest tests/
```

---

## Usage & Reproduction

### 1. Test Balanced Coordinate Steering & Destructive Interference
Demonstrate how adding negative inhibitory coordinates prevents software engineer persona collapse and recovers French generation:
```bash
python reproduce.py steer --prompt "Tell me about what you do every day."
```

### 2. Compile Latent Feature Directly into Model Weights
Permanently modify $W_{\text{down}}$ of Layer 12 using closed-form rank-1 compilation (no runtime hooks):
```bash
python reproduce.py compile --alpha 2.5
```

### 3. End-to-End Feature Discovery from Scratch
Capture activations, train a Top-$K$ SAE, and isolate monosemantic concept directions:
```bash
python -m experiments.train_and_discover --model Qwen/Qwen2.5-0.5B-Instruct --layer 12 --d_sae 4096 --k 32
```

---

## Interactive Distill Article

The complete interactive article and coordinate visualizer is available in [`docs/index.html`](docs/index.html). It includes:
- Interactive coordinate sliders showing real-time token logit shifts.
- Complete 20-neuron coordinate activation matrix with positive and negative weights.
- Empirical generation traces comparing baseline, positive-only collapse, and balanced recovery.

---

## Citation

```bibtex
@misc{sirohi2026weightspace,
  title={Weight Space: Mechanistic Interpretability, Destructive Interference, and Closed-Form Parameter Compilation},
  author={Sirohi, Kartik},
  year={2026},
  url={https://github.com/sirohikartik/weight_space}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

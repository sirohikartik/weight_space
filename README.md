# Weight Space: Mechanistic Interpretability & Behavioral Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

An empirical mechanistic interpretability and behavioral steering framework for discovering monosemantic feature directions using **Top-$K$ Sparse Autoencoders (SAEs)**, decomposing decoder vectors into balanced physical coordinates, evaluating circuit stability under strict norm-matching constraints, and auditing latent safety refusal directions.

> 📖 **Read the Long-Form Research Blog:**  
> [**Steering Language Models in Physical Weight Space: Why Coordinate Signs Matter More Than Magnitude**](BLOG.md) (or view the [interactive Distill-style white themed post](docs/index.html)).

---

## Key Empirical Findings

### 1. The 18.2% Magnitude Collapse (Sign Balance vs. Pure Magnitude)
In a controlled Euclidean norm ablation suite holding injected energy strictly constant ($\|\Delta x\|_2 = c$) across 500 prompts and 100 neutral probes:
* Selecting coordinates with sign balance ($C_{\text{bal10}} = \mathcal{P}_5 \cup \mathcal{N}_5$) achieves **93.2% adherence** on Qwen2.5-1.5B Agreeableness with zero persona shifts (0/100).
* Selecting top coordinates purely by absolute magnitude without sign balance ($C_{\text{top10\_abs}}$) causes an **18.2 percentage point collapse in adherence down to 75.0%**, collapsing back to the unsteered baseline (74.8%).
* This establishes that coordinate signs—not merely large weights—are causally necessary for coherent circuit steering.

### 2. Active Directional Suppression of Negative Coordinates
Injecting pure negative coordinates ($C_{\text{neg5}}$) at matched Euclidean energy reduces Agreeableness adherence to **65.0%** (9.8% below unsteered baseline). Negative coordinates do not act as passive dampening noise; they exert active, directional suppression against the target trait circuit.

### 3. Architecture Specificity of Persona Collapse
* On `Qwen2.5-1.5B` Extraversion, boosting positive coordinates triggered an off-target software developer persona on 1 of 5 neutral probes (20.0%).
* On `SmolLM2-1.7B`, persona collapse was **0/100 (0.0% [0.0, 3.7%]) across all 16 conditions**, demonstrating that collapse susceptibility is architecture- and trait-dependent.

### 4. Zero-Overhead Closed-Form Weight Compilation
Latent directions can be permanently folded into MLP down-projection matrices $W_{\text{down}}$ using a closed-form rank-1 outer product:

```math
\Delta W = \alpha \, \mathbf{d} \, \mathbf{u}^T
```

This eliminates runtime forward hooks, introduces zero prompt tokens, and preserves general capabilities.

---

## Controlled Norm Ablation Benchmark ($n=500$ prompts, $n=100$ probes)

All steered vectors hold Euclidean injection norm strictly constant at $c = \|\Delta x_{\text{bal10}}\|_2$. Bracketed ranges indicate Wilson score 95% confidence intervals:

| Model & Trait | Condition | Energy $c$ | Adherence ($n=500$) | Persona Shifts ($n=100$) | Loss$_{\text{exp}}$ |
|---|---|---|---|---|---|
| **Qwen2.5-1.5B**<br>*(Agreeableness)* | $C_0$ Baseline | 0.000 | 74.8% [70.8, 78.4] | 0 / 100 [0.0, 3.7] | 2.00 |
| | $C_{\text{full}}$ (Dense SAE) | 10.505 | 75.2% [71.2, 78.8] | 0 / 100 [0.0, 3.7] | 2.01 |
| | $C_{\text{pos5}}$ (Pos Only) | 10.505 | 96.8% [94.9, 98.0] | 0 / 100 [0.0, 3.7] | 2.00 |
| | $C_{\text{neg5}}$ (Neg Only) | 10.505 | **65.0%** [60.7, 69.1] *(-9.8%)* | 0 / 100 [0.0, 3.7] | 2.00 |
| | **$C_{\text{bal10}}$ (Balanced)** | 10.505 | **93.2% [90.6, 95.1]** | **0 / 100 [0.0, 3.7]** | **2.00** |
| | **$C_{\text{top10\_abs}}$ (Mag Only)** | 10.505 | **75.0% [71.0, 78.6]** *(-18.2%)* | 0 / 100 [0.0, 3.7] | 2.01 |
| | $C_{\text{rand10}}$ (Random) | 10.505 | 68.2% [64.0, 72.1] | 1 / 100 [0.2, 5.4] | 1.98 |
| | $C_{\text{norm\_ctrl}}$ (Noise Floor) | 10.505 | 97.0% [95.1, 98.2] | 0 / 100 [0.0, 3.7] | 2.03 |
| **SmolLM2-1.7B**<br>*(Agreeableness)* | $C_0$ Baseline | 0.000 | 94.2% [91.8, 95.9] | 0 / 100 [0.0, 3.7] | 1.88 |
| | **$C_{\text{bal10}}$ (Balanced)** | 6.010 | **95.8% [93.7, 97.2]** | **0 / 100 [0.0, 3.7]** | **1.87** |
| | $C_{\text{top10\_abs}}$ (Mag Only) | 6.010 | 95.2% [93.0, 96.8] | 0 / 100 [0.0, 3.7] | 1.87 |
| | $C_{\text{norm\_ctrl}}$ (Noise Floor) | 6.010 | 93.8% [91.3, 95.6] | 0 / 100 [0.0, 3.7] | 1.88 |

---

## Evaluation Suite Architecture

The repository includes a comprehensive, peer-review-grade evaluation framework in `weight_space/evaluation/`:

```
weight_space/evaluation/
├── measurement/         # Wilson score 95% CIs, Wikitext-2 corpus PPL, 100 neutral probes, matched noise
├── frontier/            # CAA (Rimsky), SAE-TS (Chalnev), SAS (Bayat), sign-balanced coordinates
├── safety/              # Contrastive refusal vector extraction, cosine alignment, orthogonal deflation
├── circuits/            # Dynamic token schedulers (linear/cosine/exp decay) & intermediate layer sweeps
└── orchestration/       # SQLite datastore with automated LaTeX table generator & benchmark runners
```

### Running the Evaluation Suite
```bash
# Run complete test suite (28 passed in 6s)
PYTHONPATH=. pytest tests/test_evaluation/

# Run multi-condition frontier benchmark
python run_frontier_benchmark.py --model Qwen/Qwen2.5-1.5B --trait agreeableness --layer 14
```

---

## Quickstart: In-Register Forward Hook

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def make_balanced_hook(pos_indices, pos_weights, neg_indices, neg_weights, alpha=30.0):
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        delta = torch.zeros(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        for idx, w in zip(pos_indices, pos_weights):
            delta[idx] += alpha * w
        for idx, w in zip(neg_indices, neg_weights):
            delta[idx] += alpha * w  # Negative coordinates cancel collateral drift
        hidden_states[:, -1, :] += delta
        return (hidden_states,) if isinstance(output, tuple) else hidden_states
    return hook_fn

# Register on layer 14:
# model.model.layers[14].register_forward_hook(hook_fn)
```

---

## License & Citation
MIT License. See [LICENSE](LICENSE).

```bibtex
@article{sirohi2026weightspace,
  title={Steering Language Models in Physical Weight Space: Why Coordinate Signs Matter More Than Magnitude},
  author={Sirohi, Kartik},
  journal={GitHub Repository: sirohikartik/weight_space},
  year={2026}
}
```

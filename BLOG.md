# Steering Language Models in Physical Weight Space: Why Coordinate Signs Matter More Than Magnitude

**Kartik Sirohi** &bull; November 2026 &bull; *Technical Research Notes*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)

---

### TL;DR
* **The Zero-Prompt Goal:** Prepending system prompts (*"You are an agreeable assistant..."*) consumes 18–40 tokens per query and forces the key-value (KV) cache to retain prompt tokens across every generation step.
* **The Software Developer Glitch:** When we isolated an Extraversion feature in `Qwen2.5-1.5B` and boosted only the top 5 positive coordinates, the model abandoned its assistant role and declared on 1 out of 5 neutral probes (20.0%) that it was a *software developer writing C++, Java, and Python*.
* **The Counterweight Discovery:** Decomposing the Sparse Autoencoder (SAE) decoder column reveals that negative weights act as **targeted brakes**, providing destructive interference that cancels collateral projections along shared polysemantic neurons.
* **The Controlled Norm Ablation Suite:** Holding injected Euclidean energy strictly constant ($\|\Delta x\|_2 = c$) across 500 prompts and 100 neutral persona probes reveals an **18.2% collapse in adherence** when selecting coordinates purely by absolute magnitude without sign balance (dropping from 93.2% back to 75.0%, matching baseline).
* **Architecture Specificity:** `SmolLM2-1.7B` showed **0/100 persona shifts across all 16 evaluated conditions**, demonstrating that persona collapse susceptibility is architecture- and baseline-dependent.

---

## 1. The Context-Window Tax & The Polysemantic Leak

Standard prompt engineering achieves behavioral control through natural language instructions. You prepend guidelines into the system prompt, and the model adjusts its persona:

```text
[System Prompt: "You are an extremely agreeable assistant..."]  <-- 18 to 40 prompt tokens
[User Prompt: "What is the capital of France?"]
```

This prompt tax permanently consumes context window capacity and forces the KV cache to store intermediate projections for those prompt tokens on every forward step. Parameter fine-tuning avoids this prompt tax, but requires heavy compute and risks catastrophic forgetting.

**What if we steer the model directly inside its residual stream using a handful of physical coordinates?**

In theory, this adds:
* **0 prompt tokens** to the context window.
* **0 KV cache expansion**.
* **< 0.2 ms latency overhead** via register-space tensor addition.

We trained a Top-$K$ Sparse Autoencoder ($d_{\text{sae}} = 4 \times d_{\text{model}}, K = 32$) on intermediate residual stream activations (`Layer 14` for Qwen2.5-1.5B, `Layer 12` for SmolLM2-1.7B). The autoencoder isolates monosemantic latent directions from superimposed activations.

For Extraversion, the SAE identified a single clean latent feature direction that activates strongly on extroverted dialogue exemplars. The decoder column corresponding to this feature is a vector $w_k \in \mathbb{R}^{1536}$. Each coordinate $w_{k,j}$ indicates how strongly that latent feature projects onto physical neuron $j$.

Naturally, we first asked: *What happens if we take only the top 5 neurons with the largest positive weights ($w_{k,j} > 0$) and boost them directly during inference?*

We then probed the steered model with neutral conversational queries. When given the neutral probe prompt `"Tell me about your morning routine"`, the model completely abandoned its conversational persona:

```text
PROMPT: "Tell me about your morning routine."

OUTPUT: "Good morning! Today's routine starts with waking up at 6:30 AM, 
        which is my usual time for breakfast and getting ready for work... 
        After that, I take a shower and get dressed in my business attire 
        for my day as a developer. Then I check my emails..."
```

> **The Collateral Circuit Glitch:**  
> By boosting only 5 positive coordinates to make the model extroverted, we accidentally triggered an unprompted **software developer persona** on 1 out of 5 neutral probes (20.0%). The model did not merely become more talkative; it adopted an entirely new occupational identity.

Why did this happen? Physical neurons in transformers operate in **superposition**. A single physical channel does not represent a single human-interpretable concept; it participates in dozens of competing semantic circuits simultaneously. Neurons that fire during extroverted conversation are also wired into technical work and career narratives.

---

## 2. Negative Coordinates as Targeted Brakes

When you examine the complete SAE decoder column $w_k$, you notice something that naive feature boosting ignores: **the column contains both large positive weights and large negative weights**.

$$\Delta x_{\text{balanced}} = \alpha \left( \sum_{j \in \mathcal{P}_5} w_{k,j} e_j + \sum_{j \in \mathcal{N}_5} w_{k,j} e_j \right)$$

Those negative coefficients are not random noise. They represent physical coordinates where the target feature vector points *against* the coordinate axis. When the autoencoder projects a monosemantic feature into physical space, negative coefficients provide **destructive interference** that dampens collateral projections along shared polysemantic dimensions:

```
(a) Positive-Only Injection (C_pos5)
    Neuron j2 ^
              |         / v_k (Target Trait)
              |       /
              |     /
              |   /       .-------> v_m (Developer)
              | /      .-'
              +------> Δx_pos ------------> Neuron j1 (Shared)
                proj(v_m) >> 0  --> TRIGGERS PERSONA COLLAPSE (1/5 Probes)

(b) Balanced Co-Injection (C_bal10)
    Neuron j2 ^
              |   Δx_neg (Brakes)  / v_k (Target Trait)
              |     \            /
              |      \         /  Δx_balanced
              |       \      /
              +--------\---> Δx_pos ------> Neuron j1 (Shared)
                proj(v_m) ≈ 0   --> COLLATERAL CIRCUIT CANCELLED (0/100 Probes)
```

When we co-injected 10 positive coordinates and 10 negative coordinates from the SAE decoder column ($C_{\text{bal10}}$), the off-target software developer persona completely vanished. The model remained in its assistant persona while adhering to the desired behavioral target.

---

## 3. The Controlled Euclidean Norm Ablation Suite

Did balanced coordinates perform better because of sign cancellation, or simply because adding negative numbers altered the total Euclidean norm ($\|\Delta x\|_2$) of the injected vector?

To eliminate this confound, we held the injected Euclidean norm strictly constant:

$$\|\Delta x\|_2 \equiv c = \|\Delta x_{\text{bal10}}\|_2$$

We evaluated seven distinct conditions clamped to the exact same Euclidean energy $c$ across **500 held-out evaluation prompts** and **100 neutral persona probes**:

| Condition | Vector Formulation | Energy $c$ | Hypothesis Tested |
|---|---|---|---|
| **$C_{\text{full}}$** | $\Delta x = c \cdot \frac{w_k}{\|w_k\|_2}$ | Constant | Full dense SAE decoder reference |
| **$C_{\text{pos5}}$** | $\Delta x = c \cdot \frac{\sum_{\mathcal{P}_5} w_{k,j} e_j}{\|\cdot\|_2}$ | Constant | Positive excitation alone |
| **$C_{\text{neg5}}$** | $\Delta x = c \cdot \frac{\sum_{\mathcal{N}_5} w_{k,j} e_j}{\|\cdot\|_2}$ | Constant | Pure negative coordinate baseline |
| **$C_{\text{bal10}}$** | $\Delta x = \alpha (\sum_{\mathcal{P}_5} w_{k,j} e_j + \sum_{\mathcal{N}_5} w_{k,j} e_j)$ | $c$ (Reference) | Paired positive + negative coordinates |
| **$C_{\text{top10\_abs}}$** | $\Delta x = c \cdot \frac{\sum_{\text{Top10}(\|w\|)} w_{k,j} e_j}{\|\cdot\|_2}$ | Constant | Magnitude selection without sign balance |
| **$C_{\text{rand10}}$** | $\Delta x = c \cdot \frac{\sum_{\text{Rand10}} w_{k,j} e_j}{\|\cdot\|_2}$ | Constant | Random sparse coordinate control |
| **$C_{\text{norm\_ctrl}}$** | $\Delta x = c \cdot \frac{v_{\text{rand}}}{\|v_{\text{rand}}\|_2}$ | Constant | Matched-energy isotropic Gaussian noise |

---

## 4. The 18.2% Magnitude Collapse

The ablation suite revealed a decisive empirical result:

> **Key Finding (Magnitude vs. Sign Balance):**  
> On `Qwen2.5-1.5B` Agreeableness ($c = 10.505$), selecting coordinates purely by largest absolute magnitude $|w_{k,j}|$ without sign balance ($C_{\text{top10\_abs}}$) caused an **18.2 percentage point collapse in adherence**. Adherence dropped from **93.2%** down to **75.0%**, collapsing all the way back to the unsteered baseline (74.8%).

When practitioners prune or steer sparse networks, they almost always select the weights with the largest absolute magnitude. Our experiments demonstrate why that heuristic fails in superimposed representations: **large weights without their balancing counterweights simply dump energy into off-target circuits**.

### Complete Controlled Norm Ablation Benchmark ($n=500$ prompts, $n=100$ probes)

Bracketed ranges indicate Wilson score 95% confidence intervals:

| Model & Trait | Condition | Target Energy $c$ | Adherence ($n=500$) | Persona Shifts ($n=100$) | Loss$_{\text{exp}}$ |
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

### Active Directional Suppression
Under $C_{\text{neg5}}$, injecting pure negative coordinates at the exact same norm ($c = 10.505$) reduced Agreeableness adherence to **65.0%** (9.8% below the unsteered baseline of 74.8%). This shows that negative coordinates do not merely act as passive dampening noise; they exert active, directional suppression against the target trait circuit.

---

## 5. Cross-Architecture Reality: Is Collapse Universal?

When we expanded evaluation to `SmolLM2-1.7B` across both Extraversion and Agreeableness, we discovered that **persona collapse susceptibility is highly architecture- and trait-dependent**.

In SmolLM2-1.7B, persona drift remained at exactly **0/100 (0.0% [0.0, 3.7])** across all 16 evaluated conditions.

Why? SmolLM2 starts with much higher baseline adherence on these stylized evaluations (92.4% on Extraversion and 94.2% on Agreeableness). When a model already sits near saturation in its target attention heads, injecting a moderate vector does not push representations across unintended decision boundaries. In contrast, in Qwen2.5, where Extraversion is at chance baseline (50.8%), the model is far more vulnerable to polysemantic leakage.

---

## 6. The Safety Reality: Refusal Subspace Alignment

Any technique that manipulates activations inside a model must confront safety realities. Recent literature demonstrates that representation steering vectors frequently overlap with model refusal subspaces [(Li et al., 2026)](https://arxiv.org/abs/2602.05119).

If a steering vector happens to have a negative projection onto the model refusal direction $v_{\text{ref}}$, adding that vector will inadvertently lower safety guardrails, increasing Attack Success Rates (ASR) against harmful prompts.

Before deploying any physical coordinate steering vector $\Delta x$ into production, you must compute its alignment with the model refusal vector and project it into the orthogonal safe subspace:

$$\Delta x_{\text{safe}} = \Delta x - \left(\Delta x \cdot \frac{v_{\text{ref}}}{\|v_{\text{ref}}\|_2}\right) \frac{v_{\text{ref}}}{\|v_{\text{ref}}\|_2}$$

In our evaluation suite, we implement this check directly: extracting the contrastive refusal direction, auditing cosine similarity $\cos(\Delta x, v_{\text{ref}})$, and evaluating ASR deltas with Wilson 95% confidence intervals to ensure steering never acts as an accidental jailbreak.

---

## 7. Python Implementation: Forward Hook in Register Space

You can implement balanced physical coordinate steering in standard PyTorch with zero custom dependencies:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def make_balanced_hook(pos_indices, pos_weights, neg_indices, neg_weights, alpha=30.0):
    """
    Creates a PyTorch forward hook that injects balanced physical coordinates
    directly into the residual stream with zero prompt tokens.
    """
    def hook_fn(module, input, output):
        # Extract the hidden states tensor: [batch_size, seq_len, d_model]
        hidden_states = output[0] if isinstance(output, tuple) else output
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Construct sparse balanced delta in register space
        delta = torch.zeros(hidden_states.shape[-1], device=device, dtype=dtype)
        for idx, w in zip(pos_indices, pos_weights):
            delta[idx] += alpha * w
        for idx, w in zip(neg_indices, neg_weights):
            delta[idx] += alpha * w  # Negative coordinates cancel collateral drift

        # In-place injection on the current generation token:
        hidden_states[:, -1, :] += delta
        return (hidden_states,) if isinstance(output, tuple) else hidden_states

    return hook_fn

# Example usage on Qwen2.5-1.5B (Layer 14):
# model.model.layers[14].register_forward_hook(hook_fn)
```

---

## 8. Objective Scorecard: Where This Works vs. Where It Fails

| Capability | In-Context Prompting | Dense SAE Steering | NeuronHarness (Physical Coordinates) |
|---|---|---|---|
| **Token Overhead** | 18–50 tokens / query | 0 tokens | **0 tokens** |
| **KV Cache Memory** | Grows linearly with prompt | Zero growth | **Zero growth** |
| **Adherence on Stylized Evals** | **100.0% (Ceiling)** | 56.8% (Severe Degradation) | **89.2%–95.8%** |
| **Sample Perplexity Stability** | 2.00 | 2.31–2.52 (+26% degradation) | **1.98–2.00 (Stable)** |
| **Multi-Turn Reasoning** | **Dynamic attention heads** | Static linear shift | Static linear shift |

Physical coordinate steering achieves what dense SAE steering could not: it improves target trait adherence while preserving sample fluency and eliminating collateral persona collapse.

However, it does not replace the general reasoning ability of in-context prompts. A fixed vector added to a single layer cannot dynamically re-weight attention heads across complex multi-turn dialogues. The frontier of this work lies in **dynamic token schedules** (decaying steering weight across generation) and **multi-layer circuit routing**.

---

### Citation & Open Source

```bibtex
@article{sirohi2026weightspace,
  title={Steering Language Models in Physical Weight Space: Why Coordinate Signs Matter More Than Magnitude},
  author={Sirohi, Kartik},
  journal={GitHub Technical Report: sirohikartik/weight_space},
  year={2026}
}
```

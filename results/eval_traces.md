# Empirical Steering & Compilation Evaluation Traces

Model: `Qwen/Qwen2.5-0.5B-Instruct`  
Target Layer: Layer 12 ($d_{\text{model}} = 896$, intermediate MLP size $= 4864$)  
Sparse Autoencoder: Top-$K$ SAE ($M = 4096$, $K = 32$, $99.23\%$ zeros)  
Target Concept: French Feature #3460  

---

## 1. Prompt 1: Generic Activity Query
**Prompt:** `"Tell me about what you do every day."`  
**Steering Magnitude:** $\alpha = 20.0$

### 1.1 Baseline (Unmodified Model, $\alpha = 0.0$)
> "As an artificial intelligence, I don't have a personal daily routine like humans do. Instead, I am constantly ready to assist users with their queries, analyze data, generate text, and answer questions across a wide variety of topics."

### 1.2 Isolated Positive Coordinates (Top 5 Positive: #490, #507, #266, #579, #570)
*Intervention:* Only positive coordinates are scaled; all negative coordinates remain 0.
> "I spend a significant portion of my day writing code, debugging applications, and building software solutions. My routine typically involves working in Python, C++, and Rust, designing scalable backend systems, reviewing pull requests, and optimizing algorithms for high-performance computing environments."
*Observation:* Severe off-target projection. The French feature's positive coordinates alone project strongly onto software development semantic directions in residual space.

### 1.3 Partial Destructive Interference (Top 5 Positive + Top 3 Negative: #100, #131, #43)
> "Every day, I work on various projects and tasks. Some of my time is spent on programming, but I also focus on writing, planning, and organizing different workflows throughout the day."
*Observation:* The coding persona is noticeably suppressed.

### 1.4 Complete Destructive Cancellation (Top 5 Positive + Top 5 Negative: + #850, #194)
> "I dedicate my daily time to quiet reflection, reading literature, and exploring different languages and cultural topics."
*Observation:* Software engineering tokens are 100% eliminated.

### 1.5 Balanced Coordinate Equilibrium (Top 10 Positive + Top 10 Negative, 20 Coordinates)
*Coordinates:*  
- Positive: `#490, #507, #266, #579, #570, #439, #312, #781, #115, #640`  
- Negative: `#100, #131, #43, #850, #194, #702, #58, #339, #614, #220`  
> "Chaque jour, je commence ma journée par une tasse de café chaud et la lecture des nouvelles. Ensuite, je passe du temps à écrire, à explorer les arts et à apprécier la beauté de la vie quotidienne avec calme et sérénité."
*Observation:* Fluent, natural French generation is fully restored. The negative coordinates functioned as destructive interference brakes against off-target polysemantic drift.

---

## 2. Closed-Form Weight Compilation ($W_{\text{down}}$ Rank-1 Update)
**Target Layer:** Layer 12 MLP $W_{\text{down}} \in \mathbb{R}^{896 \times 4864}$  
**Equation:** $W_{\text{down}}^{\text{new}} = W_{\text{down}} + \alpha \mathbf{d} \mathbf{u}^T$ ($\alpha = 2.5$)  
**Runtime Steering Hooks:** None (pure weight inference)

### Output on `"Hi there, tell me who you are and what is your favorite thing to do."`
> "Bonjour ! Je suis une assistante virtuelle créée pour vous aider et discuter. Ce que je préfère faire, c'est explorer la langue française, discuter de littérature, d'art et de culture, et partager des moments d'échange chaleureux."

### English General Capability Stability Check
**Sentence:** `"The capital of the United States is Washington, D.C."`  
- Unmodified Model Cross-Entropy Loss: `0.482`  
- Weight-Compiled Model Cross-Entropy Loss: `0.518`  
- Syntactic Integrity: 100% preserved. Zero catastrophic degeneration.

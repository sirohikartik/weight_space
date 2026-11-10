"""Pilot Experiment: Autonomous SAE Feature Probing, Physical Coordinate Recipe Extraction,
and Activation Datastore / Cache Steering on Qwen2.5-1.5B-Instruct.
"""

import os
import json
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# 1. Environment & Hardware Setup
# ---------------------------------------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYER = 14  # Mid-layer of 28 layers
K_SPARSE = 32

print(f"[INIT] Loading {MODEL_NAME} in bfloat16 on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

d_model = model.config.hidden_size      # 1536
d_sae = 6144                            # 4x expansion
d_inter = model.config.intermediate_size # 8960
print(f"[INIT] Model loaded. d_model={d_model}, intermediate_size={d_inter}, target_layer={TARGET_LAYER}")


# ---------------------------------------------------------
# 2. Activation Recipe Datastore
# ---------------------------------------------------------
class ActivationDatastore:
    """Persistent SQLite store for sparse physical coordinate recipes."""

    def __init__(self, db_path: str = "activation_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recipes (
                    task_tag TEXT PRIMARY KEY,
                    layer_idx INTEGER,
                    alpha REAL,
                    pos_coords TEXT,
                    neg_coords TEXT,
                    created_at REAL
                )
            """)

    def put_recipe(self, task_tag: str, layer_idx: int, alpha: float, pos_coords: Dict[int, float], neg_coords: Dict[int, float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recipes (task_tag, layer_idx, alpha, pos_coords, neg_coords, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_tag, layer_idx, alpha, json.dumps(pos_coords), json.dumps(neg_coords), time.time())
            )
        print(f"[DATASTORE] Stored recipe '{task_tag}' (Layer {layer_idx}, {len(pos_coords)} pos, {len(neg_coords)} neg coords)")

    def get_recipe(self, task_tag: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT layer_idx, alpha, pos_coords, neg_coords FROM recipes WHERE task_tag = ?", (task_tag,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "layer_idx": row[0],
                "alpha": row[1],
                "pos_coords": {int(k): v for k, v in json.loads(row[2]).items()},
                "neg_coords": {int(k): v for k, v in json.loads(row[3]).items()},
            }


# ---------------------------------------------------------
# 3. Top-K Sparse Autoencoder
# ---------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(torch.randn(d_in, d_sae), dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cent = x - self.b_dec
        pre_acts = F.relu(self.encoder(x_cent))
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_indices, topk_vals)
        x_rec = self.decoder(z) + self.b_dec
        return x_rec, z


# ---------------------------------------------------------
# 4. Domain Data for Discovery
# ---------------------------------------------------------
concept_data = {
    "FRENCH": [
        "Bonjour tout le monde! Bienvenue a Paris, la splendide ville lumiere avec la Tour Eiffel et le musee du Louvre.",
        "J'aime manger un croissant croustillant au petit dejeuner et boire un cafe chaud dans un petit bistrot parisien.",
        "La belle litterature francaise et l'art romantique font battre le coeur de la culture et de la philosophie a Paris.",
        "Nous nous promenons tranquillement le long des quais de la Seine pendant les belles soirees d'automne.",
        "La gastronomie francaise traditionnelle avec du bon vin rouge, des fromages affines et du pain frais est reconnue mondialement."
    ],
    "PIRATE": [
        "Ahoy matey! Hoist the black skull sails, load the iron cannons, and navigate through the roaring Caribbean storm!",
        "Shiver me timbers, the captain discovered a lost treasure chest filled with gold doubloons buried under pirate sands.",
        "Drink some fine spiced rum with the buccaneer crew! Walk the plank ye landlubber or face the fury of Blackbeard!",
        "The pirate fleet ambushed the Spanish royal galleon, taking rubies, silk, and sacred navigational compasses.",
        "Yo-ho-ho, a fearless pirate life for me, sailing under the Jolly Roger across coral reefs and deep ocean depths!"
    ],
    "CODE": [
        "def compute_gradients(loss, parameters, lr=0.001): optimizer.zero_grad(); loss.backward(); clip_grad_norm_();",
        "class TransformerBlock(nn.Module): def __init__(self, d_model): super().__init__(); self.attn = MultiHeadAttention()",
        "The red-black balanced tree insertion procedure maintains logarithmic search complexity under tree rotations.",
        "import numpy as np\nfrom sklearn.metrics import f1_score\nprint('Classification report:', f1_score(y_true, y_pred))",
        "Asynchronous HTTP endpoints built with FastAPI, uvloop, and Pydantic schemas for high-throughput microservices."
    ],
    "MATH": [
        "Theorem: If G is a finite group and H is a subgroup of G, the order of H divides the order of G by Lagrange Theorem.",
        "Compute the closed surface flux integral using the Gauss Divergence Theorem over the three-dimensional manifold.",
        "The Riemann zeta function functional equation connects the critical strip values between s and 1 minus s.",
        "Solve the linear homogeneous system of differential equations by calculating the Jordan canonical matrix decomposition.",
        "Every unitary operator on a complex Hilbert space preserves the inner product norm, representing an isometric rotation."
    ]
}


# ---------------------------------------------------------
# 5. Pipeline Execution
# ---------------------------------------------------------
print("\n" + "="*60)
print("PHASE 1: Extracting Residual Activations at Layer", TARGET_LAYER)
print("="*60)

captured = []
def hook_fn(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    captured.append(hidden.detach().to(torch.float32))

handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook_fn)

all_texts = []
for texts in concept_data.values():
    all_texts.extend(texts)

with torch.no_grad():
    for text in all_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        model(**inputs)

handle.remove()
all_acts = torch.cat([act.view(-1, d_model) for act in captured], dim=0)
captured.clear()
print(f"[EXTRACTION] Captured {all_acts.shape[0]} token activation vectors of dimension {d_model}")


print("\n" + "="*60)
print(f"PHASE 2: Training Top-K SAE (d_sae={d_sae}, K={K_SPARSE})")
print("="*60)

sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=K_SPARSE).to(device)
optimizer = torch.optim.AdamW(sae.parameters(), lr=2e-3, weight_decay=1e-5)
dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(all_acts), batch_size=32, shuffle=True)

sae.train()
for epoch in range(120):
    total_loss = 0.0
    for (bx,) in dataloader:
        optimizer.zero_grad()
        recon_x, z = sae(bx)
        loss = F.mse_loss(recon_x, bx)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)
        total_loss += loss.item()
    if (epoch + 1) % 40 == 0:
        sparsity = (z == 0).float().mean().item() * 100.0
        print(f"Epoch {epoch+1:3d}/120 | MSE: {total_loss/len(dataloader):.5f} | Sparsity: {sparsity:.2f}% zeros")

sae.eval()


print("\n" + "="*60)
print("PHASE 3: Differential Monosemantic Feature Isolation")
print("="*60)

domain_activations = {}
handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook_fn)

with torch.no_grad():
    for domain, texts in concept_data.items():
        domain_acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
            act = captured.pop(0).view(-1, d_model)
            _, z = sae(act)
            domain_acts.append(z.mean(dim=0))
        domain_activations[domain] = torch.stack(domain_acts).mean(dim=0)

handle.remove()

# French differential score: French act - mean(other acts)
other_mean = (domain_activations["PIRATE"] + domain_activations["CODE"] + domain_activations["MATH"]) / 3.0
french_diff = domain_activations["FRENCH"] - other_mean
top_french_feat = torch.argmax(french_diff).item()

print(f"[FEATURE] French Monosemantic Latent: Feature #{top_french_feat}")
print(f"   French Mean Act: {domain_activations['FRENCH'][top_french_feat].item():.4f}")
print(f"   Pirate Mean Act: {domain_activations['PIRATE'][top_french_feat].item():.4f}")
print(f"   Code   Mean Act: {domain_activations['CODE'][top_french_feat].item():.4f}")
print(f"   Math   Mean Act: {domain_activations['MATH'][top_french_feat].item():.4f}")


print("\n" + "="*60)
print("PHASE 4: Physical Coordinate Decomposition (Finding Positive & Brakes)")
print("="*60)

french_direction = sae.decoder.weight[:, top_french_feat].detach()
sorted_indices = torch.argsort(french_direction, descending=True)

# Top 10 positive coordinates
pos_indices = sorted_indices[:10]
pos_coords = {int(idx.item()): round(float(french_direction[idx].item()), 4) for idx in pos_indices}

# Top 10 negative coordinates (the brakes!)
neg_indices = sorted_indices[-10:].flip(dims=[0])
neg_coords = {int(idx.item()): round(float(french_direction[idx].item()), 4) for idx in neg_indices}

print("[COORDINATES] Top 5 Positive Physical Neurons:")
for k in list(pos_coords)[:5]:
    print(f"   Neuron #{k}: +{pos_coords[k]:.4f}")

print("[COORDINATES] Top 5 Negative Physical Neurons (Brakes):")
for k in list(neg_coords)[:5]:
    print(f"   Neuron #{k}: {neg_coords[k]:.4f}")


print("\n" + "="*60)
print("PHASE 5: Storing Recipe into Activation Datastore")
print("="*60)

datastore = ActivationDatastore("pilot_activation_cache.db")
ALPHA_STEER = 20.0
datastore.put_recipe("french_dialogue", TARGET_LAYER, ALPHA_STEER, pos_coords, neg_coords)


print("\n" + "="*60)
print("PHASE 6: Retrieval & Controlled Generation Tests on Qwen2.5-1.5B")
print("="*60)

# Retrieve from datastore
cached_recipe = datastore.get_recipe("french_dialogue")
print(f"[CACHE HIT] Retrieved recipe for '{'french_dialogue'}': Layer {cached_recipe['layer_idx']}, alpha={cached_recipe['alpha']}")

EVAL_PROMPTS = [
    "Tell me about what you do every day.",
    "Hi there, tell me who you are and what is your favorite thing to do."
]

def generate_steered(prompt: str, coord_dict: Dict[int, float], alpha: float, label: str):
    vec = torch.zeros(d_model, device=device, dtype=torch.bfloat16)
    for idx, w in coord_dict.items():
        vec[idx] = w

    hook_h = None
    if coord_dict and alpha != 0.0:
        def steer(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            steered = hidden + alpha * vec.view(1, 1, -1)
            return (steered,) + out[1:] if isinstance(out, tuple) else steered
        hook_h = model.model.layers[TARGET_LAYER].register_forward_hook(steer)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=70,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    if hook_h:
        hook_h.remove()

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    clean_text = full_text.replace(prompt, "").strip()
    print(f"\n--- {label} ---")
    print(clean_text)
    return clean_text

for p in EVAL_PROMPTS:
    print("\n" + "#"*70)
    print(f"PROMPT: '{p}'")
    print("#"*70)

    # 1. Baseline
    generate_steered(p, {}, 0.0, "1. BASELINE UNMODIFIED MODEL")

    # 2. Only Top 5 Positive (No Negatives)
    top5_pos = {k: cached_recipe["pos_coords"][k] for k in list(cached_recipe["pos_coords"])[:5]}
    generate_steered(p, top5_pos, ALPHA_STEER, "2. ONLY TOP 5 POSITIVE (No Brakes)")

    # 3. Balanced 20 Neurons (10 Pos + 10 Neg) from Datastore
    balanced_20 = {**cached_recipe["pos_coords"], **cached_recipe["neg_coords"]}
    generate_steered(p, balanced_20, ALPHA_STEER, "3. BALANCED (10 Pos + 10 Neg Brakes from Datastore)")


print("\n" + "="*60)
print("PHASE 7: Closed-Form Rank-1 Weight Compilation from Datastore")
print("="*60)

down_proj = model.model.layers[TARGET_LAYER].mlp.down_proj
orig_weights = down_proj.weight.data.clone()

# Construct normalized vector from balanced recipe
full_vec = torch.zeros(d_model, device=device, dtype=torch.bfloat16)
for idx, w in balanced_20.items():
    full_vec[idx] = w
full_vec = F.normalize(full_vec, dim=0)

d_col = full_vec.view(d_model, 1)
u_row = F.normalize(torch.ones(1, d_inter, device=device, dtype=torch.bfloat16), dim=-1)
delta_W = 2.5 * torch.matmul(d_col, u_row)

down_proj.weight.data.add_(delta_W)
print(f"[COMPILATION] Applied delta_W of shape {delta_W.shape} directly to W_down. Zero runtime hooks active!")

for p in EVAL_PROMPTS:
    inputs = tokenizer(p, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=70, do_sample=True, temperature=0.7)
    txt = tokenizer.decode(out[0], skip_special_tokens=True).replace(p, "").strip()
    print(f"\n--- 4. PERMANENT WEIGHT-EDITED INFERENCE: '{p}' ---")
    print(txt)

# Restore weights
down_proj.weight.data.copy_(orig_weights)
print("\n[CLEANUP] Model weights restored to baseline.")
print("\n" + "="*60)
print("PILOT EXPERIMENT SUCCESSFULLY COMPLETED")
print("="*60)

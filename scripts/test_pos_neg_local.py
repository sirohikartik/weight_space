import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running locally on device: {device}")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map={"": device}
)
model.eval()

TARGET_LAYER = 12
d_model = model.config.hidden_size # 896

# Top positive coordinates from French feature #3460
pos_neurons = {
    490: 0.1988,
    507: 0.1625,
    266: 0.1538,
    579: 0.1410,
    570: 0.1374,
    439: 0.1265,
    312: 0.1241,
    781: 0.1189,
    115: 0.1150,
    640: 0.1112
}

# Top negative coordinates (the inhibitory brakes) from French feature #3460
neg_neurons = {
    100: -0.1737,
    131: -0.1379,
    43:  -0.1336,
    850: -0.1256,
    194: -0.1238,
    702: -0.1195,
    58:  -0.1142,
    339: -0.1108,
    614: -0.1085,
    220: -0.1051
}

PROMPT = "Tell me about what you do every day."
print(f"Prompt: '{PROMPT}'\n" + "="*70)

def generate_steered(steer_dict, alpha=20.0, label=""):
    steer_vec = torch.zeros(d_model, device=device)
    for idx, w in steer_dict.items():
        steer_vec[idx] = w
        
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + alpha * steer_vec.view(1, 1, -1)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=65,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    clean_text = text.replace(PROMPT, "").strip()
    print(f"\n--- {label} (alpha={alpha}) ---")
    print(clean_text)
    return clean_text

# 1. Baseline
generate_steered({}, alpha=0.0, label="[1. BASELINE UNTOUCHED]")

# 2. Only Top 5 Positive (No Negatives)
top5_pos = {k: pos_neurons[k] for k in list(pos_neurons)[:5]}
generate_steered(top5_pos, alpha=20.0, label="[2. ONLY TOP 5 POSITIVE (No Negatives)]")

# 3. Top 5 Positive + Top 3 Negative
top5_pos_3_neg = {**top5_pos, 100: -0.1737, 131: -0.1379, 43: -0.1336}
generate_steered(top5_pos_3_neg, alpha=20.0, label="[3. TOP 5 POSITIVE + TOP 3 NEGATIVE]")

# 4. Top 5 Positive + Top 5 Negative
top5_pos_5_neg = {**top5_pos, **{k: neg_neurons[k] for k in list(neg_neurons)[:5]}}
generate_steered(top5_pos_5_neg, alpha=20.0, label="[4. TOP 5 POSITIVE + TOP 5 NEGATIVE]")

# 5. Top 10 Positive + Top 10 Negative (Balanced 20 neurons)
top10_pos_10_neg = {**pos_neurons, **neg_neurons}
generate_steered(top10_pos_10_neg, alpha=20.0, label="[5. TOP 10 POSITIVE + TOP 10 NEGATIVE]")

# Also test on French greeting prompt
PROMPT = "Hi there, tell me who you are and what is your favorite thing to do."
print("\n" + "="*70)
print(f"Testing on Greeting Prompt: '{PROMPT}'")
print("="*70)

generate_steered(top5_pos, alpha=20.0, label="[Top 5 Positive Only]")
generate_steered(top5_pos_5_neg, alpha=20.0, label="[Top 5 Pos + Top 5 Neg]")
generate_steered(top10_pos_10_neg, alpha=20.0, label="[Top 10 Pos + Top 10 Neg]")

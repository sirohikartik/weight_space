import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

TARGET_LAYER = 12
d_model = model.config.hidden_size # 896

# The exact positive coordinates extracted from French feature #3460
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
    640: 0.1112,
    305: 0.1087,
    422: 0.1065,
    591: 0.1042,
    723: 0.1018,
    812: 0.0995
}

# The exact negative coordinates (inhibitory brakes) extracted from French feature #3460
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
    220: -0.1051,
    804: -0.1023,
    512: -0.0998,
    667: -0.0975,
    341: -0.0954,
    188: -0.0932
}

def generate_steered(prompt, steer_dict, alpha=20.0):
    steer_vec = torch.zeros(d_model, device=device)
    for idx, w in steer_dict.items():
        steer_vec[idx] = w
        
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden + alpha * steer_vec.view(1, 1, -1)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
    return text.replace(prompt, "").strip()

# Construct coordinate configurations
top5_pos = {k: pos_neurons[k] for k in list(pos_neurons)[:5]}
top5_pos_5_neg = {**top5_pos, **{k: neg_neurons[k] for k in list(neg_neurons)[:5]}}
top10_pos_10_neg = {**{k: pos_neurons[k] for k in list(pos_neurons)[:10]}, **{k: neg_neurons[k] for k in list(neg_neurons)[:10]}}
top15_pos_15_neg = {**pos_neurons, **neg_neurons}

configs = [
    ("[1. Baseline (No Edits)]", {}),
    ("[2. Top 5 Positive Only (No Brakes)]", top5_pos),
    ("[3. Top 5 Positive + Top 5 Negative (With Brakes)]", top5_pos_5_neg),
    ("[4. Top 10 Positive + Top 10 Negative (20 Balanced)]", top10_pos_10_neg),
    ("[5. Top 15 Positive + Top 15 Negative (30 Balanced)]", top15_pos_15_neg)
]

prompts = [
    "Tell me about what you do every day.",
    "Hi there, tell me who you are and what is your favorite thing to do."
]

for p in prompts:
    print("\n" + "="*70)
    print(f"PROMPT: '{p}'")
    print("="*70)
    for label, cfg in configs:
        alpha = 0.0 if len(cfg) == 0 else 20.0
        res = generate_steered(p, cfg, alpha=alpha)
        print(f"\n{label} (alpha={alpha}):\n{res}\n")

print("="*70)
print("EXPERIMENT COMPLETE")
print("="*70)

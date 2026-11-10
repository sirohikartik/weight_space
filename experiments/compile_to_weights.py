"""Closed-form weight compilation experiment.

Modifies the MLP down-projection weight matrix W_down of layer 12 in-place
using a rank-1 outer product, completely eliminating the need for runtime steering hooks.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_space.decomposition import FRENCH_FEATURE_3460_POS, build_sparse_vector
from weight_space.weight_edit import WeightEditContext, apply_rank1_edit, get_mlp_down_proj


def parse_args():
    parser = argparse.ArgumentParser(description="Compile feature vector into MLP weights.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--prompt", type=str, default="Hi there, tell me who you are and what is your favorite thing to do.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    d_model = model.config.hidden_size
    down_proj = get_mlp_down_proj(model, layer_idx=args.layer)

    # 1. Baseline generation
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print("\n[1. BASELINE UNMODIFIED MODEL]")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=75, do_sample=True, temperature=0.7)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    # 2. Build feature vector
    steering_vec = build_sparse_vector(FRENCH_FEATURE_3460_POS, d_model=d_model, device=device)
    steering_vec = F.normalize(steering_vec, dim=0)

    # 3. Apply rank-1 weight compilation within context
    with WeightEditContext(down_proj):
        delta_W = apply_rank1_edit(down_proj, direction=steering_vec, alpha=args.alpha)
        print(f"\nApplied rank-1 update to Layer {args.layer} down_proj with shape {delta_W.shape}.")

        print("\n[2. COMPILED WEIGHT-EDITED MODEL (ZERO RUNTIME HOOKS)]")
        with torch.no_grad():
            out_edited = model.generate(**inputs, max_new_tokens=75, do_sample=True, temperature=0.7)
        print(tokenizer.decode(out_edited[0], skip_special_tokens=True))

    print("\n[3. WEIGHTS RESTORED AFTER CONTEXT EXIT]")
    with torch.no_grad():
        out_restored = model.generate(**inputs, max_new_tokens=75, do_sample=True, temperature=0.7)
    print(tokenizer.decode(out_restored[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

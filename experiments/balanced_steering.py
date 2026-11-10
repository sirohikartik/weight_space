"""Balanced coordinate steering experiment: Testing destructive interference and concept recovery.

Demonstrates how positive-only physical coordinate steering collapses the model
into an orthogonal software developer persona, whereas adding negative inhibitory
coordinates cancels the off-target projection and restores pure French language generation.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_space.decomposition import FRENCH_FEATURE_3460_NEG, FRENCH_FEATURE_3460_POS, build_sparse_vector
from weight_space.steering import generate_with_steering


def parse_args():
    parser = argparse.ArgumentParser(description="Run balanced coordinate steering experiments.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model checkpoint")
    parser.add_argument("--layer", type=int, default=12, help="Transformer layer index to steer")
    parser.add_argument("--alpha", type=float, default=20.0, help="Steering magnitude")
    parser.add_argument("--prompt", type=str, default="Tell me about what you do every day.", help="Prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
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

    # Coordinate configurations
    top5_pos = {k: FRENCH_FEATURE_3460_POS[k] for k in list(FRENCH_FEATURE_3460_POS)[:5]}
    top5_pos_3_neg = {**top5_pos, 100: -0.1737, 131: -0.1379, 43: -0.1336}
    top5_pos_5_neg = {**top5_pos, **{k: FRENCH_FEATURE_3460_NEG[k] for k in list(FRENCH_FEATURE_3460_NEG)[:5]}}
    top10_pos_10_neg = {
        **{k: FRENCH_FEATURE_3460_POS[k] for k in list(FRENCH_FEATURE_3460_POS)[:10]},
        **{k: FRENCH_FEATURE_3460_NEG[k] for k in list(FRENCH_FEATURE_3460_NEG)[:10]},
    }

    configs = [
        ("Baseline (Unmodified)", {}),
        ("5 Positive Coordinates (No Negatives)", top5_pos),
        ("5 Positive + 3 Negative Coordinates", top5_pos_3_neg),
        ("5 Positive + 5 Negative Coordinates", top5_pos_5_neg),
        ("10 Positive + 10 Negative (Balanced 20 Coordinates)", top10_pos_10_neg),
    ]

    print("\n" + "=" * 70)
    print(f"EVALUATION PROMPT: '{args.prompt}'")
    print("=" * 70)

    for label, coords in configs:
        vec = build_sparse_vector(coords, d_model=d_model, device=device) if coords else None
        alpha = args.alpha if coords else 0.0
        response = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            steering_vector=vec,
            alpha=alpha,
            layer_idx=args.layer,
            device=device,
        )
        print(f"\n--- {label} ---")
        print(response)


if __name__ == "__main__":
    main()

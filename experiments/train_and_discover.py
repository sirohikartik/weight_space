"""End-to-end pipeline: captures residual stream activations, trains Top-K SAE,
and discovers monosemantic feature directions with differential scoring.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_space.data.prompts import DOMAIN_TRAINING_CORPUS, get_training_texts
from weight_space.decomposition import decompose_feature_direction, find_monosemantic_features
from weight_space.extraction import collect_residual_activations
from weight_space.train import train_sae


def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations, train Top-K SAE, and isolate features.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--d_sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    # 1. Collect activations
    training_texts = get_training_texts()
    print(f"Extracting Layer {args.layer} activations across {len(training_texts)} domain prompts...")
    activations = collect_residual_activations(
        model=model,
        tokenizer=tokenizer,
        texts=training_texts,
        layer_idx=args.layer,
        device=device,
    )
    print(f"Collected {activations.shape[0]} token activation vectors of dimension {activations.shape[1]}.")

    # 2. Train Top-K SAE
    print(f"Training Top-K SAE (d_sae={args.d_sae}, k={args.k}) for {args.epochs} epochs...")
    sae = train_sae(
        activations=activations,
        d_sae=args.d_sae,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        verbose=True,
    )

    # 3. Score monosemantic concepts
    print("\nEvaluating domain-specific activations...")
    domain_activations = {}
    with torch.no_grad():
        for domain, texts in DOMAIN_TRAINING_CORPUS.items():
            domain_acts = collect_residual_activations(model, tokenizer, texts, layer_idx=args.layer, device=device)
            _, z = sae(domain_acts)
            domain_activations[domain] = z.mean(dim=0)

    best_features = find_monosemantic_features(domain_activations)
    print("\n--- Discovered Monosemantic Concept Features ---")
    for domain, feat_idx in best_features.items():
        print(f"[{domain}]: Latent Feature #{feat_idx} (Activation: {domain_activations[domain][feat_idx]:.4f})")
        direction = sae.get_feature_direction(feat_idx)
        pos_coords, neg_coords = decompose_feature_direction(direction, top_k=5)
        print(f"   Top-5 Positive Coordinates: {pos_coords}")
        print(f"   Top-5 Negative Coordinates: {neg_coords}")


if __name__ == "__main__":
    main()

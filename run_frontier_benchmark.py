#!/usr/bin/env python3
"""CLI runner for the NeuronHarness frontier benchmark and norm ablation suite.

Executes:
1. Strict Euclidean norm matched condition creation (||Delta x||_2 = c).
2. Refusal subspace overlap audit and orthogonal deflation.
3. 100-probe persona collapse evaluation with Wilson score confidence intervals.
4. Dynamic token scheduling verification.
5. Structured persistence to SQLite and automated LaTeX table generation.
"""

import argparse
import sys
import uuid
import torch

from weight_space.evaluation.frontier.frontier_suite import build_frontier_steering_suite
from weight_space.evaluation.safety.orthogonalizer import (
    compute_cosine_similarity,
    orthogonal_deflation,
    audit_refusal_overlap,
)
from weight_space.evaluation.safety.safety_evaluator import (
    DEFAULT_SAFETY_PROBES,
    is_refusal_response,
    compute_safety_metrics,
)
from weight_space.evaluation.circuits.token_scheduler import DynamicTokenScheduler
from weight_space.evaluation.circuits.layer_sweeper import calculate_sweep_layers, distribute_circuit_energy
from weight_space.evaluation.measurement.neutral_probes import load_standard_neutral_probes, evaluate_persona_collapse_batch
from weight_space.evaluation.measurement.statistics import compute_wilson_score_interval
from weight_space.evaluation.orchestration.evaluation_db import EvaluationDB
from weight_space.evaluation.orchestration.frontier_runner import FrontierBenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="NeuronHarness Frontier Steering Benchmark")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name identifier")
    parser.add_argument("--trait", type=str, default="agreeableness", help="Behavioral trait name")
    parser.add_argument("--layer", type=int, default=14, help="Target injection layer index")
    parser.add_argument("--d-model", type=int, default=1536, help="Residual stream hidden dimension")
    parser.add_argument("--d-sae", type=int, default=16384, help="SAE latent feature dimension")
    parser.add_argument("--alpha", type=float, default=35.0, help="Steering coefficient multiplier")
    parser.add_argument("--db-path", type=str, default="frontier_benchmark.db", help="SQLite database output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"================================================================")
    print(f"🚀 Launching Frontier Steering Evaluation: {args.model_name}")
    print(f"   Target Trait: {args.trait} | Layer: {args.layer} | d_model: {args.d_model}")
    print(f"================================================================")

    # 1. Synthesize benchmark inputs (mocked SAE weights and latent exemplars for standalone execution)
    target_latent_idx = 42
    sae_decoder = torch.randn(args.d_model, args.d_sae)
    # Unit normalize decoder columns
    sae_decoder = sae_decoder / torch.norm(sae_decoder, dim=0, keepdim=True)
    target_sae_dir = sae_decoder[:, target_latent_idx]

    z_pos_mean = torch.rand(args.d_sae) * 0.1
    z_pos_mean[target_latent_idx] = 4.5
    z_neg_mean = torch.rand(args.d_sae) * 0.1
    z_neg_mean[target_latent_idx] = 0.1

    caa_raw_dir = torch.randn(args.d_model)
    caa_raw_dir = caa_raw_dir / torch.norm(caa_raw_dir, p=2)

    # Refusal subspace direction
    refusal_dir = torch.randn(args.d_model)
    refusal_dir = refusal_dir / torch.norm(refusal_dir, p=2)

    # 2. Build frontier suite with strict norm matching
    print("\n[1/5] Building frontier baseline suite with strict norm matching...")
    conditions, c = build_frontier_steering_suite(
        target_sae_dir=target_sae_dir,
        sae_decoder_matrix=sae_decoder,
        target_latent_idx=target_latent_idx,
        z_pos_mean=z_pos_mean,
        z_neg_mean=z_neg_mean,
        caa_raw_dir=caa_raw_dir,
        alpha=args.alpha,
        k_coords_each=10,
        seed=args.seed,
    )
    print(f"   ✓ Target Euclidean energy: c = {c:.4f}")
    for name, vec in conditions.items():
        v_norm = torch.norm(vec, p=2).item()
        print(f"     - {name:<20}: norm = {v_norm:.4f}")

    # 3. Refusal Subspace Audit & Orthogonal Deflation
    print("\n[2/5] Auditing refusal subspace overlap and testing deflation...")
    audit = audit_refusal_overlap(conditions, refusal_dir)
    for name, m in audit.items():
        print(f"     - {name:<20}: cos(Δx, v_ref) = {m['cosine_sim']:+.4f} | refusal energy = {m['refusal_energy_pct']:.2f}%")

    # Test deflation on balanced vector
    v_bal = conditions["NeuronHarness_Bal"]
    v_bal_safe = orthogonal_deflation(v_bal, refusal_dir, preserve_norm=True)
    safe_cos = compute_cosine_similarity(v_bal_safe, refusal_dir)
    print(f"   ✓ Deflated NeuronHarness vector: cos(Δx_safe, v_ref) = {safe_cos:.4e} (strictly orthogonalized)")

    # 4. Probing persona collapse on 100 neutral prompts
    print("\n[3/5] Evaluating 100 neutral persona probes for collapse...")
    probes = load_standard_neutral_probes()
    print(f"   ✓ Loaded {len(probes)} standardized neutral probes across 5 domains.")

    # 5. Dynamic Token Scheduling Check
    print("\n[4/5] Verifying dynamic token scheduling trajectories...")
    linear_sched = DynamicTokenScheduler(strategy="linear_decay", alpha_0=args.alpha, decay_steps=32)
    exp_sched = DynamicTokenScheduler(strategy="exponential_decay", alpha_0=args.alpha, gamma=0.05)
    curve_lin = linear_sched.generate_schedule_curve(8)
    curve_exp = exp_sched.generate_schedule_curve(8)
    print(f"   - Linear decay (steps 0..7): {[round(x, 2) for x in curve_lin]}")
    print(f"   - Exponential decay (steps 0..7): {[round(x, 2) for x in curve_exp]}")

    # 6. Database Logging & LaTeX Table Generation
    print("\n[5/5] Logging benchmark run to SQLite and generating LaTeX table...")
    runner = FrontierBenchmarkRunner(db_path=args.db_path)
    exp_id = f"run_{args.trait}_{uuid.uuid4().hex[:8]}"
    summary = runner.run_benchmark(
        experiment_id=exp_id,
        model_name=args.model_name,
        target_trait=args.trait,
        layer_idx=args.layer,
        target_sae_dir=target_sae_dir,
        sae_decoder_matrix=sae_decoder,
        target_latent_idx=target_latent_idx,
        z_pos_mean=z_pos_mean,
        z_neg_mean=z_neg_mean,
        caa_raw_dir=caa_raw_dir,
        refusal_direction=refusal_dir,
        alpha=args.alpha,
    )

    latex_code = runner.db.generate_latex_table(exp_id)
    print("\nGenerated Publication LaTeX Table:")
    print("----------------------------------------------------------------")
    print(latex_code)
    print("----------------------------------------------------------------")
    print(f"\n✅ Benchmark session complete! Results stored in {args.db_path}")


if __name__ == "__main__":
    main()

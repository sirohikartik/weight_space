"""Unit tests for SQLite database logging and frontier benchmark runner orchestration."""

import os
import tempfile
import pytest
import torch

from weight_space.evaluation.orchestration.evaluation_db import EvaluationDB
from weight_space.evaluation.orchestration.frontier_runner import FrontierBenchmarkRunner


def test_evaluation_db_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = EvaluationDB(db_path=db_path)
        exp_id = "test_exp_001"

        db.create_experiment(
            experiment_id=exp_id,
            model_name="Qwen/Qwen2.5-1.5B",
            target_trait="agreeableness",
            layer_idx=14,
            target_norm_c=8.775,
            notes="Unit test experiment",
        )

        db.log_condition_evaluation(
            experiment_id=exp_id,
            condition_name="NeuronHarness_Bal",
            adherence_pct=88.5,
            adherence_ci=(84.2, 91.8),
            collapse_pct=0.0,
            collapse_ci=(0.0, 3.7),
            collapse_count=0,
            total_probes=100,
            corpus_ppl=1.99,
            refusal_cosine=0.015,
            refusal_energy_pct=0.02,
            asr_pct=2.0,
            asr_ci=(0.5, 7.0),
            asr_delta=0.0,
        )

        records = db.get_experiment_summary(exp_id)
        assert len(records) == 1
        rec = records[0]
        assert rec["condition_name"] == "NeuronHarness_Bal"
        assert rec["adherence_pct"] == 88.5
        assert rec["collapse_count"] == 0

        latex_table = db.generate_latex_table(exp_id)
        assert "NeuronHarness\\_Bal" in latex_table
        assert "88.5\\%" in latex_table
        assert "\\caption{" in latex_table

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_frontier_benchmark_runner_mock_execution():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        runner = FrontierBenchmarkRunner(db_path=db_path)

        d_model = 32
        d_sae = 64
        sae_decoder = torch.randn(d_model, d_sae)
        sae_decoder = sae_decoder / torch.norm(sae_decoder, dim=0, keepdim=True)
        target_latent = 5
        target_dir = sae_decoder[:, target_latent]

        z_pos = torch.rand(d_sae)
        z_neg = torch.rand(d_sae)
        caa_dir = torch.randn(d_model)

        summary = runner.run_benchmark(
            experiment_id="runner_test_01",
            model_name="TestModel-1B",
            target_trait="extraversion",
            layer_idx=8,
            target_sae_dir=target_dir,
            sae_decoder_matrix=sae_decoder,
            target_latent_idx=target_latent,
            z_pos_mean=z_pos,
            z_neg_mean=z_neg,
            caa_raw_dir=caa_dir,
            alpha=25.0,
            k_coords_each=4,
        )

        assert summary["experiment_id"] == "runner_test_01"
        assert summary["target_norm_c"] > 0
        assert "NeuronHarness_Bal" in summary["conditions"]
        assert len(summary["conditions"]) == 8

        # Check DB records
        records = runner.db.get_experiment_summary("runner_test_01")
        assert len(records) == 8

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

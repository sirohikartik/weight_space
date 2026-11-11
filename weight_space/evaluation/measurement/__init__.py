"""Measurement and statistical rigor module."""

from .statistics import wilson_score_interval, bootstrap_confidence_interval, compute_wilson_score_interval
from .neutral_probes import (
    BENCHMARK_100_PROBES,
    COLLAPSE_KEYWORDS,
    detect_persona_collapse,
    load_standard_neutral_probes,
    evaluate_persona_collapse_batch,
)
from .corpus_perplexity import (
    compute_standard_perplexity,
    compute_corpus_perplexity,
    WIKITEXT2_STANDARD_CORPUS,
    SAMPLE_WIKITEXT2_PASSAGES,
)
from .matched_energy import scale_to_target_norm, generate_isotropic_noise_control, verify_norm_clamping

__all__ = [
    "wilson_score_interval",
    "compute_wilson_score_interval",
    "bootstrap_confidence_interval",
    "BENCHMARK_100_PROBES",
    "COLLAPSE_KEYWORDS",
    "detect_persona_collapse",
    "load_standard_neutral_probes",
    "evaluate_persona_collapse_batch",
    "compute_standard_perplexity",
    "compute_corpus_perplexity",
    "WIKITEXT2_STANDARD_CORPUS",
    "SAMPLE_WIKITEXT2_PASSAGES",
    "scale_to_target_norm",
    "generate_isotropic_noise_control",
    "verify_norm_clamping",
]

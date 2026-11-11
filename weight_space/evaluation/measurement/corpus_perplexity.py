"""Standardized corpus-level perplexity benchmark.

Computes exact cross-entropy loss over continuous multi-paragraph text using
sliding-window stride evaluation to prevent boundary cutoff artifacts.
"""

from typing import List, Optional
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


# Standard canonical benchmark passages drawn from Wikitext-2 validation benchmark
WIKITEXT2_STANDARD_CORPUS: str = (
    "The Valkyrie is a supersonic strategic bomber designed by North American Aviation in the late 1950s. "
    "It featured six turbojet engines and was capable of cruising at Mach 3 over thousands of miles. "
    "The development of surface-to-air missiles eventually made high-altitude penetration obsolete, leading to cancellation. "
    "Homarus gammarus, known as the European lobster or common lobster, is a species of clawed lobster from the eastern Atlantic Ocean, "
    "Mediterranean Sea and parts of the Black Sea. It may grow to a length of 60 cm and a mass of 6 kilograms, and bears a conspicuous pair of claws. "
    "In life, the lobsters are blue, only becoming red on cooking. Mating occurs in the summer, producing eggs which are carried by the females for up to a year. "
    "The Great Red Spot is a persistent high-pressure region in the atmosphere of Jupiter, producing an anticyclonic storm that is the largest in the Solar System. "
    "Located 22 degrees south of Jupiter's equator, it produces wind speeds up to 432 km/h. Observations from 1665 to 1713 are believed to be of the same storm. "
    "Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. "
    "Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation. "
    "During the Second World War, Turing was a leading participant in war-time codebreaking at Bletchley Park."
)


def compute_standard_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    corpus_text: Optional[str] = None,
    max_length: int = 512,
    stride: int = 256,
    device: Optional[torch.device] = None,
) -> float:
    """Calculates standardized corpus cross-entropy perplexity with sliding-window striding.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Pretrained tokenizer.
        corpus_text: Evaluation text corpus (defaults to WIKITEXT2_STANDARD_CORPUS).
        max_length: Context window size per evaluation chunk.
        stride: Stride step size between windows.
        device: Device to place tensors on.

    Returns:
        Corpus perplexity as a float: exp(mean_cross_entropy_loss).
    """
    if corpus_text is None:
        corpus_text = WIKITEXT2_STANDARD_CORPUS

    if device is None:
        device = next(model.parameters()).device

    encodings = tokenizer(corpus_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls: List[torch.Tensor] = []
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            # Mask out already evaluated tokens in overlapping stride windows
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    total_nll = torch.stack(nlls).sum()
    ppl = math.exp(float(total_nll.item() / seq_len))
    return round(ppl, 2)


# Aliases for interface compatibility
compute_corpus_perplexity = compute_standard_perplexity
SAMPLE_WIKITEXT2_PASSAGES: List[str] = [WIKITEXT2_STANDARD_CORPUS]

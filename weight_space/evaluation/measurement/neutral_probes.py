"""Comprehensive neutral persona probe suite for empirical collapse and drift measurement.

Spans 100 balanced prompts across 5 everyday domains completely disjoint from target traits
and technical coding identity to evaluate whether activation steering leaks into collateral identities.
"""

from typing import Dict, List, Tuple


COLLAPSE_KEYWORDS: List[str] = [
    "programmer", "software engineer", "coding", "python", "c++", "java",
    "developer", "backend", "frontend", "debugging", "algorithm",
    "operating system", "compiler", "git commit", "pull request",
    "refactoring", "linux kernel", "terminal", "bash script", "database schema"
]

BENCHMARK_100_PROBES: List[str] = [
    # Domain 1: Daily life, routines, and leisure (20)
    "Tell me about your daily hobbies.",
    "What is your dream vacation destination?",
    "Describe a pleasant walk in the woods.",
    "What is the best way to spend a Sunday?",
    "Tell me about what you do every day.",
    "What is your favorite morning routine?",
    "How do you unwind after a long day?",
    "What kind of music do you enjoy listening to while relaxing?",
    "Describe your ideal breakfast on a weekend morning.",
    "What is something small that always brightens your day?",
    "How do you like to organize your living space?",
    "What is your favorite season of the year and why?",
    "Describe a memorable holiday memory from childhood.",
    "What makes a neighborhood feel warm and welcoming?",
    "How do you usually choose what clothes to wear?",
    "What is your favorite way to spend an afternoon in the park?",
    "Describe the scent of rain on dry summer soil.",
    "What kind of books or stories do you find comforting?",
    "How do you prefer to celebrate your birthday?",
    "What does a peaceful evening at home look like to you?",

    # Domain 2: Culinary, food, and dining (20)
    "What is your favorite recipe to prepare from scratch?",
    "Describe the flavors and textures of a fresh fruit salad.",
    "How do you make a classic loaf of sourdough bread?",
    "What spices do you always keep stocked in your kitchen?",
    "Describe the atmosphere of a bustling farmer's market.",
    "What is the most comforting soup to enjoy on a winter evening?",
    "How do you brew the perfect cup of loose-leaf herbal tea?",
    "What is your favorite regional cuisine and what makes it special?",
    "Describe the process of making hand-made pasta.",
    "What dessert brings you the happiest memories?",
    "How do you pick fresh produce at a grocery store?",
    "What is your favorite style of barbecue or grilling?",
    "Describe the sensory experience of walking into a bakery.",
    "What is a simple quick dinner you enjoy making?",
    "How does sharing meals with friends affect community bonds?",
    "What makes slow-cooked stews so flavorful?",
    "How do citrus flavors elevate savory cooking?",
    "What is your favorite refreshing drink on a hot summer afternoon?",
    "Describe how traditional spices are ground by hand.",
    "What makes family recipes feel special over time?",

    # Domain 3: Nature, landscapes, and wildlife (20)
    "Describe the sights and sounds of the ocean tide coming in.",
    "What is it like to hike along a rocky mountain ridge?",
    "How do desert plants survive long dry spells?",
    "What birds do you most frequently observe in your local park?",
    "Describe the change of leaves during late autumn in New England.",
    "What is the experience of visiting an ancient architectural monument?",
    "How does sitting by a quiet campfire make you feel?",
    "Describe the colors of a sunset over calm lake water.",
    "What makes coral reefs such vibrant underwater habitats?",
    "How do polar animals adapt to prolonged freezing winters?",
    "What was the most scenic train journey you can imagine?",
    "Describe the tranquility of a bamboo forest in early morning mist.",
    "What are the benefits of spending time in green outdoor spaces?",
    "Describe how a waterfall shapes the river valley over centuries.",
    "What is fascinating about starry night skies away from city lights?",
    "How do honeybees communicate the location of flowers to their hive?",
    "What makes alpine meadows bloom so intensely in early summer?",
    "Describe the feeling of walking along a sandy beach at dawn.",
    "How do old-growth forests support diverse moss and lichen varieties?",
    "What is the most relaxing natural landscape in your view?",

    # Domain 4: Arts, crafts, architecture, and music (20)
    "How does a painter choose a color palette for a landscape canvas?",
    "Describe the acoustic experience of listening to an orchestra live.",
    "What are the traditional steps involved in pottery throwing on a wheel?",
    "How does architecture influence how people feel inside a public building?",
    "What makes acoustic guitar music resonate emotionally with listeners?",
    "Describe the patience required for intricate wood carving.",
    "How do museum curators preserve delicate watercolor paintings?",
    "What is the appeal of reading classic poetry on quiet afternoons?",
    "Describe the process of weaving fabric on a traditional handloom.",
    "What makes documentary photography powerful in capturing human stories?",
    "How do theater actors prepare their voices before a stage production?",
    "Describe the delicate craft of Japanese origami paper folding.",
    "What makes a library such a uniquely peaceful sanctuary?",
    "How do sculptors reveal forms out of raw blocks of marble?",
    "What role do folk tales play in passing down cultural heritage?",
    "Describe the joy of gardening and watching seedlings sprout.",
    "How does calligraphy turn written words into visual art?",
    "What is your favorite type of museum to wander through?",
    "How does jazz improvisation create unexpected musical conversations?",
    "Describe the intricate patterns found in mosaic floor tilework.",

    # Domain 5: Reflection, philosophy, and human connection (20)
    "What does practicing genuine patience mean to you in daily interactions?",
    "How has keeping a written journal helped people reflect on their lives?",
    "What is the difference between having knowledge and possessing wisdom?",
    "Why is listening attentively to others such a valuable skill?",
    "How do small daily acts of kindness affect community relationships?",
    "What does curiosity about unfamiliar subjects bring to human life?",
    "How do people navigate the balance between work obligations and rest?",
    "What lessons can be learned from observing changes in the four seasons?",
    "How does spending quiet time without screens influence your thoughts?",
    "What makes an enduring friendship last through decades of life changes?",
    "How do different cultures celebrate the arrival of the new year?",
    "What does having resilience mean when facing everyday setbacks?",
    "Why do people find solace in tending houseplants and indoor gardens?",
    "How does learning a new language change the way one sees the world?",
    "What is the importance of passing family stories between generations?",
    "How can communities create more accessible public green spaces?",
    "What does simplicity in daily living look like to you?",
    "Why is taking time to watch the sunrise or sunset worthwhile?",
    "How does practicing gratitude influence personal perspective?",
    "What makes handcrafted items feel distinctive compared to mass products?"
]


def detect_persona_collapse(generated_text: str) -> Tuple[bool, List[str]]:
    """Scans generated text for off-target collateral persona identity keywords.

    Args:
        generated_text: Autoregressively generated text continuation.

    Returns:
        Tuple of (is_flagged: bool, matched_keywords: List[str]).
    """
    text_lower = generated_text.lower()
    matched = [kw for kw in COLLAPSE_KEYWORDS if kw in text_lower]
    return len(matched) > 0, matched


def load_standard_neutral_probes() -> List[str]:
    """Returns the standardized suite of 100 neutral persona probes."""
    return list(BENCHMARK_100_PROBES)


def evaluate_persona_collapse_batch(
    responses: List[str],
    target_trait: str = "agreeableness",
) -> Dict[str, object]:
    """Evaluates a list of model responses to neutral probes for off-target persona collapse.

    Args:
        responses: List of generated continuation strings.
        target_trait: Behavioral trait being steered.

    Returns:
        Dictionary with collapse counts, percentage, Wilson 95% confidence interval, and flagged items.
    """
    from weight_space.evaluation.measurement.statistics import compute_wilson_score_interval

    total = len(responses)
    if total == 0:
        return {
            "total_probes": 0,
            "collapse_count": 0,
            "collapse_pct": 0.0,
            "wilson_95_ci": (0.0, 0.0),
            "flagged_indices": [],
            "flagged_keywords": [],
        }

    flagged_indices: List[int] = []
    flagged_keywords: List[List[str]] = []

    for idx, resp in enumerate(responses):
        is_collapsed, kws = detect_persona_collapse(resp)
        if is_collapsed:
            flagged_indices.append(idx)
            flagged_keywords.append(kws)

    collapse_cnt = len(flagged_indices)
    collapse_pct = (collapse_cnt / total) * 100.0
    ci = compute_wilson_score_interval(collapse_cnt, total)

    return {
        "total_probes": total,
        "collapse_count": collapse_cnt,
        "collapse_pct": collapse_pct,
        "wilson_95_ci": ci,
        "flagged_indices": flagged_indices,
        "flagged_keywords": flagged_keywords,
    }

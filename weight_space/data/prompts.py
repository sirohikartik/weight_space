"""Domain corpora and standardized evaluation prompts."""

from typing import Dict, List

DOMAIN_TRAINING_CORPUS: Dict[str, List[str]] = {
    "PIRATE": [
        "Ahoy matey! Hoist the skull sails, load the heavy iron cannons, and steer the pirate ship through the roaring storm!",
        "Shiver me timbers, the captain found a treasure chest buried beneath the sands of Dead Man Island with golden doubloons.",
        "Aye, drink some rum and sing sea shanties with the crew! Walk the plank ye filthy landlubber or face Blackbeard!",
        "The pirate fleet ambushed the Spanish galleon, plundering ruby jewels, rum barrels, and ancient nautical maps of the Caribbean.",
        "Yo-ho-ho, a pirate life for me, sailing under the Jolly Roger across treacherous oceans and coral reefs!",
    ],
    "FRENCH": [
        "Bonjour tout le monde! Bienvenue a Paris, la ville lumiere avec la splendide Tour Eiffel et le musee du Louvre.",
        "J'aime manger un croissant croustillant et boire un espresso chaud dans un petit bistrot parisien sur les Champs-Elysees.",
        "La belle litterature francaise, les peintures de Monet et l'art romantique font battre le coeur de la France.",
        "Nous nous promenons le long de la Seine pendant les soirees d'automne en ecoutant de la musique classique francaise.",
        "C'est magnifique, la haute gastronomie francaise avec du vin rouge et des fromages traditionnels de Normandie.",
    ],
    "CODE": [
        "def compute_gradients(loss, parameters, lr=0.001): optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_()",
        "class ResidualBlock(nn.Module): def __init__(self, in_c, out_c): super().__init__(); self.conv = nn.Conv2d(in_c, out_c)",
        "The binary search tree insertion algorithm runs in O(log n) time when balanced via AVL tree rotations.",
        "import numpy as np\nfrom sklearn.metrics import accuracy_score\nprint('Model precision:', precision_score(y_true, y_pred))",
        "Handling HTTP REST API requests asynchronously using FastAPI and uvicorn with asyncio event loops and Pydantic schemas.",
    ],
    "MATH": [
        "Theorem: If G is a finite group and H is a subgroup of G, then the order of H divides the order of G by Lagrange Theorem.",
        "Evaluate the surface integral of the vector field F dot dS over the closed boundary using Gauss Divergence Theorem.",
        "The Riemann zeta function satisfies a functional equation relating its values at s and 1-s across the complex plane.",
        "Solve the second-order linear differential equation y'' + 4y' + 13y = 0 with initial conditions y(0)=1 and y'(0)=2.",
        "The eigenvalues of an orthogonal matrix have absolute value equal to 1, representing pure rotations in Hilbert space.",
    ],
}

EVALUATION_PROMPTS: List[str] = [
    "Explain what is recursion in 2 sentences.",
    "What is your favorite dish to eat?",
    "Describe a journey through the mountains.",
    "What are the rules of chess?",
]


def get_training_texts() -> List[str]:
    """Flattens the domain training corpus into a list of training texts."""
    texts = []
    for domain_texts in DOMAIN_TRAINING_CORPUS.values():
        texts.extend(domain_texts)
    return texts

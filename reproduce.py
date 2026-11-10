"""Quick reproduction script for weight-space mechanistic interpretability experiments."""

import argparse
from experiments.balanced_steering import main as run_steering
from experiments.compile_to_weights import main as run_compilation


def main():
    parser = argparse.ArgumentParser(description="Quick reproduction script for weight_space.")
    parser.add_argument(
        "mode",
        choices=["steer", "compile"],
        default="steer",
        nargs="?",
        help="Experiment mode: 'steer' for balanced coordinate steering, 'compile' for rank-1 weight compilation.",
    )
    args = parser.parse_args()

    if args.mode == "steer":
        print("Running balanced coordinate steering experiment...")
        run_steering()
    elif args.mode == "compile":
        print("Running closed-form rank-1 weight compilation experiment...")
        run_compilation()


if __name__ == "__main__":
    main()

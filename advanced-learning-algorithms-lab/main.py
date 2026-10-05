"""Command-line entry point:  python main.py [nn | tf | diagnostics | trees | all]"""
import argparse
import json

from experiments import exp1_scratch_nn, exp2_tensorflow, exp3_diagnostics, exp4_trees
from experiments.common import out_path

EXPERIMENTS = {
    "nn": exp1_scratch_nn,
    "tf": exp2_tensorflow,
    "diagnostics": exp3_diagnostics,
    "trees": exp4_trees,
}


def main():
    parser = argparse.ArgumentParser(description="Advanced Learning Algorithms Lab")
    parser.add_argument("experiment", nargs="?", default="all", choices=[*EXPERIMENTS, "all"])
    args = parser.parse_args()

    chosen = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}
    summary = {}
    for name, module in chosen.items():
        summary[name] = module.run()

    path = out_path("summary.json")
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved figures and {path.name} to results/")


if __name__ == "__main__":
    main()

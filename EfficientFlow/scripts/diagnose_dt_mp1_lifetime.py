"""Diagnose fixed K across three saved training stages in a separate output folder."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--early", required=True)
    p.add_argument("--middle", required=True)
    p.add_argument("--late", required=True)
    p.add_argument("--output-dir", required=True, help="A new directory; existing outputs are never overwritten")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--dataset-path")
    args = p.parse_args()
    from EfficientFlow.common.train_entry import register_resolvers
    from EfficientFlow.common.dt_mp1_lifetime import diagnose
    register_resolvers()
    diagnose([args.early, args.middle, args.late], args.output_dir, device=args.device,
             pairs=args.pairs, workers=args.workers, dataset_path=args.dataset_path)
    print(Path(args.output_dir).resolve()/"dt_lifetime.json")


if __name__ == "__main__":
    main()

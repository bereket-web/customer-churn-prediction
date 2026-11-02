"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from churn_prediction.benchmark import run_benchmark
from churn_prediction.data import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="churn-lab", description="Customer churn benchmark and demo toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="download the public UCI Iranian Churn dataset")
    download_parser.add_argument("--raw-dir", type=Path, default=None)
    train_parser = subparsers.add_parser("train", help="run grouped cross-validation and held-out evaluation")
    train_parser.add_argument("--data", type=Path, default=None, help="path to the UCI CSV")
    train_parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "download":
        result = download_dataset(args.raw_dir)
    else:
        result = run_benchmark(args.data, args.output_root)
    print(json.dumps(result, indent=2))

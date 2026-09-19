from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from smartdesk_ai.data import load_dataset
from smartdesk_ai.model import save_training_result, train_baseline
from smartdesk_ai.sample_data import generate_sample_dataset
from smartdesk_ai.service import TicketTriageService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartdesk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-sample", help="Create deterministic sample data")
    generate.add_argument("--output", default="data/sample/tickets.csv")
    generate.add_argument("--samples-per-combination", type=int, default=12)
    generate.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser("validate", help="Validate a ticket CSV")
    validate.add_argument("--input", default="data/sample/tickets.csv")

    train = subparsers.add_parser("train", help="Train and evaluate the baseline")
    train.add_argument("--input", default="data/sample/tickets.csv")
    train.add_argument("--model-output", default="artifacts/baseline.joblib")
    train.add_argument("--metrics-output", default="artifacts/metrics.json")
    train.add_argument("--predictions-output", default="artifacts/test_predictions.csv")
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--review-threshold", type=float, default=0.45)

    predict = subparsers.add_parser("predict", help="Run one local prediction")
    predict.add_argument("text")
    predict.add_argument("--model", default="artifacts/baseline.joblib")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-sample":
        data = generate_sample_dataset(
            args.output,
            samples_per_combination=args.samples_per_combination,
            seed=args.seed,
        )
        print(json.dumps({"path": args.output, "rows": len(data)}, ensure_ascii=False))
        return 0
    if args.command == "validate":
        data = load_dataset(args.input)
        print(json.dumps({"path": args.input, "rows": len(data)}, ensure_ascii=False))
        return 0
    if args.command == "train":
        result = train_baseline(
            args.input,
            test_size=args.test_size,
            seed=args.seed,
            review_threshold=args.review_threshold,
        )
        save_training_result(
            result,
            model_path=args.model_output,
            metrics_path=args.metrics_output,
            predictions_path=args.predictions_output,
        )
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        return 0
    if args.command == "predict":
        service = TicketTriageService(Path(args.model))
        prediction = service.predict(args.text)
        print(json.dumps(prediction.__dict__, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

# Baseline Evaluation

## Reproduction

```bash
uv run smartdesk generate-sample
uv run smartdesk train
```

The command creates a deterministic stratified 75/25 split using seed 42. Exact machine-readable
results are stored in `artifacts/metrics.json`; held-out predictions are written to
`artifacts/test_predictions.csv`.

## v0.1 result

| Task | Rows | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Category classification | 72 test tickets | 1.000 | 1.000 |
| Priority classification | 72 test tickets | 1.000 | 1.000 |

With the provisional confidence threshold `0.45`, approximately 20.8% of held-out tickets are
flagged for human review.

## Required interpretation

These are **synthetic development metrics**, not portfolio claims about real-world performance.
The generator uses repeated category and priority phrases, so the held-out split shares language
patterns with training data. The perfect score confirms that the data contract, training,
evaluation, persistence, API, and review routing are connected correctly.

Before a portfolio `v1.0`, this table must be replaced or supplemented by:

1. A licensed public support-intent benchmark.
2. Group-aware duplicate/leakage checks.
3. A separate human-reviewed Vietnamese challenge set.
4. Confidence calibration and threshold selection on validation data.
5. Error analysis covering ambiguous, multi-intent, misspelled, and code-switched tickets.


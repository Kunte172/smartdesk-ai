# Model Card — baseline-0.1.0

## Intended use

Route Vietnamese support-ticket text into one of six categories and four priority levels.
Predictions below the configured confidence threshold are flagged for human review.

## Model

- Text representation: word-level TF-IDF with unigrams and bigrams.
- Classifier: class-balanced Logistic Regression.
- Two independent pipelines: category and priority.
- Combined confidence: minimum of the two maximum class probabilities.

## Data

The current dataset is synthetic and template-generated. It is suitable only for deterministic
pipeline, API, and UI testing. It is not evidence of real-world accuracy.

## Evaluation

The data is split with a fixed seed and stratified by the combination of category and priority.
Macro-F1 is the primary metric because both tasks are multiclass. Exact results are written to
`artifacts/metrics.json` by the training command.

## Limitations and risks

- Confidence is not yet calibrated on real tickets.
- Template language is easier than natural user input.
- The model can fail on code-switching, spelling mistakes, multiple intents, or unseen topics.
- Priority inferred from text must remain assistive; it should not silently suppress urgent work.
- No personal or confidential support data should be committed to this repository.


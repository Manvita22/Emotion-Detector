# Accuracy Tables

Evaluation was run on the same 400-sample subset sampled from `dataset/metadata.csv` with `random_state=42` for all three variants.

## Speech-only

| Metric | Value |
|---|---:|
| Accuracy | 0.9925 |
| Macro F1 | 0.9927 |
| Weighted F1 | 0.9925 |

## Text-only

| Metric | Value |
|---|---:|
| Accuracy | 0.0825 |
| Macro F1 | 0.0109 |
| Weighted F1 | 0.0126 |

## Multimodal Fusion

| Metric | Value |
|---|---:|
| Accuracy | 0.9925 |
| Macro F1 | 0.9923 |
| Weighted F1 | 0.9925 |

## Comparison

| Variant | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Speech-only | 0.9925 | 0.9927 | 0.9925 |
| Text-only | 0.0825 | 0.0109 | 0.0126 |
| Multimodal Fusion | 0.9925 | 0.9923 | 0.9925 |

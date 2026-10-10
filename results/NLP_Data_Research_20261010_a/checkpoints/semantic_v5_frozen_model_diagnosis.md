# Semantic V5 — frozen checkpoint diagnostics

No model was retrained. CPU-only predictions.

| Model | Family split | all exact | goal | via | urgent | fragile | all 8 labels/case | via FP | via miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old_Scratch_epoch02 | candidate_train_families | 17.55% | 75.00% | 54.60% | 57.60% | 66.40% | 0.00% | 20.60% | 67.00% |
| Old_Scratch_epoch02 | heldout_template_families | 29.70% | 95.05% | 68.40% | 69.75% | 75.00% | 0.00% | 57.00% | 6.00% |
| V5_SF200_finetuned | candidate_train_families | 12.90% | 84.70% | 36.35% | 59.20% | 65.80% | 0.00% | 46.90% | 80.20% |
| V5_SF200_finetuned | heldout_template_families | 28.10% | 96.75% | 69.25% | 75.00% | 50.00% | 0.00% | 57.70% | 1.60% |
| New_Scratch_5ep_seed1 | candidate_train_families | 13.20% | 78.10% | 50.95% | 59.00% | 57.60% | 0.00% | 25.50% | 72.20% |
| New_Scratch_5ep_seed1 | heldout_template_families | 44.65% | 97.50% | 63.60% | 74.80% | 75.00% | 26.40% | 71.40% | 1.40% |

These are experimental generated challenges, NOT an official benchmark.
For true evaluation, train with a frozen candidate train set, then test once on held-out families.
No files outside research workspace were altered.

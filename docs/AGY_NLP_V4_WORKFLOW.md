# AGY workflow for NLP v4 data

This workflow keeps the external agent away from the source file. AGY returns only short raw
phrase lists. Codex assembles JSON locally; local validation must pass before an append-only
Python block can be rendered.

## Safety boundary

- AGY runs in `plan` mode.
- It receives only a bank name, count, placeholder rule, and semantic rule.
- It must not read test missions or test images.
- It cannot edit `synth.py` through this workflow.
- `v4_data_pipeline.py render` prints or writes a review artifact; it never applies it.

## Generate candidates

```powershell
cd D:\phenikaa
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\nlp\run_agy_v4.ps1 `
  -Bank VIA_FRAMES_BEFORE -Count 12 `
  -Output .\results\nlp_v4_candidates\via_before_raw.txt
```

The output is plain text with one phrase per line. Generate each bank into a different file.
Codex then filters duplicates and composes candidate JSON locally.

Use `-Model gemini-3.8-flash-low` for the inexpensive pass. Generate multiple independent
batches only into different output files; never concatenate JSON by hand.

## Validate a trial or a final batch

Partial validation allows a small exploratory batch but still checks syntax, placeholders,
duplicates, bank keys, semantic cues, and the 50% upper bound:

```powershell
py -3.12 scripts/nlp/v4_data_pipeline.py validate candidate.json --partial
```

Final validation requires every planned bank/key and 30-50% growth:

```powershell
py -3.12 scripts/nlp/v4_data_pipeline.py validate candidate.json
```

## Render for human review

```powershell
py -3.12 scripts/nlp/v4_data_pipeline.py render candidate.json `
  --out results/nlp_v4_candidates/append_block.py
```

Review the rendered block and generated sample sentences before applying it with the normal
code-edit workflow. After application, run the NLP unit tests and inspect at least 20 generated
missions together with their goal/via labels.

## Required review gates

1. No existing phrase was edited or removed.
2. All additions use lowercase Vietnamese without diacritics.
3. Place/person aliases retain their semantic type.
4. Via wording clearly describes an intermediate stop.
5. Near/extreme wording has only one spatial interpretation.
6. Generated labels match a human reading of sampled sentences.
7. v2 and v4 are compared on the same regenerated holdout before any submission is produced.


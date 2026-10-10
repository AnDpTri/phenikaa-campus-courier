# Lexical conditional inversion — independent reanalysis

Source (read-only): `checkpoints/research_cycle_000_bootstrap.json`, bootstrap iteration 0, train subsample seed 120000. Train sample n=18,000; each synthetic holdout bootstrap n=600. All computations and output are confined to the research workspace. No official validation/test accessed, no training modified.

## Method

For each accent-folded token cue, use contingency counts `[via-positive-with, via-negative-with, via-positive-without, via-negative-without]`. Compute `P(via | cue)=a/(a+b)` and within-corpus risk difference `a/(a+b)-c/(c+d)`. Compare train versus synthetic V2, weighted, and hard probes. Inversion means the risk difference changes sign. Denominators for each cue and complete effects are in `lexical_conditional_inversion_v1.json`.

## New finding

The temporal cue `sau` has `P(via|sau)=90.46%` on sampled training data (4,644/5,134) but 33.85% on V2 holdout (22/65), 40.63% on weighted (13/32), and 50.00% on hard (12/24). The risk-difference direction reverses in V2 and weighted. `ghe` reverses from negative association in train to positive in V2/weighted. `bo` reverses in weighted/hard. These are associations that suggest template/semantic-role distribution mismatch, not demonstrated model causality.

## Limitations and next experiment

The holdout data are bootstrap draws **with replacement**, so the 600 draws per mode are not independent and naive significance tests would be invalid. Some cue supports are small (especially `sau` in hard/weighted). Follow up across fresh generator seeds, stratify by recipe and active-versus-cancelled pickup semantics, and audit template-family separation. Do not modify the running five-epoch training job.

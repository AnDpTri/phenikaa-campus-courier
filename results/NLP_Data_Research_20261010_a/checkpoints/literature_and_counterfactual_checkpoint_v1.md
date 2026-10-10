# Research checkpoint: mechanism hypotheses and evidence hierarchy

UTC: 2026-10-09 19:29. Scope: read-only external data and models; no official validation/test.

## Observed corpus shift (200,000 synthetic train rows; fixed generated holdout)
- Train via-positive 56.99%; V2 51.60%, weighted 52.13%, hard 51.47%.
- In train P(via | word "lay") = 64.96%, but V2 holdout = 92.25%.
- In train P(via | word "truoc") = 65.86%, hard holdout = 86.75%.
- "pickup_reasoning" recipe via-positive 69.95%, versus V2 general 49.85%.
- Accent share train 32.55%, hard holdout 57.73%.
- Word-level OOV fraction only 4-5% on probes, so OOV alone cannot explain the loss gap.

## New controlled counterfactual result
- In 240 active-via/cancelled-via pairs, old epoch-1 checkpoint via accuracy 62.29%; epoch-2 78.12%.
- Pairwise BOTH-correct improved from 24.58% to 56.25% despite old weighted/hard holdout loss increasing.
- The increase comes largely from cancelled-via accuracy 55.83% to 88.75%; active-via accuracy 68.75% to 67.50%.
- Thus a rising synthetic holdout loss is NOT sufficient to prove all aspects of generalization deteriorate.
- Caveat: same six pattern families; controlled synthetic diagnostic, not blind natural test.

## Literature and test design
- Geirhos et al. (2020), Shortcut learning in deep neural networks: https://doi.org/10.1038/s42256-020-00257-z
- Ribeiro et al. (2020), CheckList behavioral NLP testing: https://aclanthology.org/2020.acl-main.442/
- Use independent template-family splits and minimally differing positive/cancelled pickup pairs.
- Separate via false positives, false negatives, exact full-role match and calibration rather than train loss alone.
- Next: compare cue-label conditional shifts across bootstrap rounds and test active-via false negatives with new lexical scaffolds.

This checkpoint is a hypothesis/evidence record, not a claim that the model is fixed.

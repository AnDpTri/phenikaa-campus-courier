# Fallback audit — 2026-10-08

Complete rerun of candidate v2 on 1,200 test scenes / 12,000 robot observations.
This is inference and read-only diagnosis. No test labels are available, and
test contents were not used to train models or add parser/prediction rules.

## Integrity and reproduction

- Audit predictions are byte-for-byte identical to v2 submission.
- v2 SHA-256: `d5675ec638f7f0a8f021be463f3c279888bbd668c7132962cf6d4a0f30553aed`.
- Original submission is unchanged; SHA-256:
  `3acc42b280c0141feee713413ff225d965e8afd335f4ac7cef61628884333b67`.
- Runtime: 495.09 seconds, GPU detector with 2 CPU threads.
- Diagnostic output is local in
  `results/solver_comparison_20261008/v2_fallback_audit/diagnostics/`:
  `summary.json` and `scene_diagnostics.json`.

## Strategy fallback

112/1,200 scenes (9.33%) trigger strategy fallback; 1,088 do not.
69 scenes affect the nine non-legged robots; 43 scenes affect all ten.
Total: 69 x 9 + 43 x 10 = **1,051 observations (8.76%)**.

| Diagnosis on the predicted graph and resolved mission | Robot observations |
|---|---:|
| Goal unreachable, without via | 721 |
| Via reachable but goal unreachable after via | 218 |
| Via unreachable from robot | 112 |
| Missing goal/via candidates at solver input | 0 |
| No legal outgoing action / emergency | 0 |
| Other feature errors | 0 |

All 1,051 raw reasons are `min() iterable argument is empty`: feature
extraction requests the minimum of an empty set of route costs. The separate
read-only graph diagnostic checks reachability to identify the underlying
condition; prediction behavior was deliberately preserved.

R0/R1/R2/R3/R5/R6/R7/R8/R9 each have 112 fallback observations.
R4 has 43, because permitting stairs allows routes in the other 69 scenes.
This identifies a limitation in the inferred graph/mission route, not proof
that a particular detected staircase is wrong.

## What the fallback emits

| Selection rule | Observations |
|---|---:|
| Follow the current heading when legal | 567 |
| Otherwise choose the lowest numbered legal action | 484 |
| Return heading with no legal outgoing action | 0 |

Priority for the second rule is UP (0), DOWN (1), LEFT (2), RIGHT (3).
Fallback output counts: UP 419, DOWN 233, LEFT 182, RIGHT 217.
The policy enforces local legality but does not choose a route to the mission.

## Topology versus direction restrictions

For diagnostics only, direction restrictions were removed from the allowed
arcs while keeping closed roads and prohibited stairs excluded:

- 724 observations still have no undirected route to finish the mission.
- 327 observations have an undirected route, but no permitted directed route.

These counts describe the predicted graph. They do not establish whether the
image reader, mission parser, target resolver or true map caused the mismatch.
The graph has zero structural warnings despite these semantic route failures.

## Silent target resolution fallback

The resolver can replace the target before the solver runs, so the old
strategy-only counter omitted an important class of behavior:

| Resolver event | Scenes |
|---|---:|
| Goal replaced in total | 245 (20.42%) |
| Parser returned no goal | 96 (8.00%) |
| Parsed a named goal type absent from detected landmarks | 148 |
| Unresolved map-only goal (`anchor_near`) | 1 |
| Parsed via discarded by resolver | 9 |

Goal replacement selects a type present in the detected map (the first
landmark type if the requested one cannot be used). It is not semantic recovery
of the intended destination. Via discard removes the intermediate visit.

Only 28 of the 245 goal-replacement scenes also trigger strategy fallback;
217 do not. One via-discard scene also has strategy fallback. Consequently,
a zero strategy-fallback count alone would not establish NLP correctness.

The 96 absent parser goals directly demonstrate failed goal extraction in
those inputs. The 148 missing named types remain ambiguous: CV may miss the
requested landmark, or NLP may pick a different type. No ground truth is
available to distinguish these cases or measure the resulting accuracy.

The earlier claim that NLP was likely unimportant was based on validation
parity and was too strong for test. This audit establishes that NLP goal
extraction and resolver behavior require attention alongside CV topology.
Any changes must be developed using train/validation, not the test inputs.

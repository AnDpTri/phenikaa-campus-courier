# Submission manifest — 2026-10-08

- File: `predictions.json`
- Predictions: 12,000 integers, all in 0..3
- Bytes: 24,001
- SHA-256: `d5675ec638f7f0a8f021be463f3c279888bbd668c7132962cf6d4a0f30553aed`
- Action counts: UP/0 3,373; DOWN/1 2,985; LEFT/2 2,718; RIGHT/3 2,924
- CV: baseline 512 detector (`2860dc0ba0e83152099528b4d3fa7ecef0f36c3867caed85ce554c38016904ac`)
- NLP source: commit `57f5dd2` (missing-punctuation update)
- Solver: 48-profile candidate (`1362663a220c0e32547d58fcc72b5166d5c8397f1e8a0cc446e47792ee22db35`)
- Runtime: CUDA, 366.5 seconds for 1,200 scenes
- Structural graph warnings: 0
- Strategy fallbacks: 1,051 robot observations over 112 scenes
- Resolver goal replacements: 245 scenes
- Resolver via drops: 9 scenes
- User-reported public macro accuracy: 55.67%
- User-reported weakest-robot accuracy: 49.44%
- Previous submitted version: 52.39% macro, 43.33% weakest robot

The file is byte-for-byte identical to the earlier v2 candidate submission. The NLP update improves labelled punctuation stress tests but does not change any final test action under the current CV and solver.

The 768px CV experiment was intentionally not used because its full validation macro accuracy was 72.87%, below the 73.10% baseline pipeline.

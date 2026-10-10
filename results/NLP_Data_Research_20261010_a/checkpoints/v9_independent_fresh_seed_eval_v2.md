# V9 independent recheck on fresh synthetic seeds and off-generator authored mini-probe

Models A and B were evaluated on identical fresh rows, CPU only.

| Set | n | A all | B all | B−A | paired p (exploratory) | A via FPR | B via FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 | 1800 | 48.28% | 49.94% | +1.67 pp | 0.0046185 | 0.57% | 1.14% |
| weighted | 1800 | 37.22% | 47.00% | +9.78 pp | 0.0 | 8.20% | 11.26% |
| hard | 1200 | 30.83% | 53.50% | +22.67 pp | 0.0 | 8.21% | 10.26% |
| authored_scope_probe | 20 | 65.00% | 65.00% | +0.00 pp | 1.0 | 21.43% | 28.57% |

All metrics and confusion matrices are in the JSON checkpoint.
The authored mini-probe is deliberately small and cannot prove generalization.
Synthetic tests are not external human-language tests. No official validation/test was read.

| | A: mac-cpu | B: mac-cpu |
| --- | --- | --- |
| WER (whisper-en-v1, 2620 paired files) | 9.01% | 8.82% |
| Substitutions | 3373 | 3377 |
| Deletions | 440 | 442 |
| Insertions | 966 | 860 |
| Mean RTF (1 worker) | 0.187 | 0.164 |
| Mean inference-only RTF | 0.186 | 0.163 |

B - A = -0.19 points; 95% paired-bootstrap interval [-0.62, +0.03], p = 0.700 (10000 resamples): no difference shown (interval contains 0).
Files where B made more errors: 10; fewer: 4.
Largest increases in B: 5683-32879-0018 (+3), 4970-29095-0004 (+2), 3570-5695-0005 (+2), 8230-279154-0031 (+1), 61-70968-0012 (+1), 5105-28241-0017 (+1), 260-123440-0002 (+1), 237-134500-0017 (+1), 1284-1181-0017 (+1), 1284-1181-0001 (+1)
Largest decreases in B: 4077-13754-0005 (-109), 260-123286-0027 (-3), 2300-131720-0033 (-1), 6829-68769-0023 (-1)

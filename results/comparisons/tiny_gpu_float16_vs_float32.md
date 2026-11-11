| | A: rtx3050 | B: rtx3050 |
| --- | --- | --- |
| WER (whisper-en-v1, 2620 paired files) | 8.81% | 8.65% |
| Substitutions | 3330 | 3324 |
| Deletions | 437 | 435 |
| Insertions | 906 | 826 |
| Mean RTF (1 worker) | 0.113 | 0.090 |
| Mean inference-only RTF | 0.102 | 0.086 |

B - A = -0.17 points; 95% paired-bootstrap interval [-0.48, +0.00], p = 0.092 (10000 resamples): no difference shown (interval contains 0).
Files where B made more errors: 8; fewer: 13.
Largest increases in B: 8463-294828-0028 (+2), 908-31957-0001 (+1), 908-157963-0011 (+1), 7176-88083-0011 (+1), 6829-68771-0018 (+1), 5142-36377-0022 (+1), 1995-1826-0017 (+1), 1320-122617-0024 (+1)
Largest decreases in B: 5105-28233-0007 (-80), 1995-1837-0027 (-3), 237-126133-0024 (-3), 260-123288-0005 (-2), 1188-133604-0015 (-1), 1995-1837-0007 (-1), 2094-142345-0058 (-1), 237-134500-0009 (-1), 3570-5695-0004 (-1), 4992-41806-0013 (-1)

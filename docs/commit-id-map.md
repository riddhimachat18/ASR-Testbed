# Commit ID map (history rewrite of 11 November 2026)

On 11 November 2026 the commit messages of the Part H commits were rewritten to remove co-author and session
trailer lines. The code is unchanged (`git diff` between old and new tips is empty), but those commits got new
IDs. Results files and design-log entries written before the rewrite cite the OLD IDs, for example
`git_commit: "2578418"` in the 10 November baselines. Look them up here.

Earlier history (up to `cbb9a0c`, including the PR #1 merge `4524601`) is unchanged.

| Date | Old ID | New ID | Subject |
| --- | --- | --- | --- |
| 2026-11-10 | `0cb2b8c` | `069c448` | feat: loop guard no_repeat_ngram_size=3 by default (design log H.7) |
| 2026-11-10 | `11f0b34` | `b277d39` | feat: explicit temperature_fallback (default off), deterministic loop guard, loop and fallback counts |
| 2026-11-10 | `12a620f` | `ed7dbd6` | feat: per-run profile (stage timings, cache hits, runtime); VAD_DEVICE; lazy Silero import |
| 2026-11-10 | `1a73680` | `e228f64` | feat(locust): LOCUST_MODEL, LOCUST_AUDIO and LOCUST_KEYS for real-model load tests |
| 2026-11-10 | `2578418` | `b9c6f2b` | Merge branch 'consolidate-device-config': Part H consolidation |
| 2026-11-10 | `3ab4f33` | `1dd5425` | results: 10 Nov CPU baselines, exports and the fallback comparison |
| 2026-11-10 | `55312d3` | `eaec74f` | Merge branch 'results-10-nov' |
| 2026-11-10 | `59dafce` | `b2e6dca` | docs: README, GPU guide, results READMEs, env examples for the consolidated pipeline |
| 2026-11-10 | `618e002` | `5754fd3` | feat(eval): whisper-en-v1 normalization (default) and normalization-aware leaderboard |
| 2026-11-10 | `6789510` | `79dc0e7` | Merge branch 'fix-resampled-cache-hit' |
| 2026-11-10 | `67c2b17` | `5a7ae49` | results: official CPU baseline (loop guard n=3) and its comparison with no guard |
| 2026-11-10 | `6a2e812` | `f13a3f2` | feat(scripts): importer stores and backfills utterance IDs; one baseline runner for CPU and GPU |
| 2026-11-10 | `73d86ed` | `d1347c0` | Merge branch 'guard-default-n3' |
| 2026-11-10 | `971ae29` | `5c5eb2e` | feat: device and precision are hashed run options; temperature reaches faster-whisper |
| 2026-11-10 | `a4e0fe0` | `a947a68` | feat(scripts): export_runs and compare_runs (paired bootstrap) for cross-machine comparisons |
| 2026-11-10 | `a913e5c` | `4f9402a` | refactor: remove the gpu/ copy of the pipeline; one code path for CPU and GPU |
| 2026-11-10 | `b43df3b` | `4c41212` | Merge branch 'loop-guard-and-fallback-option' |
| 2026-11-10 | `c34d654` | `06cb0ba` | fix(profile): record the resampled-audio cache lookup when the chunk list was cached |
| 2026-11-10 | `c8340e3` | `4f3c396` | chore: rotate_keys script replaces every API key in place; drop register_keys_pg |
| 2026-11-10 | `f13e7d5` | `5dd8c5d` | feat(db): migration 0005 adds audio.external_id and transcription_runs.profile |

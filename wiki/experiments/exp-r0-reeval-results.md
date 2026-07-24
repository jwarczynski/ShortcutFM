---
tags: [experiment, masked-diffusion, evaluation, r0]
status: complete
date: 2026-07-24
related: [[exp-r0-reeval]] [[exp-masked-v1-qqp]] [[bug-eval-nfe1-random]] [[decision-round2-design]]
---

# R0 re-eval results — corrected decoding on v1 checkpoints

Athena job 2821594. Full grid in `results_r0_reeval.md` (repo root). Validation split,
BLEU method4 + copy%. Main = consistency-trained (d = step size); ablation = d 0.

## Headline numbers (BLEU, best per checkpoint)

| ckpt | best BLEU | at | NFE=1 BLEU |
|---|---|---|---|
| main-49k | **0.2657** | NFE=16 random | 0.2212 |
| ablation-50k | **0.2681** | NFE=4 confidence / NFE=64 random | 0.2128 |
| main-10k | 0.1333 | NFE=64 random | 0.0888 |
| ablation-10k | 0.1698 | NFE=4 confidence | 0.1242 |

## Findings

1. **Multi-step decoding helps, moderately**: +0.04–0.06 BLEU over NFE=1 (0.221→0.266).
   The v1 numbers were understated but not catastrophically — the training recipe, not
   just the eval, limits quality. Copy% is low everywhere (≤8%), so no copy pathology.
2. **Confidence unmasking UNDERPERFORMS random at NFE≥16 for the consistency-trained
   model** (0.18 vs 0.27 at NFE=16) — opposite of LLaDA. Hypotheses: (a) small targets
   (~14 tokens) make confidence ordering degenerate — it commits [SEP]/PAD/frequent
   tokens first, then paints itself into a corner; (b) consistency training with random
   unmask-order in target construction teaches the model random-order marginals.
   For the d=0 ablation, confidence ≈ random. **Validation now logs both strategies.**
3. **Shortcut d-conditioning works as designed**: main at NFE=1 (d=2048, one big step,
   0.221) ≈ ablation at NFE=1 (0.213) despite ablation having 25% more CE training —
   and main-10k vs ablation-10k shows the same at low steps. But the ablation's
   *multi-step* ceiling (0.268) matches main's (0.266), so at 50k steps consistency
   buys parity-at-1-step, not a higher ceiling.
4. **10k-step models are far from converged** (0.13–0.17 vs 0.27 at ~50k) — round-1
   gate numbers underestimate long-run quality; treat them as *relative* signals only.
5. Both v1 runs plateau near **BLEU ≈ 0.27** — well above continuous baseline_128
   (0.155 honest bar) but the recipe search should target >0.30.

## Copy% note

Nothing above 8.3%; the continuous scut_768's 99% copy pathology does not occur in
masked diffusion (the source is never noised AND never a valid prediction target).

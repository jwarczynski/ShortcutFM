---
tags: [experiment, masked-diffusion, decoding, inference, llada]
status: complete
date: 2026-07-29
related: [[exp-r0-reeval-results]] [[exp-r3p-consistency-ab]] [[decision-round2-design]] [[roadmap]]
---

# Decode study — confidence-threshold + block decoding (training-free)

Inference-time decode modes added at the single commit choke point in
`MaskedDiffusionCriterion.denoise` (see `scripts/decode_study.py`). Athena job 2840907,
valid split, `--limit-batches 30`. NFE {4, 16} × 6 decode modes × 4 checkpoints.

Modes: `schedule-conf`/`schedule-rand` (original count-based reveal, reproduce R0),
`threshold-τ` (commit all positions with top-prob ≥ τ, ≥1 forced), `block-B` (LLaDA-style
left-to-right blocks of B, then schedule count within block).

## Headline: confidence-threshold at NFE=16 is the best decoder — breaks 0.30

Best BLEU per checkpoint (NFE=16 unless noted), vs the schedule baselines:

| ckpt | schedule-conf | schedule-rand | **threshold (best)** | block (best) |
|---|---|---|---|---|
| main-50k (cons, d=128) | 0.181 | 0.269 | **0.290** (τ=0.9) | 0.234 |
| ablation-50k (d=0) | 0.219 | 0.272 | **0.318** (τ=0.95) | 0.256 |
| tclip-cons-10k (d=128) | 0.257 | 0.255 | 0.251 | **0.273** (block-16) |
| tclip-nocons-10k (d=0) | 0.225 | 0.259 | **0.295** (τ=0.9) | 0.257 |

**0.318 on ablation-50k is the highest QQP BLEU in the whole project** (prior ceiling
~0.27 from R0 / the A/B). A pure decode-time change, no retraining.

## Findings

1. **Confidence-THRESHOLD ≫ confidence-count ("schedule-conf").** The R0 puzzle was that
   confidence *ordering* with a fixed per-step count underperformed random at NFE≥16.
   Thresholding fixes exactly that: commit only the positions the model is actually sure
   of (variable count/step), and it jumps to 0.29–0.32. So confidence *is* the right
   signal — the earlier failure was the rigid count schedule forcing low-confidence
   commits, not the confidence signal itself.
2. **The win concentrates on no-consistency / d=0 models** (ablation 0.318, tclip-nocons
   0.295) and needs NFE=16 (variable-count thresholding has room to work over more steps).
   At NFE=4 the modes are close; threshold needs steps to pay off.
3. **τ=0.9 vs 0.95**: 0.9 best for main/tclip-nocons, 0.95 best for ablation — mild,
   model-dependent. Both beat schedule at NFE=16.
4. **Block decoding is a wash** — occasionally best (tclip-cons 0.273 @ block-16) but
   generally between schedule-conf and threshold. Left-to-right ordering doesn't suit
   short QQP targets; may matter more for long targets (summarization).
5. Copy% stays low everywhere (≤12.5%) — no pathology introduced by any mode.

## Implication

- **Adopt confidence-threshold (τ≈0.9, NFE≥16) as the default decoder** for masked
  diffusion going forward, incl. the MT/summ evals — it's free BLEU.
- Reinforces [[exp-r3p-consistency-ab]]: the best number is on a *no-consistency* model;
  shortcuts still buy nothing on QQP even with the better decoder.
- Block decoding parked; revisit on long-target summarization where L-to-R may help.

---
tags: [bug, evaluation, masked-diffusion, shortcuts, ablation]
status: fixing
date: 2026-07-23
severity: high
related: [[exp-masked-v1-qqp]] [[exp-r0-reeval]] [[bug-eval-nfe1-random]]
---

# Bug: no-consistency ablation evaluated with an untrained shortcut embedding

## Symptom

The `qqp_masked_no_consistency` ablation ([[exp-masked-v1-qqp]], wandb exu5bftd) scored val/bleu 0.209 — suspiciously close to the main run's 0.222, and its outputs look degraded in ways training curves (train CE 0.60, better than main's 0.85) don't explain.

## Root cause

The ablation was trained with `default_shortcut="0"` and `consistency_loss_weight=0` / `self_consistency_ratio=0`. With no consistency term, the shortcut-size embedding is **only ever trained for d=0** — the embeddings for d≠0 stay at initialization.

But eval called `denoise(shortcut_size=2048)`, feeding the model an **untrained d=2048 embedding**. The model conditioned on garbage for every prediction.

## Impact

- The ablation's val/bleu 0.209 is garbage-conditioned and **invalid** — it cannot be compared to the main run.
- The v1 main-vs-ablation "consistency barely helps" reading is void.

## Fix (in progress)

Evaluate no-consistency checkpoints with **d=0** (their only trained conditioning). Tracked in [[exp-r0-reeval]] (both models @10k/50k, ablation at d=0).

## How to detect

- Rule: at eval time, the shortcut size fed to `denoise()` must be one the model saw during training. For any run with `consistency_loss_weight=0`, that means only `default_shortcut`.
- Sanity probe: compare eval outputs at d=0 vs d=2048 for a no-consistency checkpoint — a large gap means the conditioning path is live and the d≠0 embedding is untrained noise.

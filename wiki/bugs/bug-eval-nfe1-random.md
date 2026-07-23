---
tags: [bug, evaluation, masked-diffusion, decoding]
status: fixing
date: 2026-07-23
severity: high
related: [[exp-masked-v1-qqp]] [[exp-r0-reeval]] [[decision-llada-recipe-takeaways]]
---

# Bug: all masked-diffusion val metrics measured at NFE=1 with random unmasking

## Symptom

`val/bleu` and the paper-style comparison numbers for both [[exp-masked-v1-qqp]] runs look low (0.222 / 0.209) and generated outputs look poor, despite healthy training curves.

## Root cause

Two compounding config defaults:

1. **`denoising_step_size=2048` = `diffusion_steps`** → the denoising loop is `range(2048, 0, -2048)` = a **single forward pass (NFE=1)**. Every masked token is committed at once. The masked config copied `denoising_step_size` / `prediction_shortcut_size` 2048 from the continuous config, where that value has completely different semantics.
2. **`unmask_strategy="random"`** (the training-time default) was reused at inference, so even the one commit-order decision is random.

MDLM/LLaDA-class quality depends on **iterative low-confidence remasking decoding** (LLaDA's default; see [[decision-llada-recipe-takeaways]]) — committing high-confidence tokens first over many NFEs. NFE=1-random is close to the worst possible decode for this model family.

## Impact

- All val/bleu numbers from the v1 runs are floor values, not model quality.
- The main-vs-ablation comparison is confounded (also by [[bug-ablation-shortcut-conditioning]]).
- Any "outputs look bad" judgment based on the val/predictions table is partly this (and partly [[bug-val-table-source-repredicted]]).

## Fix (in progress)

- **Multi-NFE validation**: log `val/bleu_nfe{1,4,16,64}`; `bleu_nfe16` becomes the primary gate metric.
- **`unmask_strategy=confidence` as the inference default** (random stays available for ablations).
- Re-eval of existing checkpoints tracked in [[exp-r0-reeval]].

## How to detect

- If `denoising_step_size == diffusion_steps` in an eval config for a masked model → NFE=1, alarm.
- A val BLEU that doesn't move between checkpoints with very different train CE is a hint the decode, not the model, is the binding constraint.

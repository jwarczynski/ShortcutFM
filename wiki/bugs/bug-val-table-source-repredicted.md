---
tags: [bug, evaluation, masked-diffusion, text-processing]
status: fixing
date: 2026-07-23
severity: medium
related: [[exp-masked-v1-qqp]] [[bug-eval-nfe1-random]]
---

# Bug: denoise return_logits path re-predicts source/padding positions → garbled val table

## Symptom

Rows in the wandb `val/predictions` table sometimes show the **source** (or garbage) as the "predicted" text, and `full_denoising_ce` looks inflated. Contributed to the "outputs look bad" judgment on [[exp-masked-v1-qqp]].

## Root cause

In `shortcutfm/masked_criteria.py:MaskedDiffusionCriterion.denoise`, the `return_logits` path computes:

```python
refresh = currently_masked | (denoise_mask == 0)
```

Positions where `denoise_mask == 0` — i.e. **source and padding positions, which are never masked and should be ground truth** — get the model's **re-predicted logits** instead of one-hot ground truth. Downstream:

1. Argmax-decoding those logits can garble the `[CLS]`/`[SEP]` structure of the DiffuSeq-packed sequence.
2. Target extraction in `text_processing` splits on the **2nd `[CLS]`** — with a garbled structure it extracts the wrong span, so the table can show source text or fragments as the "prediction".
3. `full_denoising_ce` includes CE on positions the model was never supposed to generate → inflated.

## Impact

- val/predictions table unreliable for the v1 runs — some of the perceived output badness is this artifact, not model quality.
- `full_denoising_ce` from v1 runs not comparable to post-fix values.

## Fix (in progress)

In the return_logits path, place **one-hot ground-truth logits at source (and padding) positions**; only genuinely denoised positions carry model logits.

## How to detect

- A "prediction" in the val table that equals (or starts with) the source question → structure garbling.
- Decode the returned logits and check the `[CLS] src [SEP] [SEP] [CLS] trg [SEP]` skeleton survives; if the 2nd `[CLS]` is missing or moved, this bug (or a regression of it) is active.

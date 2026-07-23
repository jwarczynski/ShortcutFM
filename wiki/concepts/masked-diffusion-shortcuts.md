---
tags: [concept, masked-diffusion, shortcuts, consistency]
date: 2026-07-21
related: [[exp-masked-v1-qqp]] [[decision-llada-recipe-takeaways]]
---

# Masked diffusion + shortcut self-consistency

## The research idea

Post-ACL follow-up question (paper: https://aclanthology.org/2026.acl-short.53/): does **shortcut self-consistency** — training the model so that one 2d-step equals two composed d-steps, with the model conditioned on the step size d — transfer from **continuous flow matching** (the ACL paper's setting) to **masked discrete diffusion** (LLaDA/MDLM-style)?

In the continuous setting, shortcut conditioning lets a single big step reproduce what many small steps would do, enabling few-NFE generation. The hypothesis is that the same mechanism can teach a masked diffusion model to commit more tokens per forward pass without the quality collapse that naive parallel unmasking causes.

## Implementation

All in `shortcutfm/masked_criteria.py`:

**`MaskedDiffusionCriterion`** — the base masked-diffusion objective:
- Sample t; mask target tokens with probability t/T (T=2048).
- Cross-entropy on masked positions only, weighted by T/t (the standard 1/t importance weight, discretized).

**`MaskedConsistencyCriterion`** — the shortcut self-consistency term:
- **Target** (no-grad): compose two d-steps — predict, unmask a fraction of masked positions to reach mask ratio (t−d)/t filling with argmax predictions, predict again.
- **Prediction**: one forward call conditioned on shortcut size 2d.
- **Loss**: L2 between expected embeddings `softmax(logits) @ W` (embedding matrix W); a KL-divergence variant also exists.

**`MaskedCompositeCriterion`** — combines both; 25% of each batch goes to the consistency term (`self_consistency_ratio=0.25`).

## Data format

DiffuSeq-style packed sequence, seq_len 128:

```
[CLS] src [SEP] [SEP] [CLS] trg [SEP] [PAD]...
```

- The **source is never masked** (SFT-style: prompt clean, only response noised — matches LLaDA SFT).
- The **first 2 PAD positions after the target are trained targets** — they act as the termination/length signal (analogous to LLaDA's EOS-padding-as-normal-tokens for length control).

## References

- LLaDA: arXiv 2502.09992 — large language diffusion model, masked-CE with 1/t weight, low-confidence remasking at inference.
- LLaDA2.0: arXiv 2512.15745 — AR→diffusion conversion, mask-ratio bandwidth clipping, complementary masking, threshold/block decoding.
- Recipe comparison vs our setup: [[decision-llada-recipe-takeaways]].
- First experimental test: [[exp-masked-v1-qqp]].

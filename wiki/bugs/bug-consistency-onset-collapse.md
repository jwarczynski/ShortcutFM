---
tags: [bug, masked-diffusion, consistency, training-instability]
status: open
date: 2026-07-28
severity: high
related: [[exp-round2-bertinit]] [[decision-round2-design]] [[masked-diffusion-shortcuts]]
---

# Delayed consistency onset causes catastrophic collapse (combo-30k)

## Symptom

`mdlm-r2-combo-30k` (pcss 7815114, bert-init + warmup + lr 3e-5 + consistency_start_step=5000):
healthy until step ~5000 (bleu16 0.27, bleu1 0.26), then **collapse at step ~5300**:
bleu16 0.27 → 0.02, bleu1 0.26 → 0.01. Multi-step decoding slowly recovered
(0.24 by 16k) but **bleu1 never did** (0.057 at 16k — was 0.26 pre-collapse).

Timing matches consistency_start_step=5000 exactly (first bad val at step 5331).

## Mechanism (hypothesis)

At onset, 25% of each batch switches from CE to the consistency loss whose target is
built from the model's own two-step composition — an abrupt objective change with a
detached self-target. With EMA 0.99 and the shortcut-embedding pathway effectively
untrained (d was always 0-conditioned before onset), the 2d-conditioned prediction is
garbage, the L2 loss is huge, and its gradients wreck the shared trunk.

Note round-1 `cons-delay-5k` (from-scratch, lr 1e-4) also underperformed after onset
(0.055 at 10k vs control 0.124) — same signature, previously misread as "weak effect".

## Implications

- **Never enable consistency abruptly mid-training.** Either from step 0 (v1 runs were
  stable that way) or with a weight ramp (linear 0→w over a few k steps) — ramp not
  implemented yet.
- The r3 candidate pair (cons vs nocons, consistency from step 0) is unaffected and
  remains the definitive A/B.
- bleu1's non-recovery suggests the d-embedding pathway, once wrecked, doesn't heal
  at lr 3e-5·decayed — supports from-step-0 training for any shortcut model.

## Also

Job hit the 720-min walltime at ~16k/30k steps — 30k-step PCSS runs need
`timeout_min` raised (or resume-from-checkpoint).

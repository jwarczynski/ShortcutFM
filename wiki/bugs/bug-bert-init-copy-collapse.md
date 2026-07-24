---
tags: [bug, masked-diffusion, bert-init, copy, evaluation]
status: open
date: 2026-07-24
severity: high
related: [[exp-r0-reeval-results]] [[decision-round2-design]] [[bug-eval-nfe1-random]]
---

# bert-init at lr 1e-4 collapses to source copying (BLEU 0.271 is fake)

## Symptom

`mdlm-r1-bert-init` (athena 2821327): val/bleu_nfe16 frozen at exactly 0.2712 across
all validations and across NFE 1/16 — and **val/copy_pct = 100%** everywhere.
BLEU ≈ 0.27 is precisely the score of copying the source on QQP val (sources and
targets are paraphrases; cf. continuous scut_768's 0.2757 at 99% copy).

## Root cause (two indistinguishable mechanisms, same conclusion)

1. True copy collapse: with pretrained-BERT init, "reproduce the visible source" is an
   immediately available low-CE solution; lr 1e-4 locks it in.
2. Format failure: the target region begins with its own `[CLS]`; if the model never
   emits it, eval extraction (`_process_prediction_text` fallback) returns the source
   text as the hypothesis. Copy% cannot distinguish these — either way the reported
   BLEU is not generation quality.

Counter-evidence that it's LR-dependent: `mdlm-r1-combo` (bert-init + **lr 3e-5** +
warmup) shows copy 4.8% and a climbing real BLEU — the collapse is avoidable.

## Consequences

- The "bert-init dominant, 0.271 at step 800" log entry of 2026-07-24 is **retracted**.
- Round-1 gate: bert-init = FAIL (pathological), combo = the honest bert-init signal.
- The copy% validation metric caught this immediately — continuous experiments went a
  full paper cycle before the same pathology was noticed. Keep copy% in every eval.

## Detection

val/copy_pct near 100, or BLEU bit-identical across NFE values / validation epochs.

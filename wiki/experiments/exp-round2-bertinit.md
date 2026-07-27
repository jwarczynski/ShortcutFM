---
tags: [experiment, masked-diffusion, recipe-search, round2, bert-init]
status: complete
date: 2026-07-27
related: [[decision-round2-design]] [[bug-bert-init-copy-collapse]] [[exp-r0-reeval-results]]
---

# Round-2 B-series results — bert-init at lr 3e-5 (10k steps, Athena)

Jobs 2823164/65/66 (finished 2026-07-25). Gate metrics at step 10k, valid split.

| run | bleu16(conf) | bleu16(random) | bleu1 | copy16 |
|---|---|---|---|---|
| bertinit-lr3e-5 | 0.191 | 0.263 | 0.080 | 6.6 |
| bertinit-**tclip** | **0.254** | 0.250 | 0.062 | 12.3 |
| bertinit-tclip-**nocons** | 0.236 | **0.269** | **0.136** | 6.8 |

Round-1 references at 10k: t-clip from scratch 0.163 peak; v1 control 0.124; combo 0.123.

## Findings

1. **bert-init(lr 3e-5) + t-clip reaches ~0.25–0.27 BLEU at 10k steps** — matches the
   from-scratch 50k ceiling (R0: 0.266) at 1/5 the steps, with sane copy% (<13). The
   collapse from [[bug-bert-init-copy-collapse]] is fully avoided at lr 3e-5.
2. t-clip stacks with bert-init: +0.06 bleu16(conf) over bert-init alone.
3. **Consistency A/B at the new base**: nocons wins bleu16(random) 0.269 vs 0.250 and
   bleu1 0.136 vs 0.062. The bleu1 gap is surprising (shortcut training should help
   1-step most) — BUT at 10k the cons model spends 25% of its batches on consistency,
   and its d-conditioning needs more training to pay off; R0 showed the same early
   pattern reversing by 50k. The 30k candidate pair (below) settles it.
4. Confidence-vs-random unmasking remains model-dependent; random still generally
   stronger. Both logged everywhere.

## Follow-ups launched (2026-07-27)

- **Recipe-candidate pair on Athena (30k steps)**: `mdlm-r3-cand-cons` (2828089) vs
  `mdlm-r3-cand-nocons` (2828088) — bert-init + lr 3e-5 + t-clip[0.15,0.95], the only
  delta is consistency on/off. This is the definitive shortcut A/B.
- PCSS C-series resubmitted after venv fix (uv-managed python on project storage,
  see [[cluster-pcss]]): tclip-30k (7810123), combo-30k (7810124).

---
tags: [experiment, masked-diffusion, recipe-search, round3, consistency, shortcut-ab]
status: complete
date: 2026-07-29
related: [[exp-round2-bertinit]] [[decision-round2-design]] [[bug-consistency-onset-collapse]] [[masked-diffusion-shortcuts]] [[exp-r0-reeval-results]]
---

# R3p — definitive consistency (shortcut) A/B on the winning recipe

The decisive experiment of the whole recipe search: does shortcut self-consistency
help masked discrete diffusion? Same recipe, the **only** delta is consistency on/off,
trained from step 0 (so the [[bug-consistency-onset-collapse]] onset trap does not apply).

- Base recipe: **bert-init + lr 3e-5 + t-clip[0.15, 0.95]** (the round-2 winner, [[exp-round2-bertinit]]).
- Cluster: PCSS (H100), 15k steps, 2 GPUs + accumulate_grad_batches 8, timeout_min 1440.
- Runs: `mdlm-r3p-cand-cons` (job 7822695) vs `mdlm-r3p-cand-nocons` (job 7822696).
- The Athena duplicate pair (2828088/89) queued >24h behind GenieSAE priority and was
  cancelled once PCSS finished ("ATHENA_DUPS_CANCELLED").

## Final results (valid split, 15k steps)

| variant | BLEU@NFE16 (conf) | BLEU@NFE16 (random) | BLEU@NFE1 | copy% | best b16r |
|---|---|---|---|---|---|
| **cons** (shortcut on) | 0.2648 | 0.2734 | 0.1067 | 9.9 | 0.2734 @14739 |
| **nocons** (shortcut off) | 0.2541 | **0.2735** | **0.1552** | 7.5 | 0.2812 @11603 |

nocons best BLEU@NFE1 = 0.1565 @14111.

## Conclusion — shortcuts do NOT help short-target paraphrasing

1. **Identical multi-step ceilings.** cons and nocons both plateau at BLEU@NFE16(random)
   ≈ 0.273 (nocons peak 0.281). Consistency training buys no higher ceiling.
2. **nocons is actually BETTER at NFE=1** (0.155 vs 0.107) — the exact operating point
   shortcuts were meant to help. On QQP the consistency batches (25% of training) are
   pure opportunity cost: they neither raise the ceiling nor improve one-step decoding.
3. This confirms and sharpens the R0 read ([[exp-r0-reeval-results]] finding 3): at
   convergence consistency bought at-most NFE=1 parity; here, with a stronger base and
   consistency from step 0, it does not even do that.

## Why (hypothesis) + what it implies

QQP targets are ~14 tokens; even NFE=1 is a reasonable decode, so there is little
multi-step cost for shortcuts to amortize. The shortcut hypothesis needs **long
targets** (many denoising steps to skip) to have anything to gain. → the real test is
**summarization (long targets) and MT**, which is the next phase ([[roadmap]]).

**Recipe locked for downstream tasks:** bert-init + lr 3e-5 + t-clip[0.15, 0.95] + EMA
0.99 + myle warmup 2000, consistency from step 0 when testing the shortcut (never abrupt
onset — [[bug-consistency-onset-collapse]]).

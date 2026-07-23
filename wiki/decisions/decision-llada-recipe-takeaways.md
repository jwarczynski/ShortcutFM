---
tags: [decision, masked-diffusion, recipe, llada]
status: open
date: 2026-07-23
related: [[masked-diffusion-shortcuts]] [[exp-masked-v1-qqp]] [[roadmap]] [[bug-eval-nfe1-random]]
---

# Decision: LLaDA / LLaDA2.0 recipe takeaways → Round-1 recipe search

Recipe facts extracted 2026-07-23 from **LLaDA** (arXiv 2502.09992) and **LLaDA2.0** (arXiv 2512.15745), annotated with our status. These drive the Round-1 run matrix in [[roadmap]].

## LLaDA (2502.09992)

| Recipe fact | Our status |
|---|---|
| t ~ U(0,1], per-token masking prob t, CE on masked with 1/t weight | ✅ **match** (t/T masking, T/t weight) |
| AdamW, weight decay 0.1 | ✅ **match** |
| LR warmup 2000 steps | ❌ **we have NO warmup** — testing R2 |
| SFT: prompt kept clean, only response masked | ✅ **match** (source never masked) |
| EOS-padding trained as normal tokens for length control | ✅ **match** (our first-2-PAD-targets equivalent) |
| Inference: low-confidence remasking best overall; semi-AR block decode better only when heavy EOS padding causes early termination | ❌ **we used random unmasking at NFE=1** — fixing ([[bug-eval-nfe1-random]]) |
| Generation length insensitive | (noted; not a concern at seq_len 128) |
| CFG helps but unused in main results | not used; low priority |

## LLaDA2.0 (2512.15745)

| Recipe fact | Our status |
|---|---|
| AR→diffusion conversion — never trained from scratch | ❌ **we trained from scratch** — testing R1 bert-init |
| SFT mask-ratio bandwidth clipping [α_min, α_max]: extreme ratios = high gradient variance, minimal signal | testing R3 t-clip 0.15–0.95 |
| Complementary masking (mask + inverse mask in same batch) for SFT data efficiency | round-2 candidate |
| Inference: confidence threshold 0.95, block size 32 best | round-2 candidate |
| Block-size ablation: 16 ≈ 32 > 64 | (reference for round-2 decoding round) |

## Decision

**Round 1 tests: bert-init, warmup-2k, t-clip, consistency-delay, no-EMA, and a combo** — 6 runs × 10k steps each (full matrix, controls, and round-2 branch logic in [[roadmap]]).

**Gate metric**: BLEU@NFE16-confidence (primary) + BLEU@NFE1 (shortcut payoff) + copy% (degenerate-copy rate).

## What would change our minds

- If [[exp-r0-reeval]] shows the 50k main model was already good at NFE16-confidence, decoding was the dominant problem and recipe deltas get judged on smaller margins.
- If bert-init dominates everything, the from-scratch recipe questions (warmup, t-clip) become second-order and Round 2 sweeps the bert-init base instead.

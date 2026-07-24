---
tags: [decision, masked-diffusion, recipe-search, round2]
status: in-progress
date: 2026-07-24
related: [[exp-r0-reeval-results]] [[bug-bert-init-copy-collapse]] [[decision-llada-recipe-takeaways]]
---

# Round-2 design

## Round-1 gate summary (10k steps, BLEU@NFE16 best-strategy / copy%)

| run | bleu16 | copy% | verdict |
|---|---|---|---|
| t-clip | **0.142** (peak 0.163) | 6.3 | WINNER (from scratch) |
| combo (bert-init+warmup+lr3e-5+delay) | 0.123, still climbing | 4.8 | honest bert-init signal |
| cons-delay-5k | 0.055 | 0.6 | weak |
| no-ema | 0.009 (unstable) | 0.2 | EMA helps — keep |
| warmup-2k (myle) | 0.003 | 0.2 | invalid test (inverse-sqrt ≠ linear decay) |
| bert-init (lr 1e-4) | 0.271 **copy 100%** | 100 | FAIL — [[bug-bert-init-copy-collapse]] |
| control (v1 main @10k, R0 eval) | 0.124 | 1.5 | baseline |
| control (v1 ablation @10k, R0 eval) | 0.170 | 3.2 | baseline (no consistency) |

Context from R0: v1 ceilings at ~50k steps are 0.266/0.268; source-copy BLEU on QQP ≈ 0.27.

## Round-2 runs

Hypotheses: (a) bert-init pays off at the right LR; (b) t-clip stacks with bert-init;
(c) t-clip from scratch keeps climbing past 10k; (d) consistency on/off retest at the
new operating point (round-1/R0 suggest consistency mostly buys NFE=1 parity).

| # | run | config | steps | cluster |
|---|---|---|---|---|
| B1 | r2-bertinit-lr3e-5 | bert-init + linear lr 3e-5 (isolates LR; no warmup/delay unlike combo) | 10k | athena |
| B2 | r2-bertinit-tclip | B1 + t_min/max_frac 0.15/0.95 | 10k | athena |
| B3 | r2-bertinit-tclip-nocons | B2 + consistency off (clean shortcut A/B at new base) | 10k | athena |
| C1 | r2-tclip-30k | round-1 t-clip continued regime, 30k steps | 30k | pcss |
| C2 | r2-combo-30k | round-1 combo recipe, 30k steps (was still climbing) | 30k | pcss |
| C3 | r2-tclip-band-01-09 | t-clip with band 0.10–0.90 | 10k | pcss |

Gate: BLEU@NFE16 (both strategies) + copy% + BLEU@NFE1, vs round-1 winners.
Round-3 branch: best of {B2 vs B3} answers whether consistency stays in the recipe;
best overall goes to 50k steps + NFE sweep as the recipe candidate.

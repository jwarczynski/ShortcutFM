---
tags: [meta, roadmap]
last_updated: 2026-07-23
---

# Roadmap

Current focus: **Round-1 recipe search** for the masked-diffusion + shortcut setup ([[masked-diffusion-shortcuts]]), motivated by the mixed v1 outcome ([[exp-masked-v1-qqp]]) and the LLaDA recipe gap analysis ([[decision-llada-recipe-takeaways]]).

## Round 0 (prerequisite, in flight)

- Fix the three eval bugs: [[bug-eval-nfe1-random]], [[bug-ablation-shortcut-conditioning]], [[bug-val-table-source-repredicted]].
- Re-eval existing checkpoints: [[exp-r0-reeval]] — NFE {1,4,16,64} × {confidence,random} on main-50k; both models @10k/50k (ablation at d=0).

## Round 1 — recipe search (6 runs)

All runs: **10k steps**, deltas vs `configs/training/qqp_masked_diffusion.yaml`, run names `mdlm-r1-*`.

| ID | Name | Delta vs base config | Tests |
|---|---|---|---|
| R1 | bert-init | `use_pretrained_weights: true`, `use_pretrained_embeddings: true` | LLaDA2.0's "never from scratch" |
| R2 | warmup-2k | myle scheduler, `warmup_steps: 2000` | LLaDA's warmup (we had none) |
| R3 | t-clip | `t_min_frac: 0.15`, `t_max_frac: 0.95` (**new code needed**) | LLaDA2.0 mask-ratio bandwidth clipping |
| R4 | cons-delay-5k | `consistency_start_step: 5000` | let CE stabilize before consistency kicks in |
| R5 | no-ema | `ema: null` | whether EMA hides/hurts at 10k scale |
| R6 | combo | bert-init + warmup + `lr: 3e-5` + cons-delay | best-guess stack |

**Controls reused for free** (no new compute):
- main run step-10000 checkpoint (identical LR trajectory → clean 10k control)
- ablation step-10000 checkpoint (no-consistency control, evaluated at d=0)

**Gate metric** (per [[decision-llada-recipe-takeaways]]): **BLEU@NFE16-confidence** (primary) + **BLEU@NFE1** (shortcut payoff) + **copy%**.

## Round 2 — branch logic

Which Round 2 happens depends on Round 0 + Round 1 outcomes:

- **Eval was the problem** (main-50k jumps at NFE16-confidence in [[exp-r0-reeval]]) → **decoding round**: confidence threshold ~0.95, block decode (block size 16/32 per LLaDA2.0), NFE sweeps, shortcut-conditioned few-NFE comparison.
- **bert-init wins** → sweep on the bert-init base: lr / weight decay / freeze-embeddings / ModernBERT init.
- **t-clip wins** → band sweep ([α_min, α_max] grid) + try dropping the 1/t weight (clipping may remove the need for it).
- **Nothing wins** → step back: seq_len 64 (denser signal per batch), anisotropy diagnostics on the embedding space, rethink the DiffuSeq packing format.

Round-2 candidates regardless of branch: complementary masking (LLaDA2.0 SFT data-efficiency trick).

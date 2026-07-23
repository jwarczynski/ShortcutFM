---
tags: [experiment, masked-diffusion, evaluation, qqp]
status: in-progress
date: 2026-07-23
related: [[exp-masked-v1-qqp]] [[bug-eval-nfe1-random]] [[bug-ablation-shortcut-conditioning]] [[roadmap]]
---

# exp-r0-reeval — re-evaluate the v1 checkpoints with fixed decoding (Round 0)

Placeholder for the re-evaluation of the [[exp-masked-v1-qqp]] checkpoints once the eval bugs ([[bug-eval-nfe1-random]], [[bug-ablation-shortcut-conditioning]], [[bug-val-table-source-repredicted]]) are fixed. This is "Round 0" of the recipe search: it establishes whether the models were already good and only the decoding was broken.

## Planned eval grid

**On main-50k** (`checkpoints/qqp/masked_diffusion/run_r43zphwi`):
- NFE ∈ {1, 4, 16, 64} × unmask_strategy ∈ {confidence, random} — full 8-cell grid.

**Both models @ 10k and 50k checkpoints:**
- main and no-consistency ablation, with the ablation evaluated at **d=0** (its only trained shortcut embedding — the conditioning fix).
- The 10k checkpoints double as free controls for the Round-1 10k-step recipe runs (identical LR trajectory) — see [[roadmap]].

## Gate metric

Same gate as Round 1: **BLEU@NFE16-confidence** (primary) + **BLEU@NFE1** (shortcut payoff) + **copy%** (degenerate copy-the-source rate).

## What each outcome means

- If main-50k BLEU@NFE16-confidence jumps well above 0.222 → eval was (much of) the problem; Round 2 pivots to a decoding round (see [[roadmap]] branch logic).
- If numbers stay flat across NFE/strategy → the training recipe itself is the bottleneck; Round-1 recipe search carries the weight.
- Main-vs-ablation at matched, correctly-conditioned decoding finally gives a clean read on whether the consistency term helps at NFE=1.

## Results

_Pending — eval fixes in flight._

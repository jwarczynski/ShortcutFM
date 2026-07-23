---
tags: [infrastructure, wandb]
date: 2026-07-23
related: [[exp-masked-v1-qqp]] [[bug-eval-nfe1-random]]
---

# Wandb

- Org: **jedrasowicz**
- Project: **Thesis** — https://wandb.ai/jedrasowicz/Thesis

## Key runs

| Run ID | Name | SLURM | What | Page |
|---|---|---|---|---|
| `r43zphwi` | qqp_masked_diffusion | 995218 (hgx2) | 50k main masked-diffusion + consistency run | [[exp-masked-v1-qqp]] |
| `exu5bftd` | qqp_masked_no_consistency | 995219 (hgx2) | 50k no-consistency ablation (⚠ eval invalid, see [[bug-ablation-shortcut-conditioning]]) | [[exp-masked-v1-qqp]] |

## Metric conventions

- **`val/bleu_nfe16`** — primary gate metric **after the eval fix** ([[bug-eval-nfe1-random]]); the legacy `val/bleu` from the v1 runs is NFE=1-random and not comparable.
- `train/masked_ce_loss` — the masked-CE (base diffusion) term.
- `train/consistency_loss` — the shortcut self-consistency term (only nonzero when `consistency_loss_weight > 0`).
- Planned per-NFE family: `val/bleu_nfe{1,4,16,64}` with confidence unmasking; report copy% alongside.

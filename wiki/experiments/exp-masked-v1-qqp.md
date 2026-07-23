---
tags: [experiment, masked-diffusion, qqp, consistency, ablation]
status: complete
date: 2026-07-21
related: [[masked-diffusion-shortcuts]] [[bug-eval-nfe1-random]] [[bug-ablation-shortcut-conditioning]] [[bug-val-table-source-repredicted]] [[exp-r0-reeval]]
---

# exp-masked-v1-qqp — first masked-diffusion runs on QQP (main + no-consistency ablation)

First test of [[masked-diffusion-shortcuts]]: two 50k-step runs on QQP, identical except for the consistency term. Launched 2026-07-21 on hgx2 (4 GPUs each), both completed by 2026-07-23.

## Runs

### Main: `qqp_masked_diffusion`

- Config: `configs/training/qqp_masked_diffusion.yaml` (composite criterion, 25% consistency batch share)
- Wandb: `jedrasowicz/Thesis/r43zphwi`
- SLURM: 995218 (hgx2, 4 GPUs)
- Checkpoints: `checkpoints/qqp/masked_diffusion/run_r43zphwi`
- Final: train CE **0.85**, val/bleu **0.222**, consistency loss **0.22 → 0.035** (clean decay)

### Ablation: `qqp_masked_no_consistency`

- Same config with overrides `consistency_loss_weight=0 self_consistency_ratio=0`
- Wandb: `jedrasowicz/Thesis/exu5bftd`
- SLURM: 995219
- Checkpoints: `checkpoints/qqp/masked_no_consistency/run_exu5bftd`
- Final: train CE **0.60**, val/bleu **0.209**

## Outcome — MIXED, numbers contaminated

Training itself looks healthy (losses converge, consistency loss decays 6×). But:

1. **Outputs judged poor by the user** on manual inspection.
2. **All val numbers were measured at NFE=1 with random unmasking** — see [[bug-eval-nfe1-random]]. `denoising_step_size=2048=diffusion_steps` collapses denoising to a single forward pass committing every masked token at once, in random order. MDLM-class quality requires iterative low-confidence decoding, so 0.222 / 0.209 are floor numbers, not the models' real quality.
3. **The ablation's numbers are invalid outright** — see [[bug-ablation-shortcut-conditioning]]. It was trained with `default_shortcut="0"` and no consistency loss (shortcut embedding never trained for d≠0), yet eval called `denoise(shortcut_size=2048)`, feeding an untrained embedding. Its val/bleu 0.209 is garbage-conditioned.
4. Part of "outputs look bad" may be a display/extraction artifact — see [[bug-val-table-source-repredicted]] (source positions re-predicted instead of ground-truth in the denoise return_logits path).

Consequence: **no conclusion about the consistency term can be drawn yet.** The main-vs-ablation comparison (0.222 vs 0.209) is confounded by all three bugs. Re-eval pending in [[exp-r0-reeval]].

## What we learned anyway

- The masked composite criterion trains stably at 50k steps; consistency loss does not destabilize the CE term (CE 0.85 vs 0.60 gap is expected — 25% of batch capacity goes to consistency).
- Eval-config defaults copied from the continuous setup are actively dangerous for the masked setup (step size semantics differ) — the whole eval path needs its own defaults.

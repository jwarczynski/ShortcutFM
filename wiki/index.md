---
tags: [meta, index]
last_updated: 2026-07-23
---

# Index

This is the catalog of all wiki pages. Start here when exploring. See [[schema]] for conventions and [[log]] for a chronological record.

## Concepts

- [[masked-diffusion-shortcuts]] — The research idea: does shortcut self-consistency (one 2d-step ≈ two composed d-steps) transfer from continuous flow matching to masked discrete diffusion? Implementation in `shortcutfm/masked_criteria.py`

## Experiments

- [[exp-masked-v1-qqp]] — First 2×50k QQP runs (main r43zphwi + no-consistency ablation exu5bftd); complete, outcome mixed — all val numbers contaminated by eval bugs
- [[exp-r0-reeval]] — Planned re-eval of v1 checkpoints: NFE {1,4,16,64} × {confidence,random}, ablation at d=0; in progress

## Bugs

- [[bug-eval-nfe1-random]] — val/bleu measured at NFE=1 with random unmasking (denoising_step_size=2048=diffusion_steps); fix = multi-NFE validation + confidence unmasking default
- [[bug-ablation-shortcut-conditioning]] — no-consistency ablation evaluated at d=2048 with an untrained shortcut embedding; its 0.209 is invalid; fix = eval at d=0
- [[bug-val-table-source-repredicted]] — denoise return_logits re-predicts source/padding positions instead of ground truth → garbled val table + inflated full_denoising_ce

## Decisions

- [[decision-llada-recipe-takeaways]] — Recipe facts from LLaDA + LLaDA2.0 with our-status annotations; drives the Round-1 recipe search

## Infrastructure

- [[cluster-hgx]] — Primary cluster: ssh alias `slurm`, nodes hgx1/hgx2 (8 GPUs), exca submission, log paths, uv login-shell gotcha, dataset locations
- [[cluster-athena]] — Secondary A100 cluster: ssh alias `athena` (never `ares`), plgrid-gpu-a100 partition, no internet on compute nodes, only parasci+paws_wiki datasets
- [[wandb]] — Org jedrasowicz / project Thesis; key run table; metric conventions (val/bleu_nfe16 primary gate)

## Meta

- [[schema]] — Conventions, page types, workflows
- [[log]] — Chronological event log
- [[roadmap]] — Round-1 recipe search (6 runs) + round-2 branch logic

## Quick links

- Repo: [jwarczynski/ShortcutFM](https://github.com/jwarczynski/ShortcutFM)
- Wandb: [jedrasowicz/Thesis](https://wandb.ai/jedrasowicz/Thesis)
- Paper: [ACL 2026 short](https://aclanthology.org/2026.acl-short.53/)

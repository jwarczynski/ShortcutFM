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
- [[exp-r0-reeval]] — Re-eval of v1 checkpoints: NFE {1,4,16,64} × {confidence,random}, ablation at d=0; complete → [[exp-r0-reeval-results]]
- [[exp-r0-reeval-results]] — Corrected numbers: masked ~0.27 at ~50k, above continuous baseline (0.155); confidence underperforms random at NFE≥16
- [[exp-round2-bertinit]] — Round-2 B-series: bert-init lr 3e-5 + t-clip hits the ceiling in ~1/5 the steps
- [[exp-r3p-consistency-ab]] — DEFINITIVE shortcut A/B: consistency gives no benefit on short-target QQP; recipe search closed
- [[exp-decode-study]] — Confidence-THRESHOLD decoding (τ≈0.9, NFE16) breaks 0.30 BLEU (0.318 best) — new project best, training-free
- [[exp-mt-iwslt]] — Task generalization: MT (opus-100 de→en) masked diffusion, winning recipe, cons/nocons A/B; running on Athena
- [[exp-summ-xsum]] — Task generalization: summarization (XSum, seq_len 256), winning recipe, cons/nocons A/B; setup on PCSS

## Bugs

- [[bug-eval-nfe1-random]] — val/bleu measured at NFE=1 with random unmasking (denoising_step_size=2048=diffusion_steps); fix = multi-NFE validation + confidence unmasking default
- [[bug-ablation-shortcut-conditioning]] — no-consistency ablation evaluated at d=2048 with an untrained shortcut embedding; its 0.209 is invalid; fix = eval at d=0
- [[bug-val-table-source-repredicted]] — denoise return_logits re-predicts source/padding positions instead of ground truth → garbled val table + inflated full_denoising_ce
- [[bug-bert-init-copy-collapse]] — bert-init at lr 1e-4 scores a fake 0.271 BLEU that is 100% source-copy; caught by copy% metric; use lr 3e-5
- [[bug-consistency-onset-collapse]] — STAR FINDING: enabling consistency abruptly mid-training (start_step=5000) causes catastrophic collapse; bleu1 never recovers. From step 0 or ramped only
- [[bug-mbert-vocab-oom]] — mbert's 119547 vocab OOMs 40GB A100 (V×V anisotropy matmul + batch-128 logits); fixed with O(V·d) anisotropy + batch 64

## Decisions

- [[decision-llada-recipe-takeaways]] — Recipe facts from LLaDA + LLaDA2.0 with our-status annotations; drives the Round-1 recipe search
- [[decision-round2-design]] — Round-2 design + OUTCOME: recipe locked (bert-init + lr 3e-5 + t-clip[0.15,0.95]); shortcuts don't help short targets

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

- [report-outline](report-outline.md) — narrative outline for the final HTML report (key findings, in order)

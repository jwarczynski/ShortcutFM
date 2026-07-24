---
tags: [meta, log]
---

# Log

Append-only chronological record. Newest entries at the top. Each entry uses `## [YYYY-MM-DD] <type> | <short title>` for easy grepping:

```bash
grep "^## \[" wiki/log.md | head -20
grep "experiment\|bug" wiki/log.md
```

Types: `ingest`, `experiment`, `bug`, `decision`, `meta`, `note`.

---

## [2026-07-24] infrastructure | PCSS/Eagle cluster onboarded

- SSH alias `pcss` working (key auth; fixed concatenated authorized_keys line). Repo + uv env + QQP dataset bootstrapped at `/mnt/storage_6/project_data/pl1095-01/ShortcutFM`.
- 76 nodes x 4x H100 on `tesla` partition, 94k GPU-h remaining on grant — biggest pool of the three clusters. See [[cluster-pcss]].


## [2026-07-23] experiment | Round-1 recipe search launched (6 runs) + R0 re-eval queued

- R0 re-eval: hgx job 995688 (NFE {1,4,16,64} x {confidence,random} grid on v1 checkpoints; main run's last ckpt is step **49000** — hgx2 died mid-training).
- Round-1 (10k steps each): hgx 995689 `mdlm-r1-bert-init`; athena 2818487 `mdlm-r1-t-clip` (RUNNING), 2818518 `mdlm-r1-warmup-2k`, 2818526 `mdlm-r1-no-ema`, 2818527 `mdlm-r1-cons-delay-5k`, 2818528 `mdlm-r1-combo`.
- Placement policy set: default Athena (hgx unstable/occupied — see [[cluster-hgx]]).
- Gotchas hit: exca `cfg.train()` blocks until job completion → submissions must be nohup-backgrounded; `bash -lc` resets cwd to $HOME on athena → `cd` must be inside the same subshell as the nohup.


## [2026-07-23] experiment | Both masked-diffusion 50k runs done; quality investigation → 3 bugs found, recipe search planned

Both 50k-step runs from [[exp-masked-v1-qqp]] completed: main `qqp_masked_diffusion` (r43zphwi) final train CE 0.85, val/bleu 0.222, consistency loss 0.22→0.035; ablation `qqp_masked_no_consistency` (exu5bftd) train CE 0.60, val/bleu 0.209. Generated outputs judged poor by user → deep-dive investigation found three bugs that contaminate the numbers:

- [[bug-eval-nfe1-random]] — all val/bleu measured at NFE=1 with random unmasking (denoising_step_size=2048=diffusion_steps → single forward pass); MDLM-class models need iterative low-confidence decoding. Fixing.
- [[bug-ablation-shortcut-conditioning]] — ablation trained with default_shortcut="0" but eval called denoise(shortcut_size=2048) → untrained shortcut embedding; its 0.209 is garbage-conditioned. Fixing.
- [[bug-val-table-source-repredicted]] — denoise return_logits path refreshes source/padding positions with re-predicted logits instead of ground truth → val/predictions table + full_denoising_ce artifacts. Fixing.

Extracted training-recipe facts from LLaDA + LLaDA2.0 papers → [[decision-llada-recipe-takeaways]]. Planned Round-1 recipe search (6× 10k-step runs: bert-init, warmup, t-clip, cons-delay, no-ema, combo) → [[roadmap]]. Re-eval of existing checkpoints at multi-NFE with confidence unmasking pending → [[exp-r0-reeval]].

## [2026-07-22] note | Training progress check

Both runs healthy mid-flight on hgx2: main (995218) consistency loss decaying cleanly, masked CE trending down; ablation (995219) CE lower than main as expected (no consistency term competing). No intervention needed.

## [2026-07-21] experiment | Masked discrete diffusion implemented; 2×50k QQP runs launched

Implemented the masked-diffusion port of shortcut self-consistency (`shortcutfm/masked_criteria.py`: MaskedDiffusionCriterion, MaskedConsistencyCriterion, MaskedCompositeCriterion) — see [[masked-diffusion-shortcuts]] for the research idea. Launched two 50k-step runs on hgx2 (4 GPUs each):

- main `qqp_masked_diffusion` (config `configs/training/qqp_masked_diffusion.yaml`, SLURM 995218, wandb r43zphwi) — composite loss, 25% consistency batch share.
- ablation `qqp_masked_no_consistency` (SLURM 995219, wandb exu5bftd, overrides `consistency_loss_weight=0 self_consistency_ratio=0`).

Tracked in [[exp-masked-v1-qqp]].

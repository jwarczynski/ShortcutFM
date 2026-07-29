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

## [2026-07-29] experiment | Recipe search CLOSED — shortcuts don't help short-target paraphrasing

- Definitive consistency A/B finished (PCSS 15k, `mdlm-r3p-cand-cons` 7822695 vs `nocons` 7822696): cons and nocons reach **identical multi-step ceilings** (b16r ~0.273; nocons peak 0.281), and **nocons is better at NFE=1** (0.155 vs 0.107). Consistency training is pure opportunity cost on QQP's ~14-token targets. See [[exp-r3p-consistency-ab]].
- **Recipe locked**: bert-init + lr 3e-5 + t-clip[0.15,0.95] + EMA 0.99 + myle warmup 2000. Round-2 design closed ([[decision-round2-design]]).
- Answers the user's core question ("are shortcuts making it better?"): NO for short targets. The shortcut hypothesis needs long targets (many denoising steps to skip) → next phase = MT (iwslt14) + summarization (XSum), plus a training-free decode study (confidence-threshold / block decoding). Athena duplicate pair 2828088/89 cancelled.


## [2026-07-28] experiment | tclip-30k timed out at 13.7k (numbers excellent); candidate pair hedged onto PCSS

- `mdlm-r2-tclip-30k` (pcss): TIMEOUT at 12h/13.7k steps (720-min default again — my miss). But the data is decisive: **bleu16-random 0.277 and bleu1 0.229 at ~12.5k steps, still climbing** — from-scratch t-clip beats the v1 50k ceiling (0.266) in a quarter of the steps. t-clip is confirmed as the strongest single from-scratch factor.
- Athena candidate pair (2828088/89) queued >24h (GenieSAE jobs share the account priority). Hedged: same A/B submitted on PCSS as `mdlm-r3p-cand-{cons,nocons}` (7822668/69), 15k steps, timeout_min=1440, 2 GPUs, gpu16 excluded. First to run wins; the other gets cancelled.


## [2026-07-28] bug | combo-30k collapse at consistency onset; PCSS maintenance

- `mdlm-r2-combo-30k`: healthy 0.27 bleu16 until step 5000, then **collapse at consistency_start_step=5000** (bleu16 0.27→0.02; bleu1 0.26→0.01, never recovered). Abrupt objective switch + untrained d-pathway wrecks the trunk. Rule: consistency from step 0 or with a weight ramp — never abruptly mid-training. See [[bug-consistency-onset-collapse]]. Retroactively explains round-1 cons-delay-5k weakness.
- Also: combo-30k hit the 720-min walltime at 16k/30k; PCSS went into maintenance ("System is going down") — tclip-30k rerun (7815289) fate unknown until it's back. Athena candidate pair still queued.
- Peak numbers so far: combo pre-collapse **bleu1 0.26-0.27 at NFE=1 at ~5k steps** (bert-init+warmup) — best 1-step result of the whole search.


## [2026-07-27] experiment | Round-2 B-series gated; recipe-candidate pair launched (30k)

- B-series (bert-init lr 3e-5, 10k): **bertinit-tclip 0.254 bleu16** / nocons variant 0.269 (random) — matches from-scratch 50k ceiling at 1/5 steps, no copy collapse. See [[exp-round2-bertinit]].
- Candidate recipe = bert-init + lr 3e-5 + t-clip[0.15,0.95]. Definitive consistency A/B at 30k launched on athena: `mdlm-r3-cand-cons` 2828089 vs `mdlm-r3-cand-nocons` 2828088.
- PCSS venv fixed (uv-managed python on project storage — login/compute node python mismatch documented in [[cluster-pcss]]); C-series resubmitted: 7810123 (tclip-30k), 7810124 (combo-30k).


## [2026-07-24] experiment | bert-init retracted (copy collapse); round 2 launched (6 runs)

- **RETRACTION**: `mdlm-r1-bert-init` BLEU 0.271 was **100% source copying** (val/copy_pct caught it; BLEU bit-identical across NFE). At lr 1e-4, pretrained BERT locks onto reproduce-the-source. Run cancelled. See [[bug-bert-init-copy-collapse]]. Honest bert-init signal = round-1 combo (lr 3e-5, copy 4.8%).
- Round 2 launched per [[decision-round2-design]]: athena B-series 2823164/65/66 (bertinit-lr3e-5, +tclip, +tclip-nocons); pcss C-series 7785151/52/53 (tclip-30k, combo-30k, tclip-band-01-09). First PCSS jobs ever (H100s).


## [2026-07-24] experiment | R0 re-eval complete + bert-init early signal is dominant

- R0 (athena 2821594): multi-step decoding adds +0.04-0.06 BLEU (best 0.266-0.268 at ~50k steps, both v1 runs). No copy pathology (<=8%). SURPRISE: confidence unmasking UNDERPERFORMS random at NFE>=16 for consistency-trained models — val now logs both strategies. See [[exp-r0-reeval-results]].
- `mdlm-r1-bert-init` (athena 2821327): **BLEU@NFE16 = 0.271 at step 800** — matches the 50k-step from-scratch ceiling with 60x fewer steps. H2 (pretrained init) is the dominant round-1 factor, ahead of t-clip (0.142@10k).


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

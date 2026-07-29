---
tags: [meta, roadmap]
last_updated: 2026-07-29
---

# Roadmap

## Status (2026-07-29): QQP recipe search CLOSED → next phase active

The masked-diffusion recipe search on QQP is **complete**. Recipe locked:
**bert-init + lr 3e-5 + t-clip[0.15, 0.95]** (+ EMA 0.99, myle warmup 2000). The
definitive shortcut A/B ([[exp-r3p-consistency-ab]]) showed consistency gives **no**
benefit on short-target paraphrasing (identical multi-step ceiling; worse at NFE=1).

**Current phase (in priority order):**
1. **Decode study** (training-free): confidence-threshold + block decoding on existing
   QQP checkpoints — does either beat plain confidence/random at matched NFE?
2. **MT — iwslt14 de-en**: add `[MASK]` to opus-mt vocab + `iwslt` tokenize branch;
   winning recipe; consistency on/off A/B. First long(er)-target test of shortcuts.
3. **Summarization — XSum**: net-new tokenize branch, seq_len 256; winning recipe;
   consistency on/off A/B. Strongest shortcut test (longest targets).
4. **Living HTML report** (visual-comms, highlight-to-comment) built while MT/summ train.
5. Bonus if all else finishes: complementary masking (LLaDA2.0 SFT trick).

The shortcut hypothesis needs long targets (many denoising steps to skip) to pay off —
QQP's ~14-token targets were too short. MT/summ are the real test.

---

## Archived focus (complete): Round-1/2 recipe search

Below is the original recipe-search plan, kept for provenance. Motivated by the mixed v1
outcome ([[exp-masked-v1-qqp]]) and the LLaDA recipe gap analysis ([[decision-llada-recipe-takeaways]]).

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


## Later: task generalization (user, 2026-07-24)

Once the best masked-diffusion recipe is found on QQP paraphrasing, extend to other
seq2seq tasks — **summarization** and **machine translation** first. Notes:

- MT: tokenization pipeline already supports WMT19 / iwslt14 de-en
  (`Helsinki-NLP/opus-mt-en-de` tokenizer path in configs; hgx has raw data).
- Summarization: needs a tokenization pass (XSum / CNN-DM); longer targets will
  stress the seq_len-128 format — likely needs seq_len 256+ (parasci configs already
  use `max_position_embeddings: 256` as precedent).
- Context from R0: the continuous paper's QQP numbers were inflated by src-copying
  (99% at NFE=1); masked diffusion has no copy pathology and BLEU *rises* with NFE —
  the honest cross-paradigm comparison on new tasks should use the corrected
  multi-NFE eval from the start.

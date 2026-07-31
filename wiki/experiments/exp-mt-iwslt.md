---
tags: [experiment, masked-diffusion, machine-translation, task-generalization, shortcut-ab]
status: running
date: 2026-07-29
related: [[exp-r3p-consistency-ab]] [[decision-round2-design]] [[roadmap]] [[cluster-athena]]
---

# MT — iwslt (opus-100 de→en) masked diffusion, winning recipe

First task-generalization run: does the QQP-winning recipe transfer to machine
translation, and — the real question — do shortcuts help now that targets are longer
than QQP's ~14 tokens? ([[exp-r3p-consistency-ab]] showed no shortcut benefit on short
targets.)

## Setup

- Dataset: **Helsinki-NLP/opus-100 de-en**, train subsampled to 200k pairs (iwslt2017 is
  no longer loadable — datasets>=4 dropped script datasets). Tokenized de→en, seq_len 128,
  DiffuSeq packing `[CLS] de [SEP] [SEP] [CLS] en [SEP] [PAD]…`. Output at
  `datasets/tokenized/bert-base-multilingual-cased/iwslt/`.
- Backbone/tokenizer: **bert-base-multilingual-cased** (native `[MASK]`=103, covers de+en;
  bert-init transfers directly — the opus-mt German tokenizer has no `[MASK]` and its
  58k vocab can't take bert-base init). vocab_size 119547.
- Recipe: bert-init + lr 3e-5 + t-clip[0.15, 0.95] + myle warmup 2000 + EMA 0.99,
  30k steps. Config `configs/training/iwslt_masked_combo_athena.yaml`.
- Cluster: Athena (plgrid-gpu-a100), 4 GPUs, timeout 2880 min.

## Runs (shortcut A/B — the only delta is consistency on/off)

| arm | job | wandb run | delta |
|---|---|---|---|
| cons | 2840543 | mdlm-mt-iwslt-cons | consistency ON (self_consistency_ratio 0.25) |
| nocons | 2840545 | mdlm-mt-iwslt-nocons | self_consistency_ratio=0, consistency_loss_weight=0 |

Both PENDING (priority queue) as of 2026-07-29 15:40. Gate: BLEU@NFE{1,16} (conf+random)
+ copy% on valid, cons vs nocons — does the shortcut buy anything at longer targets?

## OOM saga (resolved 2026-07-31)

Three failure rounds before both arms ran clean — all from **mbert's 119547 vocab**
(~4× QQP's 30522), which inflates every vocab-sized tensor on 40GB A100s:

1. **Anisotropy diagnostic OOM** (`_calculate_anisotropy`): `torch.mm(emb, emb.T)` on
   119547×119547 = 53 GiB. Fixed in code (commit c1a3d8a): `||sum_i ê_i||²` form, O(V·d),
   numerically identical. See [[bug-mbert-vocab-oom]].
2. **Full-batch CE logits OOM at batch 128**: 128×128×119547×4B ≈ 7.8 GiB logits tensor
   tips over. nocons (full-batch CE) died in 56 s; cons (75% CE) limped to the first
   validation then died.
3. **Validation NFE=16 OOM**: the multi-step denoising loop allocates more than training,
   so even surviving training, cons OOM'd at step 2500 val.

**Fix:** batch 64 + accumulate 8 (eff. batch 512 unchanged) for both arms, plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Config updated (f4220b1).
Final jobs: cons 2847706, nocons 2847700 (both batch 64).

## Notes

- opus-100 is web-crawled and noisy (some misaligned pairs); fine for a first signal,
  not a headline BLEU number. If the recipe transfers, a cleaner set (WMT news / cleaned
  iwslt) is the follow-up.
- Code: iwslt branch in `scripts/run_tokenize.py`; `[MASK]` support + `mask_token_id`
  in `shortcutfm/tokenizer.py`; mbert allow-listed in `shortcutfm/config.py` +
  `shortcutfm/model/factory.py`. Also fixed a pre-existing syntax error in run_tokenize.py.

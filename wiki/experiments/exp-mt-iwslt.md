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

## Notes

- opus-100 is web-crawled and noisy (some misaligned pairs); fine for a first signal,
  not a headline BLEU number. If the recipe transfers, a cleaner set (WMT news / cleaned
  iwslt) is the follow-up.
- Code: iwslt branch in `scripts/run_tokenize.py`; `[MASK]` support + `mask_token_id`
  in `shortcutfm/tokenizer.py`; mbert allow-listed in `shortcutfm/config.py` +
  `shortcutfm/model/factory.py`. Also fixed a pre-existing syntax error in run_tokenize.py.

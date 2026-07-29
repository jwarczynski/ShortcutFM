---
tags: [experiment, masked-diffusion, summarization, task-generalization, shortcut-ab]
status: setup
date: 2026-07-29
related: [[exp-r3p-consistency-ab]] [[exp-mt-iwslt]] [[roadmap]] [[cluster-pcss]]
---

# Summarization — XSum masked diffusion, winning recipe

The strongest test of the shortcut hypothesis in the task-generalization phase:
longest targets → most denoising steps to potentially skip. ([[exp-r3p-consistency-ab]]:
no shortcut benefit at QQP's ~14-token targets.)

## Setup

- Dataset: **EdinburghNLP/xsum** (document → single-sentence summary). document→src,
  summary→trg. **seq_len 256** (`--max_seq_length 256`) to fit the long source.
- Backbone/tokenizer: **bert-base-uncased** (English; native `[MASK]`=103). Keep bert-base's
  native 512 position embeddings (≥256) — shrinking to 256 breaks bert-init (pretrained
  pos-emb table is 512-wide; verified the size-mismatch error locally).
- Recipe: bert-init + lr 3e-5 + t-clip[0.15, 0.95] + myle warmup 2000 + EMA 0.99, 30k steps.
  Config `configs/training/xsum_masked_combo_pcss.yaml`, 2 GPUs + accumulate 8 (PCSS tesla).
- A/B: consistency on (`mdlm-summ-xsum-cons`) vs off (override, `mdlm-summ-xsum-nocons`).

## Status

- Tokenization on PCSS login node (in progress 2026-07-29). **Gotcha hit:** first attempt
  failed with `OSError: Disk quota exceeded` — the HF dataset download filled PCSS's 1 GB
  home. Fix: `HF_HOME=/mnt/storage_5/scratch/pl1095-01/hf_home` (scratch, 20 TB) and clear
  `~/.cache/huggingface`. See [[cluster-pcss]].
- Training A/B not yet submitted — waiting on tokenized data.

## Note on target length

XSum summaries are single-sentence (~25 tokens) — longer than QQP but the *source* is what's
long. If the shortcut still shows nothing here, **CNN-DM** (multi-sentence, ~55-token
summaries) is the stronger follow-up target for the shortcut question.

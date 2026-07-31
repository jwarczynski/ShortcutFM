---
tags: [bug, oom, mbert, machine-translation, vocab, memory]
status: resolved
date: 2026-07-31
severity: high
related: [[exp-mt-iwslt]] [[cluster-athena]]
---

# Large vocab (mbert 119547) OOMs on 40GB A100 — anisotropy matmul + batch-128 logits

## Symptom

MT iwslt jobs (bert-base-multilingual-cased backbone) OOM on Athena's 40GB A100s where
the QQP recipe (bert-base-uncased, vocab 30522) ran fine at batch 128. Three distinct OOM
sites, all vocab-size-driven (119547 ≈ 3.9× 30522).

## Root causes & fixes

1. **Anisotropy diagnostic** — `_calculate_anisotropy` did `torch.mm(emb, emb.T)`, a
   V×V matrix = 119547² × 4B = **53 GiB**. Fired at `on_train_epoch_end`.
   **Fix (commit c1a3d8a):** for L2-normalized embeddings,
   `sum_{i,j} cos(e_i,e_j) = ||sum_i ê_i||²` — compute the summed vector then its dot
   product. O(V·d) instead of O(V²), numerically identical (verified).
2. **Full-batch CE logits at batch 128** — logits tensor 128×128×119547×4B ≈ **7.8 GiB**
   (matched the "tried to allocate 7.30 GiB" error). nocons (full-batch CE) OOM'd in 56 s;
   cons (25% of batch on consistency) survived training but OOM'd at first validation.
3. **Validation NFE=16 denoising** allocates more than a training step (iterative loop),
   so batch 128 OOM'd at val even when training survived.
   **Fix (commit f4220b1):** batch 64 + accumulate_grad_batches 8 (effective batch 512
   unchanged) for both arms, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Lesson for future large-vocab tasks

Any task with a bigger tokenizer than bert-base-uncased (mbert, XLM-R, byte-level, opus-mt
58k) must budget vocab-sized tensors: embedding-table diagnostics, CE logits, and the
denoising logits in validation all scale with V. Prefer batch ≤ 64 on 40GB cards, and never
materialize a V×V matrix. XSum uses bert-base-uncased (30522) so is unaffected by (2)/(3),
but the anisotropy fix (1) benefits every run.

---
tags: [infrastructure, cluster, slurm, athena]
date: 2026-07-23
related: [[cluster-hgx]]
---

# Cluster: Athena (secondary, A100)

## Access

- SSH alias: **`athena`** — **never `ares`** (different cluster, easy to confuse; ares is NOT this project's target).
- Project dir: `/net/tscratch/people/plgjentker/ShortcutFM` — **already cloned**; branch `fix/padding-noise`, at commit ``c647a52`` as of 2026-07-23.

## SLURM

- Partition: **`plgrid-gpu-a100`** (A100 40GB), max walltime **2880 min** (48h)
- Account: **`plgnarnlg-gpu-a100`**

## Gotchas

- **No internet on compute nodes.** Pre-download HuggingFace models/tokenizers on the **login node** before submitting (populate the HF cache); a job that tries to fetch from the hub will hang/fail.
- A100s are 40GB here — smaller than hgx GPUs; watch batch size.

## Datasets

Only **parasci** and **paws_wiki** are present under `datasets/tokenized/bert-base-uncased/`. **QQP-Official must be synced from hgx** ([[cluster-hgx]]) before any QQP run on Athena.

## Existing Athena configs

- `configs/training/parasci_scut_768_fix_athena.yaml`
- `configs/training/pawswiki_scut_768_fix_athena.yaml`

Their infra block (reuse for new Athena configs):

```yaml
gpus_per_node: 8
tasks_per_node: 8
timeout_min: 2880
```

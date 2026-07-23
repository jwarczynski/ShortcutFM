---
tags: [infrastructure, cluster, slurm, hgx]
date: 2026-07-21
related: [[cluster-athena]] [[exp-masked-v1-qqp]]
---

# Cluster: HGX (secondary — see reliability note)

> **Placement policy (2026-07-23, from user):** hgx nodes are unstable and heavily
> occupied — hgx2 went down mid-training on 2026-07-22/23 (killed the
> [[exp-masked-v1-qqp]] main run at step 49000/50000) and hgx1 is often fully
> allocated. **Default new GPU jobs to [[cluster-athena]]**: lower occupancy,
> easier to get multiple instances; A100s are less powerful per-GPU but fine for
> our ~110M models. Use hgx opportunistically when idle, and don't pin jobs to a
> single node unless required.

## Access

- SSH alias: **`slurm`** (`ssh slurm`)
- Project dir: `/home/inf148234/projects/ShortcutFM`

## Hardware

- Nodes **hgx1** / **hgx2**, 8 GPUs each
- Partition: **`hgx`**

## Submitting training

Submission goes through exca (`scripts/submit_training.py`) and is **non-blocking** — it registers the job and returns. Always run through a **login shell** over ssh, because `uv` is only on PATH after profile init:

```bash
ssh slurm "bash -lc 'cd /home/inf148234/projects/ShortcutFM && uv run python scripts/submit_training.py <cfg> [overrides]'"
```

Example (the [[exp-masked-v1-qqp]] runs):

```bash
bash -lc 'cd /home/inf148234/projects/ShortcutFM && uv run python scripts/submit_training.py configs/training/qqp_masked_diffusion.yaml'
bash -lc 'cd ... && uv run python scripts/submit_training.py configs/training/qqp_masked_diffusion.yaml consistency_loss_weight=0 self_consistency_ratio=0'
```

Check status with `squeue -u inf148234`.

## Logs

Exca writes job logs under:

```
jobs/shortcutfm.config.TrainingConfig.train,0/logs/inf148234/<jobid>/
```

(relative to the project dir; `<jobid>` = SLURM job ID, e.g. 995218).

## Gotchas

- **`uv` needs a login shell over ssh** — a bare `ssh slurm 'uv run ...'` fails with command-not-found; always wrap in `bash -lc '...'`.

## Datasets

Tokenized datasets live at `datasets/tokenized/bert-base-uncased/`:

- `QQP-Official`
- `parasci`
- `paws_wiki`
- (others alongside)

This is the canonical copy — Athena only has a subset ([[cluster-athena]]).

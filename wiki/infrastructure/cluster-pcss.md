---
tags: [infrastructure, cluster, slurm, pcss, eagle, h100]
date: 2026-07-24
related: [[cluster-athena]] [[cluster-hgx]]
---

# Cluster: PCSS / Eagle (largest GPU pool)

## Access

- SSH alias: **`pcss`** (also `eagle`) — `eagle.man.poznan.pl`, user `jentker`, key `~/.ssh/id_ed25519_pcss`
- Web fallback: https://ssh-ui.pcss.plcloud.pl
- Docs: https://help.pcss.plcloud.pl/portal/hpc/ · status: https://stats.hpc.pcss.pl · portal: https://pcss.plcloud.pl
- Grant **pl1095-01**: 2M CPU-h, **100k GPU-h** (~6% used as of 2026-07-24), 30 TB project + 20 TB scratch

## Hardware

- Partition **`tesla`**: **76 nodes × 4× H100** (plus 8 older 8× tesla nodes), walltime up to 7 days
- Other partitions: `standard` (CPU, 1280 nodes), `fast` (1h), `interactive` (10h)

## Storage layout (home is only 1 GB — never install/build there)

| Path | Quota | Use |
|---|---|---|
| `/mnt/storage_3/home/jentker` | 1 GB | ssh keys only |
| `/mnt/storage_6/project_data/pl1095-01` | 30 TB | **repo, uv env, tools, datasets, checkpoints** |
| `/mnt/storage_5/scratch/pl1095-01` | 20 TB | uv cache, temp job data |

- Repo: `/mnt/storage_6/project_data/pl1095-01/ShortcutFM` (branch fix/padding-noise)
- uv binary: `/mnt/storage_6/project_data/pl1095-01/tools/uv` (NOT on default PATH)
- Env vars needed for every uv call:
  `UV_CACHE_DIR=/mnt/storage_5/scratch/pl1095-01/uv_cache UV_LINK_MODE=copy`
  (link-mode copy because cache and venv are on different filesystems)

## Submitting training

```bash
ssh pcss 'cd /mnt/storage_6/project_data/pl1095-01/ShortcutFM && \
  UV_CACHE_DIR=/mnt/storage_5/scratch/pl1095-01/uv_cache UV_LINK_MODE=copy \
  nohup /mnt/storage_6/project_data/pl1095-01/tools/uv run python scripts/submit_training.py \
  configs/training/qqp_masked_r1_pcss.yaml <overrides> > /tmp/job.log 2>&1 < /dev/null &'
```

Config: `configs/training/qqp_masked_r1_pcss.yaml` (tesla partition, 4 GPUs/node).

## Gotchas

- **Home quota 1 GB** — HF cache, uv cache, and venv must live on project/scratch storage.
- Login node has internet; assume compute nodes may not — **pre-download HF models on the login node** (`AutoTokenizer/AutoConfig.from_pretrained` once) before submitting.
- First-time authorized_keys edit: an existing ssh-rsa key had no trailing newline, so an appended key got concatenated onto it and silently ignored — check `cat -A ~/.ssh/authorized_keys` if key auth fails.
- `exca` submissions block until job completion (same as other clusters) — always nohup-background them.
- **wandb API key must be in `~/.netrc`** (`machine api.wandb.ai ...`) — compute jobs die
  at `wandb login` otherwise (hit 2026-07-27, jobs 7810123/24). Copied from Athena's netrc.
- **venv must use a uv-managed Python on project storage** (`UV_PYTHON_INSTALL_DIR=.../tools/pythons`,
  then `uv venv --python 3.12`): the login node's `/usr/bin/python3.12` does not exist on
  compute nodes, so a system-python venv fails with `execve: No such file or directory`.
- 4-GPU-free tesla nodes are scarce; **2 GPUs + accumulate_grad_batches=8** (same effective
  batch as 4×acc4) schedules much faster on the 50+ mixed nodes.

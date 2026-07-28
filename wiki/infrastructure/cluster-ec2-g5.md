---
tags: [infrastructure, ec2, gpu, a10g]
date: 2026-07-28
related: [[cluster-pcss]] [[cluster-athena]]
---

# EC2 g5.24xlarge (zero-queue scratch box)

## Access (two-hop)

```bash
ssh m7i 'ssh -i ~/warczynj-llm2.pem -o StrictHostKeyChecking=no ec2-user@54.91.239.230 "<cmd>"'
```

- Instance `i-0ac0a95f20afeac92`, IP 54.91.239.230 (changes if stopped/started)
- Key `~/warczynj-llm2.pem` lives on **m7i** (alias in local ssh config), not locally
- m7i's ssh is old: use `StrictHostKeyChecking=no`, not `accept-new`

## Hardware / software (verified 2026-07-28)

- 4× A10G (23GB each), ~23 TFLOPS fp32-matmul per GPU in smoke test
- Deep Learning AMI: venv at `/opt/pytorch` (torch 2.12.1+cu130, Python 3.13)
- **Full outbound internet** (PyPI/GitHub/HF all reachable) — bootstrap via git clone + uv, no rsync needed

## Caveats

- **Root disk only 25GB** (12 free) — do NOT put repo/env/datasets there.
- `/dev/nvme1n1` = 3.5TB **unformatted ephemeral** NVMe. One-time setup:
  `sudo mkfs.ext4 /dev/nvme1n1 && sudo mkdir /scratch && sudo mount /dev/nvme1n1 /scratch && sudo chown ec2-user /scratch`
  Data there DIES with the instance — scratch training only; push checkpoints out.
- No SLURM — run directly with nohup/tmux. Zero queue time (its main advantage).
- Billed per hour — stop when idle.
- Not yet bootstrapped for ShortcutFM (as of 2026-07-28); pending user go-ahead.

## Intended use

Next-phase work where zero-queue latency matters: MT/summarization tokenization + first
runs, decode-sweep experiments, quick ablations. Recipe-search main line stays on
Athena/PCSS.

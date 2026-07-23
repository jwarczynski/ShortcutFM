#!/usr/bin/env bash
# Submit round-1 masked-diffusion recipe-search runs (10k steps each).
# Usage: bash scripts/submit_round1.sh <hgx|athena>
# hgx runs R1, R2, R5; athena runs R3, R4, R6 (see wiki/roadmap.md).
# Scheduler-switching variants (R2, R6) have dedicated YAMLs because the
# scheduler discriminated union rejects mixed-type CLI overrides.
set -euo pipefail

CLUSTER="${1:?usage: submit_round1.sh <hgx|athena>}"
if [[ "$CLUSTER" == "hgx" ]]; then
  BASE=configs/training/qqp_masked_r1_base.yaml
else
  BASE=configs/training/qqp_masked_r1_athena.yaml
fi

submit() {
  local name="$1" cfg="$2"; shift 2
  echo "=== submitting mdlm-r1-${name} ==="
  uv run python scripts/submit_training.py "$cfg" \
    "wandb.run_name=mdlm-r1-${name}" \
    "checkpoint.save_folder=checkpoints/qqp/mdlm_r1_${name}" \
    "$@"
}

if [[ "$CLUSTER" == "hgx" ]]; then
  # R1: pretrained BERT init (H2 — top pick)
  submit bert-init "$BASE" \
    model.use_pretrained_weights=true model.use_pretrained_embeddings=true

  # R2: LR warmup 2000 steps (H4) — dedicated myle-scheduler config
  submit warmup-2k configs/training/qqp_masked_r1_warmup.yaml

  # R5: no EMA (H7 — LLaDA uses none)
  submit no-ema "$BASE" ema=null
else
  # R3: mask-ratio bandwidth clipping (H8 — LLaDA2.0 SFT)
  submit t-clip "$BASE" \
    model.t_min_frac=0.15 model.t_max_frac=0.95

  # R4: delay consistency loss to step 5000 (H6)
  submit cons-delay-5k "$BASE" consistency_start_step=5000

  # R6: combo candidate recipe (bert-init + warmup + lr 3e-5 + cons delay)
  submit combo configs/training/qqp_masked_r1_combo_athena.yaml
fi

echo "All submissions for $CLUSTER done."

#!/bin/bash
# Full evaluation pipeline: generate → metrics by ref length → Themis
# Usage: bash scripts/eval_pipeline.sh <training_config> <checkpoint> <output_base> <dataset_label>
set -e

export PATH=$HOME/.local/bin:$PATH
cd /home/inf148234/projects/ShortcutFM

TCFG=$1
CKPT=$2
OUT_BASE=$3
LABEL=$4
BASELINE_DIR=$5

echo "=== Generating NFE=1 ==="
uv run python -m shortcutfm.decoding.generate configs/generation/qqp_scut_768_nfe.yaml \
    training_config_path=$TCFG \
    checkpoint_path=$CKPT \
    generation_shortcut_size=2048 denoising_step_size=2048 \
    output_folder=$OUT_BASE/scut=2048 \
    use_exca=false run_plot_analysis=false force_regeneration=true

SCUT_DIR=$OUT_BASE/scut=2048/seed_44

echo "=== Eval by ref length ==="
uv run python scripts/eval_by_ref_length.py \
    --gen_dirs $SCUT_DIR $BASELINE_DIR \
    --labels "Shortcut (fixed)" "Baseline" \
    --threshold 9

# Create Themis subdirs
for g in short long; do
    mkdir -p $SCUT_DIR/themis_$g
    cp $SCUT_DIR/generation_texts_${g}.json $SCUT_DIR/themis_$g/generation_texts_.json
done

echo "=== Themis (all) ==="
uv run python scripts/evaluate_themis.py \
    --generation_dir $SCUT_DIR \
    --model PKU-ONELab/Themis --batch_size 2 --suffix paper --max_samples 500 \
    --aspects paper_fluency paper_semantic_similarity paper_overall_quality

echo "=== Themis (short) ==="
uv run python scripts/evaluate_themis.py \
    --generation_dir $SCUT_DIR/themis_short \
    --model PKU-ONELab/Themis --batch_size 2 --suffix paper --max_samples 200 \
    --aspects paper_fluency paper_semantic_similarity paper_overall_quality

echo "=== Themis (long) ==="
uv run python scripts/evaluate_themis.py \
    --generation_dir $SCUT_DIR/themis_long \
    --model PKU-ONELab/Themis --batch_size 2 --suffix paper --max_samples 200 \
    --aspects paper_fluency paper_semantic_similarity paper_overall_quality

echo "=== Raw outputs ==="
uv run python scripts/dump_raw_outputs.py $SCUT_DIR

echo "=== DONE ==="

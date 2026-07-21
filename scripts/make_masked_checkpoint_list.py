"""Build the checkpoint list for the masked-diffusion NFE sweep.

Finds the highest-step checkpoint in each masked-diffusion run directory and
writes the pipe-separated list consumed by generate.py's checkpoint_list_file
flow (checkpoint|training_config|subdir|run_label|step).

Usage (on the cluster, after training finishes):
    uv run python scripts/make_masked_checkpoint_list.py
    uv run python -m shortcutfm.decoding.generate configs/generation/qqp_masked_nfe.yaml
"""

import re
import sys
from pathlib import Path

RUN_DIRS = [
    ("checkpoints/qqp/masked_diffusion", "masked_diffusion"),
    ("checkpoints/qqp/masked_no_consistency", "masked_no_consistency"),
]
OUT_FILE = Path("configs/generation/individual_runs/qqp/masked_diffusion_checkpoints.txt")

STEP_RE = re.compile(r"step=(\d+)")


def find_best_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
    best = None
    for ckpt in run_dir.glob("*.ckpt"):
        match = STEP_RE.search(ckpt.name)
        step = int(match.group(1)) if match else 0
        if best is None or step > best[1]:
            best = (ckpt, step)
    return best


def main() -> int:
    lines = []
    for base, subdir in RUN_DIRS:
        base_path = Path(base)
        if not base_path.exists():
            print(f"WARNING: {base} does not exist, skipping")
            continue
        # wandb-run subdirectories look like run_<id>; skip failed-logger artifacts
        for run_dir in sorted(base_path.glob("run_*")):
            if not run_dir.is_dir() or "<" in run_dir.name:
                continue
            training_config = run_dir / "training_config.yaml"
            if not training_config.exists():
                print(f"WARNING: no training_config.yaml in {run_dir}, skipping")
                continue
            best = find_best_checkpoint(run_dir)
            if best is None:
                print(f"WARNING: no checkpoints in {run_dir}, skipping")
                continue
            ckpt, step = best
            lines.append(f"{ckpt}|{training_config}|{subdir}|{subdir}|{step}")
            print(f"{subdir}/{run_dir.name}: {ckpt.name} (step {step})")

    if not lines:
        print("ERROR: no checkpoints found — has training reached the first val_interval?")
        return 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} entries to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

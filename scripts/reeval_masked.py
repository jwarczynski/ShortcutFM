"""R0 re-evaluation of masked-diffusion checkpoints with corrected decoding.

Evaluates checkpoints on the validation split across an NFE x unmask-strategy grid,
reporting BLEU and source-copy%%. No-consistency checkpoints are conditioned on d=0
(their shortcut embedding is untrained for d != 0).

Usage (on a GPU node):
    uv run python scripts/reeval_masked.py [--limit-batches N] [--out results_r0_reeval.md]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shortcutfm.batch import collate
from shortcutfm.config import TrainingConfig
from shortcutfm.evaluation import compute_generation_metrics_from_batch
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.trainer_factory import create_criterion, load_unit_from_checkpoint

CHECKPOINTS = [
    # (label, run_dir, ckpt_name, consistency_trained)
    ("main-50k", "checkpoints/qqp/masked_diffusion/run_r43zphwi", "epoch=704-step=50000-val_bleu=0.0000.ckpt", True),
    ("main-10k", "checkpoints/qqp/masked_diffusion/run_r43zphwi", "epoch=140-step=10000-val_bleu=0.0000.ckpt", True),
    ("ablation-50k", "checkpoints/qqp/masked_no_consistency/run_exu5bftd", "epoch=704-step=50000-val_bleu=0.0000.ckpt", False),
    ("ablation-10k", "checkpoints/qqp/masked_no_consistency/run_exu5bftd", "epoch=140-step=10000-val_bleu=0.0000.ckpt", False),
]
NFES = [1, 4, 16, 64]
STRATEGIES = ["confidence", "random"]


def find_checkpoint(run_dir: Path, preferred: str) -> Path:
    ckpt = run_dir / preferred
    if ckpt.exists():
        return ckpt
    # fall back to matching the step number in the name
    step = preferred.split("step=")[1].split("-")[0]
    matches = list(run_dir.glob(f"*step={step}*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for step {step} in {run_dir}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="results_r0_reeval.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = [
        "# R0 re-evaluation — masked diffusion, corrected decoding",
        "",
        "Validation split, BLEU (NLTK method4) + copy%. Ablation conditioned on d=0.",
        "",
    ]

    for label, run_dir, ckpt_name, consistency_trained in CHECKPOINTS:
        run_path = Path(run_dir)
        cfg_path = run_path / "training_config.yaml"
        if not cfg_path.exists():
            print(f"SKIP {label}: {cfg_path} missing")
            continue
        ckpt_path = find_checkpoint(run_path, ckpt_name)
        training_cfg = TrainingConfig(**om.to_container(om.load(str(cfg_path)), resolve=True))

        tokenizer = AutoTokenizer.from_pretrained(training_cfg.model.tokenizer_config_name)
        criterion = create_criterion(training_cfg, tokenizer)
        unit = load_unit_from_checkpoint(criterion, ckpt_path, training_cfg, tokenizer)
        unit.to(device).eval()

        val_ds = TextDataset(Dataset.load_from_disk(training_cfg.validation_data_path))
        loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

        T = training_cfg.model.diffusion_steps
        lines += [f"## {label} ({ckpt_path.name})", "", "| NFE | strategy | d | BLEU | copy% |", "|---|---|---|---|---|"]

        for nfe in NFES:
            step_size = max(T // nfe, 1)
            shortcut = step_size if consistency_trained else 0
            for strategy in STRATEGIES:
                agg = {"bleu": 0.0, "copy_pct": 0.0}
                n_batches = 0
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        if args.limit_batches and i >= args.limit_batches:
                            break
                        batch = batch.to(device)
                        preds = unit.criterion.denoise(
                            batch,
                            shortcut_size=shortcut,
                            probe_every_step=False,
                            return_logits=False,
                            step_size=step_size,
                            unmask_strategy=strategy,
                        )
                        m = compute_generation_metrics_from_batch(batch, preds, tokenizer)
                        agg["bleu"] += m["bleu"]
                        agg["copy_pct"] += m["copy_pct"]
                        n_batches += 1
                bleu = agg["bleu"] / max(n_batches, 1)
                copy_pct = agg["copy_pct"] / max(n_batches, 1)
                lines.append(f"| {nfe} | {strategy} | {shortcut} | {bleu:.4f} | {copy_pct:.1f} |")
                print(f"{label} NFE={nfe} {strategy} d={shortcut}: BLEU={bleu:.4f} copy={copy_pct:.1f}%")
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

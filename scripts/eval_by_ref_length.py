"""Evaluate generation outputs split by reference length (< 9 words vs >= 9 words).

Computes BLEU, ROUGE, BERTScore, distinct-n, and source copy rate per group.
Also prepares Themis input files per group for separate Themis evaluation.

Usage:
    uv run python scripts/eval_by_ref_length.py \
        --gen_dirs dir1 dir2 \
        --labels "Shortcut (fixed)" "Baseline" \
        --threshold 9
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_gen_texts(gen_dir: Path) -> list[dict]:
    for name in ["generation_texts.json", "generation_texts_.json"]:
        p = gen_dir / name
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"No generation_texts in {gen_dir}")


def compute_metrics(items: list[dict]) -> dict:
    if not items:
        return {}
    sources = [x["source"] for x in items]
    references = [x["reference"] for x in items]
    hypotheses = [x["hypothesis"] for x in items]

    # BLEU
    tok_hyp = [h.split() for h in hypotheses]
    tok_ref = [[r.split()] for r in references]
    smooth = SmoothingFunction().method4
    bleu = corpus_bleu(tok_ref, tok_hyp, smoothing_function=smooth)

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=hypotheses, references=references)

    # BERTScore
    bert_score = evaluate.load("bertscore")
    bs = bert_score.compute(predictions=hypotheses, references=references, model_type="microsoft/deberta-xlarge-mnli")
    bs_f1 = sum(bs["f1"]) / len(bs["f1"])

    # Distinct-n
    def distinct_n(texts, n):
        all_ngrams = []
        for t in texts:
            tokens = t.split()
            all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
        return len(set(all_ngrams)) / max(len(all_ngrams), 1)

    # Source copy rate
    src_copy = sum(1 for s, h in zip(sources, hypotheses) if s.strip() == h.strip()) / len(items) * 100

    return {
        "n": len(items),
        "bleu": round(bleu, 4),
        "rouge1": round(float(rouge_scores["rouge1"]), 4),
        "rougeL": round(float(rouge_scores["rougeL"]), 4),
        "bertscore_f1": round(bs_f1, 4),
        "distinct_1": round(distinct_n(hypotheses, 1), 4),
        "distinct_2": round(distinct_n(hypotheses, 2), 4),
        "src_copy_pct": round(src_copy, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--threshold", type=int, default=9)
    parser.add_argument("--output", type=str, default="eval_by_ref_length.json")
    args = parser.parse_args()

    results = {}
    for gen_dir_str, label in zip(args.gen_dirs, args.labels):
        gen_dir = Path(gen_dir_str)
        items = load_gen_texts(gen_dir)
        logger.info(f"Loaded {len(items)} items from {gen_dir} ({label})")

        short = [x for x in items if len(x["reference"].split()) < args.threshold]
        long = [x for x in items if len(x["reference"].split()) >= args.threshold]

        logger.info(f"  Short (< {args.threshold} words): {len(short)}")
        logger.info(f"  Long (>= {args.threshold} words): {len(long)}")

        results[label] = {
            "all": compute_metrics(items),
            f"short_ref_lt{args.threshold}": compute_metrics(short),
            f"long_ref_gte{args.threshold}": compute_metrics(long),
        }

        # Save per-group generation_texts for Themis
        for group_name, group_items in [("short", short), ("long", long)]:
            out_path = gen_dir / f"generation_texts_{group_name}.json"
            json.dump(group_items, open(out_path, "w"), indent=2, ensure_ascii=False)
            logger.info(f"  Saved {len(group_items)} items to {out_path}")

    # Print results
    for label, data in results.items():
        for group, metrics in data.items():
            logger.info(f"{label} | {group}: {metrics}")

    # Save
    out_path = Path(args.gen_dirs[0]).parent / args.output
    json.dump(results, open(out_path, "w"), indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

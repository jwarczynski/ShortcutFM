"""Evaluate generation outputs split by reference length.

Usage:
    uv run python scripts/eval_by_length.py \
        --dirs dir1 dir2 ... \
        --threshold 9 \
        --run_themis \
        --themis_model PKU-ONELab/Themis
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def load_gen_texts(gen_dir: Path) -> list[dict]:
    for name in ["generation_texts.json", "generation_texts_.json"]:
        p = gen_dir / name
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"No generation_texts in {gen_dir}")


def compute_metrics(items: list[dict]) -> dict:
    """Compute BLEU, ROUGE, BERTScore, distinct-n, src_copy% on a subset."""
    import evaluate
    import torch
    from bert_score import score as bert_score_fn

    refs = [d["reference"] for d in items]
    hyps = [d["hypothesis"] for d in items]
    srcs = [d["source"] for d in items]
    n = len(items)

    # BLEU
    tok_hyps = [h.split() for h in hyps]
    tok_refs = [[r.split()] for r in refs]
    bleu = corpus_bleu(tok_refs, tok_hyps, smoothing_function=SmoothingFunction().method4)

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=hyps, references=refs)

    # BERTScore
    P, R, F1 = bert_score_fn(hyps, refs, model_type="microsoft/deberta-xlarge-mnli", verbose=False)
    bs_f1 = F1.mean().item()

    # Distinct n-grams
    def distinct_n(texts, n):
        all_ngrams = []
        for t in texts:
            tokens = t.split()
            all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
        return len(set(all_ngrams)) / max(len(all_ngrams), 1)

    # Src copy %
    src_copy = sum(1 for s, h in zip(srcs, hyps) if s.strip() == h.strip()) / max(n, 1) * 100

    return {
        "n": n,
        "bleu": round(bleu, 4),
        "rouge1": round(float(rouge_scores["rouge1"]), 4),
        "rougeL": round(float(rouge_scores["rougeL"]), 4),
        "bertscore_f1": round(bs_f1, 4),
        "distinct_1": round(distinct_n(hyps, 1), 4),
        "distinct_2": round(distinct_n(hyps, 2), 4),
        "src_copy_pct": round(src_copy, 1),
    }


def run_themis_on_subset(items: list[dict], model, tokenizer, aspects, batch_size=2, max_new_tokens=2048):
    """Run Themis on a subset, reusing already-loaded model."""
    import math
    import torch

    PROMPT = (
        "###Instruction###\n"
        "Please act as an impartial and helpful evaluator for natural language generation (NLG), "
        "and the audience is an expert in the field.\n"
        "Your task is to evaluate the quality of {task} strictly based on the given evaluation criterion.\n"
        "Begin the evaluation by providing your analysis concisely and accurately, and then on the next line, "
        'start with "Rating:" followed by your rating on a Likert scale from 1 to 5 (higher means better).\n'
        "You MUST keep to the strict boundaries of the evaluation criterion and focus solely on the issues "
        "and errors involved; otherwise, you will be penalized.\n"
        "Make sure you read and understand these instructions, as well as the following evaluation criterion "
        "and example content, carefully.\n\n"
        "###Evaluation Criterion###\n{aspect}\n\n"
        "###Example###\n{source_des}:\n{source}\n\n{target_des}:\n{target}\n\n"
        "###Your Evaluation###\n"
    )
    ASPECT_DEFS = {
        "paper_fluency": "Fluency: Whether the paraphrase is meaningful and grammatical?",
        "paper_semantic_similarity": "Semantic Similarity: Whether the paraphrase maintains similar semantics to the original text?",
        "paper_overall_quality": "Overall Quality: The paraphrase should not only maintain similar semantics to the original text, but also possess lexical or syntactic differences from the original text, with fluent and coherent content",
    }

    device = next(model.parameters()).device
    results = {}

    for aspect_key in aspects:
        prompts = []
        for item in items:
            prompts.append(PROMPT.format(
                task="Paraphrase Generation",
                aspect=ASPECT_DEFS[aspect_key],
                source_des="Source Question", source=item["source"],
                target_des="Generated Paraphrase", target=item["hypothesis"],
            ))

        ratings = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
            for j, g in enumerate(gen):
                new_tokens = g[inputs["input_ids"][j].shape[0]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                last_line = text.strip().split("\n")[-1]
                rating = 0.0
                if last_line.startswith("Rating: "):
                    try:
                        r = float(last_line[8:])
                        if math.isfinite(r):
                            rating = r
                    except ValueError:
                        pass
                ratings.append(rating)

        avg = sum(ratings) / max(len(ratings), 1)
        results[aspect_key] = round(avg, 2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True, help="Generation output dirs")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each dir")
    parser.add_argument("--threshold", type=int, default=9, help="Word count threshold for splitting")
    parser.add_argument("--run_themis", action="store_true")
    parser.add_argument("--themis_model", default="PKU-ONELab/Themis")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    labels = args.labels or [Path(d).parts[-3] + "/" + Path(d).parts[-2] for d in args.dirs]
    aspects = ["paper_fluency", "paper_semantic_similarity", "paper_overall_quality"]

    # Load Themis model once if needed
    themis_model = themis_tokenizer = None
    if args.run_themis:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading Themis: {args.themis_model}")
        themis_tokenizer = AutoTokenizer.from_pretrained(args.themis_model, padding_side="left")
        if themis_tokenizer.pad_token is None:
            themis_tokenizer.pad_token = themis_tokenizer.eos_token
        themis_model = AutoModelForCausalLM.from_pretrained(args.themis_model, torch_dtype=torch.float16)
        themis_model = themis_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    all_results = {}
    for gen_dir, label in zip(args.dirs, labels):
        gen_dir = Path(gen_dir)
        items = load_gen_texts(gen_dir)
        short = [d for d in items if len(d["reference"].split()) < args.threshold]
        long_ = [d for d in items if len(d["reference"].split()) >= args.threshold]

        logger.info(f"\n{'='*60}")
        logger.info(f"{label}: {len(items)} total, {len(short)} short (<{args.threshold} words), {len(long_)} long")

        result = {"label": label}
        for split_name, subset in [("short", short), ("long", long_), ("all", items)]:
            if not subset:
                continue
            logger.info(f"  Computing metrics for {split_name} ({len(subset)} examples)...")
            m = compute_metrics(subset)
            if args.run_themis and themis_model is not None:
                logger.info(f"  Running Themis for {split_name}...")
                t = run_themis_on_subset(subset, themis_model, themis_tokenizer, aspects, args.batch_size)
                m.update(t)
            result[split_name] = m
            logger.info(f"  {split_name}: {m}")

        all_results[label] = result

    # Save
    out_path = Path("eval_by_length_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # Print summary table
    print("\n" + "=" * 100)
    for label, res in all_results.items():
        for split_name in ["short", "long", "all"]:
            if split_name not in res:
                continue
            m = res[split_name]
            line = f"{label:30s} {split_name:5s} n={m['n']:3d}  BLEU={m['bleu']:.4f}  R1={m['rouge1']:.4f}  RL={m['rougeL']:.4f}  BS={m['bertscore_f1']:.4f}  Copy={m['src_copy_pct']:.1f}%"
            if "paper_fluency" in m:
                line += f"  Flu={m['paper_fluency']:.2f}  Sem={m['paper_semantic_similarity']:.2f}  OQ={m['paper_overall_quality']:.2f}"
            print(line)


if __name__ == "__main__":
    main()

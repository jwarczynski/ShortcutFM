"""
Evaluate generation outputs using the Themis model (PKU-ONELab/Themis).
Themis is a reference-free NLG evaluator that provides ratings + analysis.

This script:
1. Reads generation_texts.json from a generation output directory
2. Formats data for Themis evaluation across multiple aspects
3. Runs Themis inference via vLLM
4. Saves results alongside existing metrics

Usage (on GPU node):
    uv run python scripts/evaluate_themis.py \
        --generation_dir generation_outputs/qqp/scut_768/run_ll9cnmi5/scut=2048/seed_44 \
        --model PKU-ONELab/Themis \
        --aspects fidelity fluency lexical_similarity overall_quality

Or via exca:
    uv run python scripts/evaluate_themis.py \
        --generation_dir generation_outputs/qqp/scut_768/run_ll9cnmi5/scut=2048/seed_44 \
        --use_exca
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Themis prompt template (no additional info)
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
    "and example content, carefully.\n"
    "\n"
    "###Evaluation Criterion###\n"
    "{aspect}\n"
    "\n"
    "###Example###\n"
    "{source_des}:\n"
    "{source}\n"
    "\n"
    "{target_des}:\n"
    "{target}\n"
    "\n"
    "###Your Evaluation###\n"
)

# Aspect definitions for paraphrase generation (QQP task)
ASPECT_DEFINITIONS = {
    "fidelity": (
        "Fidelity/Faithfulness: Evaluate whether the generated paraphrase preserves the original meaning "
        "of the source question. A high-fidelity paraphrase should convey the same intent and information "
        "as the source, without adding, omitting, or distorting any key details. "
        "Rate 1 if the meaning is completely changed, 5 if the meaning is perfectly preserved."
    ),
    "fluency": (
        "Fluency: Evaluate the grammatical correctness and naturalness of the generated paraphrase. "
        "A fluent paraphrase should read naturally, be grammatically correct, and use appropriate vocabulary. "
        "Rate 1 if the text is incomprehensible or severely ungrammatical, 5 if it is perfectly fluent and natural."
    ),
    "lexical_similarity": (
        "Lexical Diversity: Evaluate how much the generated paraphrase differs from the source in terms of "
        "word choice and phrasing while still preserving meaning. A good paraphrase should use different words "
        "and sentence structures rather than copying the source verbatim. "
        "Rate 1 if it is an exact copy, 5 if it uses substantially different wording while keeping the meaning."
    ),
    "overall_quality": (
        "Overall Quality: Evaluate the overall quality of the generated paraphrase considering all aspects: "
        "meaning preservation, fluency, naturalness, and lexical diversity. "
        "A high-quality paraphrase should faithfully convey the original meaning using different wording "
        "in a natural and grammatically correct way. "
        "Rate 1 for very poor quality, 5 for excellent quality."
    ),
}

# Aspect definitions from the Themis paper appendix (used in their training data)
PAPER_ASPECT_DEFINITIONS = {
    "paper_fluency": (
        "Fluency: Whether the paraphrase is meaningful and grammatical?"
    ),
    "paper_semantic_similarity": (
        "Semantic Similarity: Whether the paraphrase maintains similar semantics to the original text?"
    ),
    "paper_overall_quality": (
        "Overall Quality: The paraphrase should not only maintain similar semantics to the original text, "
        "but also possess lexical or syntactic differences from the original text, "
        "with fluent and coherent content"
    ),
}

ALL_ASPECTS = {**ASPECT_DEFINITIONS, **PAPER_ASPECT_DEFINITIONS}


def load_generation_texts(generation_dir: Path) -> list[dict]:
    """Load generation_texts.json from a generation output directory."""
    # Try with and without suffix
    for name in ["generation_texts.json", "generation_texts_.json"]:
        path = generation_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(f"No generation_texts.json found in {generation_dir}")


def format_for_themis(
    generation_texts: list[dict],
    aspect_name: str,
) -> list[dict]:
    """Format generation outputs for Themis evaluation."""
    aspect_criterion = ALL_ASPECTS[aspect_name]
    samples = []
    for item in generation_texts:
        samples.append({
            "task": "Paraphrase Generation",
            "aspect": aspect_criterion,
            "source_des": "Source Question",
            "source": item["source"],
            "target_des": "Generated Paraphrase",
            "target": item["hypothesis"],
        })
    return samples


def get_prompt(ex: dict) -> str:
    """Build Themis prompt from a sample dict."""
    return PROMPT.format_map(ex)


def parse_rating(out: str) -> dict:
    """Parse Themis output into analysis + rating."""
    last_line = out.strip().split("\n")[-1]
    if last_line.startswith("Rating: "):
        try:
            rating = float(last_line[8:])
            if math.isfinite(rating):
                return {
                    "Analysis": "\n".join(out.strip().split("\n")[:-1]),
                    "Rating": rating,
                }
        except ValueError:
            pass
    return {"Analysis": out.strip(), "Rating": 0}


def _run_transformers(
    prompts_by_aspect: dict[str, list[str]],
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, list[str]]:
    """Run inference using HuggingFace transformers (no vllm needed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading Themis model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    raw_outputs: dict[str, list[str]] = {}
    for aspect, prompts in prompts_by_aspect.items():
        logger.info(f"Evaluating aspect: {aspect} ({len(prompts)} samples)")
        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode only the new tokens
            for j, gen in enumerate(generated):
                input_len = inputs["input_ids"][j].shape[0]
                new_tokens = gen[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  {aspect}: {min(i + batch_size, len(prompts))}/{len(prompts)}")

        raw_outputs[aspect] = outputs

    return raw_outputs


def _run_vllm(
    prompts_by_aspect: dict[str, list[str]],
    model_name: str,
    tensor_parallel_size: int,
    max_new_tokens: int,
) -> dict[str, list[str]]:
    """Run inference using vLLM (faster)."""
    from vllm import LLM, SamplingParams

    logger.info(f"Loading Themis model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.95,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    raw_outputs: dict[str, list[str]] = {}
    for aspect, prompts in prompts_by_aspect.items():
        logger.info(f"Evaluating aspect: {aspect} ({len(prompts)} samples)")
        results = llm.generate(prompts, sampling_params)
        raw_outputs[aspect] = [r.outputs[0].text for r in results]

    return raw_outputs


def run_themis_evaluation(
    generation_dir: Path,
    model_name: str,
    aspects: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    use_vllm: bool = False,
    tensor_parallel_size: int = 1,
    max_samples: int | None = None,
    suffix: str = "",
) -> dict:
    """Run Themis evaluation on generation outputs.

    Supports two backends:
    - vllm (faster, requires vllm installed): use_vllm=True
    - transformers (no extra deps): use_vllm=False (default)
    """
    import random as _rng

    generation_texts = load_generation_texts(generation_dir)
    logger.info(f"Loaded {len(generation_texts)} generation samples from {generation_dir}")

    if max_samples is not None and max_samples < len(generation_texts):
        _rng.seed(42)
        generation_texts = _rng.sample(generation_texts, max_samples)
        logger.info(f"Subsampled to {max_samples} samples for Themis evaluation")

    # Prepare all prompts per aspect
    prompts_by_aspect: dict[str, list[str]] = {}
    for aspect in aspects:
        samples = format_for_themis(generation_texts, aspect)
        prompts_by_aspect[aspect] = [get_prompt(s) for s in samples]

    total_prompts = sum(len(v) for v in prompts_by_aspect.values())
    logger.info(f"Total prompts: {total_prompts} ({len(aspects)} aspects × {len(generation_texts)} samples)")

    if use_vllm:
        raw_outputs = _run_vllm(prompts_by_aspect, model_name, tensor_parallel_size, max_new_tokens)
    else:
        raw_outputs = _run_transformers(prompts_by_aspect, model_name, batch_size, max_new_tokens)

    # Parse results per aspect
    results = {}
    for aspect in aspects:
        aspect_outputs = raw_outputs[aspect]  # list of output strings
        aspect_results = []
        ratings = []
        for idx in range(len(generation_texts)):
            parsed = parse_rating(aspect_outputs[idx])
            aspect_results.append({
                "source": generation_texts[idx]["source"],
                "hypothesis": generation_texts[idx]["hypothesis"],
                "reference": generation_texts[idx].get("reference", ""),
                "analysis": parsed["Analysis"],
                "rating": parsed["Rating"],
            })
            ratings.append(parsed["Rating"])

        avg_rating = sum(ratings) / max(len(ratings), 1)
        valid_ratings = [r for r in ratings if r > 0]
        avg_valid = sum(valid_ratings) / max(len(valid_ratings), 1) if valid_ratings else 0

        results[aspect] = {
            "average_rating": avg_rating,
            "average_valid_rating": avg_valid,
            "num_samples": len(ratings),
            "num_valid": len(valid_ratings),
            "details": aspect_results,
        }
        logger.info(f"  {aspect}: avg={avg_rating:.3f} (valid: {avg_valid:.3f}, {len(valid_ratings)}/{len(ratings)})")

    # Save results
    sfx = f"_{suffix}" if suffix else ""
    output_path = generation_dir / f"themis_metrics{sfx}.json"
    summary = {
        aspect: {k: v for k, v in data.items() if k != "details"}
        for aspect, data in results.items()
    }
    summary_path = generation_dir / f"themis_summary{sfx}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved Themis results to {output_path}")
    logger.info(f"Saved Themis summary to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation outputs with Themis")
    parser.add_argument("--generation_dir", required=True, type=str, help="Path to generation output directory")
    parser.add_argument("--model", default="PKU-ONELab/Themis", type=str, help="Themis model name/path")
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=["fidelity", "fluency", "lexical_similarity", "overall_quality"],
        choices=list(ALL_ASPECTS.keys()),
        help="Evaluation aspects",
    )
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend (faster, requires vllm)")
    parser.add_argument("--tensor_parallel_size", "-tp", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for transformers backend")
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--max_samples", default=None, type=int, help="Max samples to evaluate (random subsample)")
    parser.add_argument("--suffix", default="", type=str, help="Suffix for output filenames (e.g. 'paper')")

    args = parser.parse_args()

    generation_dir = Path(args.generation_dir)
    if not generation_dir.exists():
        logger.error(f"Generation directory does not exist: {generation_dir}")
        sys.exit(1)

    run_themis_evaluation(
        generation_dir=generation_dir,
        model_name=args.model,
        aspects=args.aspects,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        max_samples=args.max_samples,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()

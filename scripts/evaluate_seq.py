"""
Standalone script for evaluating generated sequences.
This script now uses the evaluation module from shortcutfm.evaluation.
"""

import logging
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf import OmegaConf as om

# add shotcutfm to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from shortcutfm.evaluation import evaluate_generations
from shortcutfm.evaluation_config import EvaluationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_results(metrics):
    """Print evaluation metrics in a formatted way."""
    print("\nEvaluation Results:")
    print("==================")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                if isinstance(sub_value, list):  # Handle lists (like 'precisions')
                    print(f"  {sub_metric}: {[round(x, 4) for x in sub_value]}")
                else:  # Handle single floats
                    print(f"  {sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")


def find_seed_directories(base_path: Path) -> list[Path]:
    """
    Find all directories starting with 'seed' in the given path and subdirectories.
    If multiple versions exist (e.g., seed_44 and seed_44_v1), return only the highest version.

    Args:
        base_path: Base directory to search in

    Returns:
        List of paths to seed directories with highest versions
    """
    seed_dirs = []

    # Recursively find all directories starting with "seed"
    for path in base_path.rglob("seed_*"):
        if path.is_dir():
            seed_dirs.append(path)

    # Group directories by their base name (without version)
    grouped_dirs = {}
    version_pattern = re.compile(r'(.+?)(?:_v(\d+))?$')

    for dir_path in seed_dirs:
        match = version_pattern.match(dir_path.name)
        if match:
            base_name = match.group(1)
            version = int(match.group(2)) if match.group(2) else 0

            # Get the parent directory as context (to handle same seed names in different locations)
            context_key = (str(dir_path.parent), base_name)

            if context_key not in grouped_dirs or version > grouped_dirs[context_key][1]:
                grouped_dirs[context_key] = (dir_path, version)

    # Return only the highest version directories
    return [dir_info[0] for dir_info in grouped_dirs.values()]


def run_batch_evaluation(qqp_base_path: Path, config: EvaluationConfig) -> None:
    """
    Find all seed directories in QQP folder and submit evaluation jobs for each.

    Args:
        qqp_base_path: Path to the QQP generation outputs folder
        config: Base evaluation configuration
    """
    logger.info(f"Searching for seed directories in: {qqp_base_path}")

    if not qqp_base_path.exists():
        logger.error(f"QQP base path does not exist: {qqp_base_path}")
        return

    seed_directories = find_seed_directories(qqp_base_path)

    if not seed_directories:
        logger.warning("No seed directories found!")
        return

    logger.info(f"Found {len(seed_directories)} seed directories")

    # Filter out directories that already have metrics files (unless force_overwrite is True)
    if config.skip_existing and not config.force_overwrite:
        filtered_directories = []
        skipped_count = 0

        for seed_dir in seed_directories:
            # Check for existing metrics file (with or without suffix)
            metrics_file = seed_dir / "metrics.json"
            if config.suffix:
                metrics_file_with_suffix = seed_dir / f"metrics_{config.suffix}.json"
            else:
                metrics_file_with_suffix = metrics_file

            if metrics_file.exists() or (config.suffix and metrics_file_with_suffix.exists()):
                logger.info(f"Skipping {seed_dir} - metrics file already exists")
                skipped_count += 1
            else:
                filtered_directories.append(seed_dir)

        seed_directories = filtered_directories
        logger.info(f"After filtering: {len(seed_directories)} directories to evaluate, {skipped_count} skipped")

    if not seed_directories:
        logger.warning("No directories need evaluation (all already have metrics files)")
        return

    logger.info("Directories to evaluate:")
    for seed_dir in seed_directories:
        logger.info(f"  - {seed_dir}")

    if not config.use_exca:
        # Run evaluations directly
        for seed_dir in seed_directories:
            logger.info(f"Evaluating: {seed_dir}")
            try:
                metrics = evaluate_generations(
                    seed_dir,
                    config.tokenizer,
                    config.device,
                    config.use_fallback_processing,
                    suffix=config.suffix
                )
                print_results(metrics)
                logger.info(f"Completed evaluation for: {seed_dir}")
            except Exception as e:
                logger.error(f"Evaluation failed for {seed_dir}: {e}")
    else:
        # Use exca for job submission
        logger.info("Using exca for batch job submission...")

        with config.infra.job_array() as array:
            for seed_dir in seed_directories:
                eval_config_copy = config.infra.clone_obj({
                    "output_dir": str(seed_dir),
                })
                array.append(eval_config_copy)


def parse_evaluation_config(config_path: str, args_list: list[str]) -> EvaluationConfig:
    """Parse and validate evaluation config from YAML file"""
    if not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(yaml_cfg, om.from_cli(args_list))
    merged_dict = OmegaConf.to_container(merged_cfg, resolve=True)

    # Use model_validate to create config which will trigger field validators
    cfg = EvaluationConfig(**merged_dict)  # type: ignore
    return cfg


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_seq.py <config_path> <cli_args>")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Parse and validate evaluation config
    eval_cfg = parse_evaluation_config(yaml_path, args_list)
    logger.info("Evaluation Configuration:\n" + om.to_yaml(eval_cfg.model_dump()))

    if eval_cfg.batch_qqp:
        # Batch evaluation mode: find all seed directories in QQP folder
        qqp_base_path = Path(eval_cfg.output_dir)
        run_batch_evaluation(qqp_base_path, eval_cfg)
    else:
        # Single evaluation mode: evaluate specific directory
        output_dir = Path(eval_cfg.output_dir)
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            sys.exit(1)

        # Check for existing metrics file if skip_existing is True
        if eval_cfg.skip_existing and not eval_cfg.force_overwrite:
            metrics_file = output_dir / "metrics.json"
            if eval_cfg.suffix:
                metrics_file_with_suffix = output_dir / f"metrics_{eval_cfg.suffix}.json"
            else:
                metrics_file_with_suffix = metrics_file

            if metrics_file.exists() or (eval_cfg.suffix and metrics_file_with_suffix.exists()):
                logger.warning(f"Metrics file already exists in {output_dir}. Use --force_overwrite=true to overwrite.")
                sys.exit(0)

        if not eval_cfg.use_exca:
            # Run evaluation directly
            try:
                metrics = evaluate_generations(
                    output_dir,
                    eval_cfg.tokenizer,
                    eval_cfg.device,
                    eval_cfg.use_fallback_processing,
                    suffix=eval_cfg.suffix
                )
                print_results(metrics)
                logger.info("Evaluation completed successfully!")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                sys.exit(1)
        else:
            # Use exca for single job submission
            logger.info("Using exca for job submission...")
            eval_cfg.run_evaluation()


if __name__ == "__main__":
    main()

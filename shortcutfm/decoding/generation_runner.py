"""
Generation logic extraction module.

This module contains the core generation functionality extracted from generate.py
to eliminate code duplication and ensure consistent behavior between direct
generation and exca job submission.
"""

import logging
from pathlib import Path

import lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import Dataset
from shortcutfm.batch import collate
from shortcutfm.config import GenerationConfig
from shortcutfm.evaluation import evaluate_generations
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import SaveTestOutputsCallback
from shortcutfm.train.pl.trainer_factory import (
    create_criterion,
    get_ema_callback,
    load_unit_from_checkpoint,
)

logger = logging.getLogger(__name__)


def determine_test_data_path(training_config_path: str, split: str) -> str:
    """
    Automatically determine test_data_path based on training config and split.

    Args:
        training_config_path: Path to the training configuration file
        split: Dataset split to use ("test" or "valid")

    Returns:
        Path to the test/validation dataset

    Raises:
        ValueError: If dataset cannot be determined or path doesn't exist
    """
    # Extract dataset name
    dataset = determine_dataset_from_config_path(training_config_path)

    # Load training config to get tokenizer info
    if not Path(training_config_path).exists():
        raise ValueError(f"Training config file not found: {training_config_path}")

    with open(training_config_path) as f:
        yaml_cfg = OmegaConf.load(f)

    # Get tokenizer name from training config
    # Convert OmegaConf to regular dict for easier access
    yaml_dict = OmegaConf.to_container(yaml_cfg, resolve=True)
    if isinstance(yaml_dict, dict):
        model_cfg = yaml_dict.get("model", {})
        if isinstance(model_cfg, dict):
            tokenizer_name = model_cfg.get("tokenizer_config_name", "bert-base-uncased")
        else:
            tokenizer_name = "bert-base-uncased"
    else:
        tokenizer_name = "bert-base-uncased"

    # Convert tokenizer name to directory format
    if "/" in tokenizer_name:
        tokenizer_dir = tokenizer_name.split("/")[-1]
    else:
        tokenizer_dir = tokenizer_name

    # Map dataset names to their directory names
    dataset_dir_mapping = {
        "qqp": "QQP-Official",
        "webnlg": "webnlg",
        "wmt19": "wmt",
        "commonsenseconversation": "CommonsenseConversation",
        "parasci": "parasci",
        "pawswiki": "paws_wiki",
        "quasar": "Quasar-T",
        "wiki": "Wiki-alignment",
        "grammar": "grammar_correction",
    }

    dataset_dir = dataset_dir_mapping.get(dataset)
    if not dataset_dir:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Special handling for WMT19 which uses opus-mt-en-de tokenizer
    if dataset == "wmt19":
        tokenizer_dir = "opus-mt-en-de"

    # Construct the path
    test_data_path = f"datasets/tokenized/{tokenizer_dir}/{dataset_dir}/{split}"

    # Validate path exists
    if not Path(test_data_path).exists():
        raise ValueError(f"Test data path does not exist: {test_data_path}")

    logger.info(f"Auto-determined test_data_path: {test_data_path}")
    return test_data_path


def parse_checkpoint_path(line: str) -> dict[str, str]:
    """
    Parse a line from checkpoint list file.

    Format: checkpoint_path|training_config_path|subdir|run_id|step

    Args:
        line: Line from checkpoint list file

    Returns:
        Dictionary with parsed checkpoint information

    Raises:
        ValueError: If line format is invalid
    """
    parts = line.strip().split("|")
    if len(parts) != 5:
        raise ValueError(f"Invalid checkpoint line format: {line}")

    return {
        "checkpoint_path": parts[0],
        "training_config_path": parts[1],
        "subdir": parts[2],
        "run_id": parts[3],
        "step": parts[4],
    }


def load_checkpoint_list(checkpoint_list_file: str) -> list[dict[str, str]]:
    """
    Load and parse checkpoint list file.

    Args:
        checkpoint_list_file: Path to checkpoint list file

    Returns:
        List of checkpoint dictionaries

    Raises:
        ValueError: If file doesn't exist or has invalid format
    """
    if not Path(checkpoint_list_file).exists():
        raise ValueError(f"Checkpoint list file not found: {checkpoint_list_file}")

    checkpoints = []
    with open(checkpoint_list_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            try:
                checkpoint_info = parse_checkpoint_path(line)
                checkpoints.append(checkpoint_info)
            except ValueError as e:
                raise ValueError(f"Error in {checkpoint_list_file} line {line_num}: {e}") from e

    if not checkpoints:
        raise ValueError(f"No valid checkpoints found in {checkpoint_list_file}")

    logger.info(f"Loaded {len(checkpoints)} checkpoints from {checkpoint_list_file}")
    return checkpoints


def check_existing_metrics_for_checkpoint(checkpoint_info: dict[str, str], suffix: str = "") -> bool:
    """
    Check if metrics file already exists for a specific checkpoint.

    Args:
        checkpoint_info: Dictionary with checkpoint information
        suffix: Generation suffix for metrics filename

    Returns:
        True if metrics file exists, False otherwise
    """
    # Extract dataset from checkpoint path
    checkpoint_path = checkpoint_info["checkpoint_path"]

    # Try to determine dataset from checkpoint path
    try:
        # Extract dataset from the path structure: checkpoints/{dataset}/...
        path_parts = Path(checkpoint_path).parts
        checkpoints_idx = next(i for i, part in enumerate(path_parts) if part == "checkpoints")
        dataset = path_parts[checkpoints_idx + 1]
    except (StopIteration, IndexError) as e:
        raise ValueError(f"Cannot determine dataset from checkpoint path: {checkpoint_path}") from e

    # For checkpoint lists, we need to check the specific step, not just any step
    subdir = checkpoint_info["subdir"]
    run_id = checkpoint_info["run_id"]
    step = checkpoint_info["step"]

    # Use shortcut size 2048 (hardcoded for now as in create_single_checkpoint_config)
    shortcut_size = 2048

    # Build the specific path for this checkpoint
    base_path = Path("generation_outputs") / dataset / subdir / run_id / f"step={step}" / f"scut={shortcut_size}"

    if not base_path.exists():
        return False

    # Check seed directories for metrics files
    for seed_dir in base_path.glob("seed_*"):
        if not seed_dir.is_dir():
            continue

        # Determine metrics filename based on suffix
        if suffix:
            metrics_file = seed_dir / f"metrics_{suffix}.json"
        else:
            metrics_file = seed_dir / "metrics.json"

        if metrics_file.exists():
            logger.debug(f"Found existing metrics for checkpoint: {metrics_file}")
            return True

    return False


def create_test_dataloader(gen_cfg: GenerationConfig) -> DataLoader:
    """Create test dataloader from config."""
    logger.info("Creating test dataloader...")

    # Use the effective test data path (auto-determined if needed)
    test_data_path = gen_cfg.effective_test_data_path
    if test_data_path is None:
        raise ValueError("test_data_path is None - cannot create dataloader")

    test_ds = Dataset.load_from_disk(test_data_path)
    logger.info(f"Loaded test dataset from {test_data_path}")

    logger.info(f"Test dataset size: {len(test_ds)}")
    logger.info("Creating TextDataset for test data...")
    test_text_ds = TextDataset(test_ds)
    logger.info("TextDataset created")

    return DataLoader(
        test_text_ds,
        batch_size=gen_cfg.batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=1,  # CPU efficient for generation
        persistent_workers=False,
    )


def create_single_checkpoint_config(gen_cfg: GenerationConfig, checkpoint_info: dict[str, str]) -> GenerationConfig:
    """
    Create a new GenerationConfig for a single checkpoint from the list.

    Args:
        gen_cfg: Original generation config
        checkpoint_info: Dictionary with checkpoint information

    Returns:
        New GenerationConfig configured for the specific checkpoint
    """
    checkpoint_path = checkpoint_info["checkpoint_path"]
    training_config_path = checkpoint_info["training_config_path"]
    subdir = checkpoint_info["subdir"]
    run_id = checkpoint_info["run_id"]
    step = checkpoint_info["step"]

    # Determine dataset from checkpoint path
    try:
        # Extract dataset from the path structure: checkpoints/{dataset}/...
        path_parts = Path(checkpoint_path).parts
        checkpoints_idx = next(i for i, part in enumerate(path_parts) if part == "checkpoints")
        dataset = path_parts[checkpoints_idx + 1]
    except (StopIteration, IndexError) as e:
        raise ValueError(f"Cannot determine dataset from checkpoint path: {checkpoint_path}") from e

    # Use shortcut size 2048 for now
    shortcut_size = 2048

    # Create output folder with step info
    output_folder = f"generation_outputs/{dataset}/{subdir}/{run_id}/step={step}/scut={shortcut_size}"

    # Create new config with updated values
    config_dict = gen_cfg.model_dump()
    config_dict.update(
        {
            "training_config_path": training_config_path,
            "checkpoint_path": checkpoint_path,
            "generation_shortcut_size": shortcut_size,
            "denoising_step_size": shortcut_size,
            "output_folder": output_folder,
        }
    )

    return GenerationConfig(**config_dict)


def run_generation_from_checkpoint_list(gen_cfg: GenerationConfig) -> None:
    """
    Run generation locally from a checkpoint list file.

    This function processes a list of specific checkpoints from a file and runs
    generation locally (not using ExCA job submission).
    """
    logger.info("Running generation locally from checkpoint list...")

    if gen_cfg.checkpoint_list_file is None:
        raise ValueError("checkpoint_list_file is required for list-based generation")

    # Load checkpoints from file
    checkpoints = load_checkpoint_list(gen_cfg.checkpoint_list_file)
    logger.info(f"Processing {len(checkpoints)} checkpoints from list")

    # Filter based on existing metrics if force is False
    if gen_cfg.force_regeneration:
        filtered_checkpoints = checkpoints
        logger.info(f"Force mode enabled - processing all {len(filtered_checkpoints)} checkpoints")
    else:
        filtered_checkpoints = []
        for checkpoint_info in checkpoints:
            if not check_existing_metrics_for_checkpoint(checkpoint_info, gen_cfg.generation_suffix):
                filtered_checkpoints.append(checkpoint_info)
            else:
                logger.debug(
                    f"Skipping {checkpoint_info['subdir']}/{checkpoint_info['run_id']} "
                    f"(step {checkpoint_info['step']}) - metrics already exist"
                )

        logger.info(f"After filtering: {len(filtered_checkpoints)} checkpoints without existing metrics")

    if not filtered_checkpoints:
        logger.info("No checkpoints to process after filtering")
        return

    # Process checkpoints one by one
    successful_generations = []
    failed_generations = []

    for i, checkpoint_info in enumerate(filtered_checkpoints, 1):
        checkpoint_path = checkpoint_info["checkpoint_path"]
        training_config_path = checkpoint_info["training_config_path"]
        subdir = checkpoint_info["subdir"]
        run_id = checkpoint_info["run_id"]
        step = checkpoint_info["step"]

        logger.info(f"=== Processing checkpoint {i}/{len(filtered_checkpoints)}: {subdir}/{run_id} (step {step}) ===")

        try:
            # Verify checkpoint file exists
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            # Verify training config exists
            if not Path(training_config_path).exists():
                raise FileNotFoundError(f"Training config not found: {training_config_path}")

            # Create config for this specific checkpoint
            checkpoint_gen_cfg = create_single_checkpoint_config(gen_cfg, checkpoint_info)

            # Run generation for this checkpoint
            logger.info(f"Starting generation for {subdir}/{run_id} (step {step})")
            run_single_checkpoint_generation(checkpoint_gen_cfg)

            successful_generations.append(f"{subdir}/{run_id} (step {step})")
            logger.info(f"✓ Successfully completed generation for {subdir}/{run_id} (step {step})")

        except Exception as e:
            error_msg = f"✗ Failed generation for {subdir}/{run_id} (step {step}): {e}"
            logger.error(error_msg)
            failed_generations.append(f"{subdir}/{run_id} (step {step}): {str(e)}")
            # Continue with next checkpoint
            continue

    # Print summary
    logger.info("=== GENERATION SUMMARY ===")
    logger.info(f"Total checkpoints processed: {len(filtered_checkpoints)}")
    logger.info(f"Successful: {len(successful_generations)}")
    logger.info(f"Failed: {len(failed_generations)}")

    if successful_generations:
        logger.info("Successful generations:")
        for success in successful_generations:
            logger.info(f"  ✓ {success}")

    if failed_generations:
        logger.info("Failed generations:")
        for failure in failed_generations:
            logger.info(f"  ✗ {failure}")


def run_single_checkpoint_generation(gen_cfg: GenerationConfig) -> None:
    """
    Run generation for a single checkpoint.

    This is essentially the original run_generation_with_evaluation logic
    but assumes all required paths are properly set in the config.
    """
    logger.info("=== GENERATION JOB STARTED ===")
    logger.info(f"Generation config: output_folder={gen_cfg.output_folder}")
    logger.info(f"Checkpoint path: {gen_cfg.checkpoint_path}")

    # Ensure required fields are not None
    if gen_cfg.training_config is None:
        raise ValueError("training_config is required for generation")
    if gen_cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required for generation")

    pl.seed_everything(gen_cfg.seed)

    # Ensure output directory exists before generation
    output_path = Path(gen_cfg.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if gen_cfg.use_ema_weights:
        callbacks.append(get_ema_callback(gen_cfg.training_config, gen_cfg.checkpoint_path))

    save_outputs_callback = SaveTestOutputsCallback(
        save_path=output_path,
        diff_steps=gen_cfg.training_config.model.diffusion_steps,
        shortcut_size=gen_cfg.generation_shortcut_size,
        start_example_idx=1,
    )
    callbacks.append(save_outputs_callback)

    tokenizer = AutoTokenizer.from_pretrained(gen_cfg.training_config.model.tokenizer_config_name)

    criterion = create_criterion(gen_cfg.training_config)
    unit = load_unit_from_checkpoint(
        criterion,
        gen_cfg.checkpoint_path,
        gen_cfg.training_config,
        tokenizer=tokenizer,
        denoising_step_size=gen_cfg.denoising_step_size,
        prediction_shortcut_size=gen_cfg.generation_shortcut_size,
    )

    # Create test dataloader
    test_dataloader = create_test_dataloader(gen_cfg)

    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_test_batches=gen_cfg.limit_test_batches,
    )

    # Run generation
    logger.info("Unit loaded successfully. Creating trainer...")
    logger.info("Trainer created successfully. Starting testing...")
    trainer.test(unit, dataloaders=test_dataloader)

    # Run evaluation if enabled (this creates generation_texts.json and metrics.json)
    if gen_cfg.run_evaluation:
        logger.info("Starting evaluation of generated sequences...")

        try:
            metrics = evaluate_generations(
                output_dir=output_path,
                tokenizer_name=gen_cfg.training_config.model.tokenizer_config_name,
                device=gen_cfg.evaluation_device,
                use_fallback_processing=gen_cfg.use_fallback_processing,
                suffix=gen_cfg.generation_suffix,
            )
            logger.info("Evaluation completed successfully!")
            logger.info(f"Evaluation metrics: {metrics}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    # Run cosine similarity analysis if enabled
    if gen_cfg.run_plot_analysis:
        logger.info("Starting cosine similarity and velocity analysis...")

        try:
            from shortcutfm.analysis.cosine_analysis import run_cosine_similarity_analysis

            run_cosine_similarity_analysis(
                unit=unit,
                test_dataloader=test_dataloader,
                output_dir=output_path,
                generation_shortcut_size=gen_cfg.generation_shortcut_size,
                denoising_step_size=gen_cfg.denoising_step_size,
                analysis_config=gen_cfg.plot_analysis_config.model_dump(),
            )
            logger.info("Cosine similarity analysis completed successfully!")
            logger.info(f"Analysis plots saved to {output_path / 'plots'}")
        except Exception as e:
            logger.error(f"Cosine similarity analysis failed: {e}", exc_info=True)
            # Don't raise here as analysis is optional
            logger.warning("Continuing without cosine similarity analysis...")


def run_generation_with_evaluation(gen_cfg: GenerationConfig) -> None:
    """
    Run the complete generation pipeline with evaluation and analysis.

    This function implements the same logic as GenerationConfig.generate()
    but can be called independently.
    """
    # Check if we should process multiple checkpoints from a list
    if gen_cfg.checkpoint_list_file is not None:
        return run_generation_from_checkpoint_list(gen_cfg)

    # Single checkpoint generation (original logic)
    run_single_checkpoint_generation(gen_cfg)


def scan_checkpoint_directories(dataset: str) -> list[dict[str, str]]:
    """
    Scan checkpoint directories for a dataset and return all valid runs.

    Args:
        dataset: Dataset name (qqp, webnlg, wmt19)

    Returns:
        List of dictionaries with 'subdir' and 'run_id' keys
    """
    checkpoints_path = Path("checkpoints") / dataset
    if not checkpoints_path.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoints_path}")
        return []

    runs = []

    # Scan all subdirectories in the dataset checkpoint folder
    for subdir in checkpoints_path.iterdir():
        if not subdir.is_dir():
            continue

        # Scan for run directories in each subdirectory
        for run_dir in subdir.iterdir():
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name

            # Skip malformed directories starting with "run_<bound"
            if run_name.startswith("run_<bound"):
                logger.debug(f"Skipping malformed directory: {run_name}")
                continue

            # Check if it's a valid run directory (starts with "run_")
            if run_name.startswith("run_") and len(run_name) > 4:
                # Verify checkpoint file exists (either last.ckpt or step-based checkpoint)
                checkpoint_file = run_dir / "last.ckpt"
                step_checkpoints = list(run_dir.glob("epoch=*-step=*.ckpt"))

                if checkpoint_file.exists() or step_checkpoints:
                    runs.append({"subdir": subdir.name, "run_id": run_name})
                    logger.debug(f"Found valid run: {subdir.name}/{run_name}")
                else:
                    logger.debug(f"Skipping run without checkpoint: {subdir.name}/{run_name}")

    return runs


def find_highest_step_checkpoint(run_dir: Path) -> tuple[str, int]:
    """
    Find the checkpoint with the highest step number in a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        Tuple of (checkpoint_filename, step_number)

    Raises:
        ValueError: If no valid checkpoint is found
    """
    # Look for step-based checkpoints first
    step_checkpoints = list(run_dir.glob("epoch=*-step=*.ckpt"))

    if step_checkpoints:
        # Extract step numbers and find the highest one
        max_steps = 0
        best_checkpoint = None

        for ckpt_file in step_checkpoints:
            try:
                # Extract step number from filename like "epoch=281-step=40000.ckpt"
                filename = ckpt_file.name
                step_part = filename.split("-step=")[1].split(".ckpt")[0]
                step_number = int(step_part)

                if step_number > max_steps:
                    max_steps = step_number
                    best_checkpoint = filename

            except (IndexError, ValueError) as e:
                logger.debug(f"Could not parse step number from {ckpt_file.name}: {e}")
                continue

        if best_checkpoint and max_steps > 0:
            logger.debug(f"Found highest step checkpoint: {best_checkpoint} (step {max_steps})")
            return best_checkpoint, max_steps

    # Fall back to last.ckpt if no step-based checkpoints found
    last_ckpt = run_dir / "last.ckpt"
    if last_ckpt.exists():
        logger.debug(f"Using fallback last.ckpt for {run_dir.name}")
        return "last.ckpt", 0  # Use 0 as step number for last.ckpt

    raise ValueError(f"No valid checkpoint found in {run_dir}")


def check_existing_metrics(dataset: str, subdir: str, run_id: str, suffix: str = "") -> bool:
    """
    Check if metrics file already exists for a given run.

    Args:
        dataset: Dataset name
        subdir: Subdirectory name
        run_id: Run ID
        suffix: Generation suffix for metrics filename

    Returns:
        True if metrics file exists, False otherwise
    """
    base_path = Path("generation_outputs") / dataset / subdir / run_id

    if not base_path.exists():
        return False

    # Check for metrics files in step directories first (new structure)
    for step_dir in base_path.glob("step=*"):
        if not step_dir.is_dir():
            continue

        # Check for metrics files in shortcut size directories
        for scut_dir in step_dir.glob("scut=*"):
            if not scut_dir.is_dir():
                continue

            # Check seed directories
            for seed_dir in scut_dir.glob("seed_*"):
                if not seed_dir.is_dir():
                    continue

                # Determine metrics filename based on suffix
                if suffix:
                    metrics_file = seed_dir / f"metrics_{suffix}.json"
                else:
                    metrics_file = seed_dir / "metrics.json"

                if metrics_file.exists():
                    logger.debug(f"Found existing metrics: {metrics_file}")
                    return True

    # Also check old structure (shortcut directories directly under run_id) for backward compatibility
    for scut_dir in base_path.glob("scut=*"):
        if not scut_dir.is_dir():
            continue

        # Check seed directories
        for seed_dir in scut_dir.glob("seed_*"):
            if not seed_dir.is_dir():
                continue

            # Determine metrics filename based on suffix
            if suffix:
                metrics_file = seed_dir / f"metrics_{suffix}.json"
            else:
                metrics_file = seed_dir / "metrics.json"

            if metrics_file.exists():
                logger.debug(f"Found existing metrics: {metrics_file}")
                return True

    return False


def get_dataset_checkpoints(force: bool = False, generation_suffix: str = "") -> dict[str, list[dict[str, str]]]:
    """
    Get checkpoint configurations for all datasets by scanning the filesystem.

    Args:
        force: If True, include runs even if metrics already exist
        generation_suffix: Suffix used for metrics filename

    Returns:
        Dictionary mapping dataset names to lists of checkpoint configurations.
    """
    # Dynamically discover available datasets
    checkpoints_base = Path("checkpoints")
    if not checkpoints_base.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoints_base}")
        return {}

    # Get all dataset directories
    datasets = [d.name for d in checkpoints_base.iterdir() if d.is_dir()]
    logger.info(f"Found dataset directories: {datasets}")

    result = {}

    for dataset in datasets:
        logger.info(f"Scanning checkpoints for dataset: {dataset}")

        # Get all runs for this dataset
        all_runs = scan_checkpoint_directories(dataset)
        logger.info(f"Found {len(all_runs)} total runs for {dataset}")

        # Filter based on existing metrics if force is False
        if force:
            filtered_runs = all_runs
            logger.info(f"Force mode enabled - including all {len(filtered_runs)} runs")
        else:
            filtered_runs = []
            for run in all_runs:
                if not check_existing_metrics(dataset, run["subdir"], run["run_id"], generation_suffix):
                    filtered_runs.append(run)
                else:
                    logger.debug(f"Skipping run with existing metrics: {run['subdir']}/{run['run_id']}")

            logger.info(f"After filtering: {len(filtered_runs)} runs without existing metrics")

        result[dataset] = filtered_runs

    return result


def determine_dataset_from_config_path(training_config_path: str) -> str:
    """
    Determine the dataset name from the training config path.

    Args:
        training_config_path: Path to the training configuration file

    Returns:
        Dataset name

    Raises:
        ValueError: If dataset cannot be determined from the path
    """
    path_lower = training_config_path.lower()

    if "qqp" in path_lower:
        return "qqp"
    elif "webnlg" in path_lower:
        return "webnlg"
    elif "wmt19" in path_lower or "wmt" in path_lower:
        return "wmt19"
    elif "commonsenseconversation" in path_lower:
        return "commonsenseconversation"
    elif "parasci" in path_lower:
        return "parasci"
    elif "paws_wiki" in path_lower or "pawswiki" in path_lower:
        return "pawswiki"
    elif "quasar-t" in path_lower or "quasar" in path_lower:
        return "quasar"
    elif "wiki-alignment" in path_lower or "wiki_alignment" in path_lower or "wiki" in path_lower:
        return "wiki"
    elif "grammar" in path_lower:
        return "grammar"
    else:
        raise ValueError(f"Could not determine dataset from training_config_path: {training_config_path}")


def run_exca_job_submission_from_list(gen_cfg: GenerationConfig) -> None:
    """
    Run generation using exca job submission from a checkpoint list file.

    This function processes a list of specific checkpoints from a file instead of
    automatically discovering them.
    """
    logger.info("Using exca for job submission with checkpoint list...")

    if gen_cfg.checkpoint_list_file is None:
        raise ValueError("checkpoint_list_file is required for list-based job submission")

    # Load checkpoints from file
    checkpoints = load_checkpoint_list(gen_cfg.checkpoint_list_file)
    logger.info(f"Processing {len(checkpoints)} checkpoints from list")

    # Filter based on existing metrics if force is False
    if gen_cfg.force_regeneration:
        filtered_checkpoints = checkpoints
        logger.info(f"Force mode enabled - processing all {len(filtered_checkpoints)} checkpoints")
    else:
        filtered_checkpoints = []
        for checkpoint_info in checkpoints:
            if not check_existing_metrics_for_checkpoint(checkpoint_info, gen_cfg.generation_suffix):
                filtered_checkpoints.append(checkpoint_info)
            else:
                logger.debug(
                    f"Skipping {checkpoint_info['subdir']}/{checkpoint_info['run_id']} "
                    f"(step {checkpoint_info['step']}) - metrics already exist"
                )

        logger.info(f"After filtering: {len(filtered_checkpoints)} checkpoints without existing metrics")

    if not filtered_checkpoints:
        logger.info("No checkpoints to process after filtering")
        return

    # Submit jobs for filtered checkpoints
    with gen_cfg.infra.job_array() as array:
        for checkpoint_info in filtered_checkpoints:
            checkpoint_path = checkpoint_info["checkpoint_path"]
            training_config_path = checkpoint_info["training_config_path"]
            subdir = checkpoint_info["subdir"]
            run_id = checkpoint_info["run_id"]
            step = checkpoint_info["step"]

            # Verify checkpoint file exists
            if not Path(checkpoint_path).exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_path}, skipping")
                continue

            # Verify training config exists
            if not Path(training_config_path).exists():
                logger.warning(f"Training config not found: {training_config_path}, skipping")
                continue

            # Determine dataset from checkpoint path
            try:
                path_parts = Path(checkpoint_path).parts
                checkpoints_idx = next(i for i, part in enumerate(path_parts) if part == "checkpoints")
                dataset = path_parts[checkpoints_idx + 1]
            except (StopIteration, IndexError):
                logger.warning(f"Cannot determine dataset from checkpoint path: {checkpoint_path}, skipping")
                continue

            logger.info(f"Processing {subdir}/{run_id} (step {step})")

            # Test different shortcut sizes for this specific checkpoint
            for shortcut_size in [2048, 1024, 512, 256]:
            # for shortcut_size in [2048]:
                # Create output folder with step info
                output_folder = f"generation_outputs/{dataset}/{subdir}/{run_id}/step={step}/scut={shortcut_size}"

                gen_cfg_copy = gen_cfg.infra.clone_obj(
                    {
                        "checkpoint_list_file": None,  # Clear this to prevent recursive processing
                        "training_config_path": training_config_path,
                        "checkpoint_path": checkpoint_path,
                        "generation_shortcut_size": shortcut_size,
                        "denoising_step_size": shortcut_size,
                        "output_folder": output_folder,
                    }
                )
                array.append(gen_cfg_copy)


def run_exca_job_submission(gen_cfg: GenerationConfig) -> None:
    """
    Run generation using exca job submission with multiple checkpoints.

    This function handles the complex logic for submitting multiple generation
    jobs across different checkpoints and shortcut sizes.
    """
    logger.info("Using exca for job submission...")

    # Route to appropriate function based on configuration
    if gen_cfg.checkpoint_list_file is not None:
        return run_exca_job_submission_from_list(gen_cfg)

    # Original automatic discovery logic
    if gen_cfg.training_config_path is None:
        raise ValueError("training_config_path is required when not using checkpoint_list_file")

    checkpoints = get_dataset_checkpoints(force=gen_cfg.force_regeneration, generation_suffix=gen_cfg.generation_suffix)
    dataset = determine_dataset_from_config_path(gen_cfg.training_config_path)

    logger.info(f"Running generation for dataset: {dataset}")
    logger.info(f"Found {len(checkpoints[dataset])} checkpoints for {dataset}")

    # The actual generation will be handled by the @infra.apply decorated method
    with gen_cfg.infra.job_array() as array:
        for checkpoint in checkpoints[dataset]:
            run_id = checkpoint["run_id"]
            subdir = checkpoint["subdir"]

            # Find the checkpoint with highest step number
            run_dir = Path("checkpoints") / dataset / subdir / run_id
            try:
                checkpoint_filename, step_number = find_highest_step_checkpoint(run_dir)

                # Update paths to use the best checkpoint
                new_training_config_path = f"checkpoints/{dataset}/{subdir}/{run_id}/training_config.yaml"
                new_checkpoint_path = f"checkpoints/{dataset}/{subdir}/{run_id}/{checkpoint_filename}"

                logger.info(f"Using checkpoint {checkpoint_filename} (step {step_number}) for {subdir}/{run_id}")

                for shortcut_size in [2048, 1024, 512, 256]:
                    # for shortcut_size in [2048]:
                    # Create output folder with step info
                    if step_number > 0:
                        new_output_folder = (
                            f"generation_outputs/{dataset}/{subdir}/{run_id}/step={step_number}/scut={shortcut_size}"
                        )
                    else:
                        # For last.ckpt, use "step=final" or similar
                        new_output_folder = (
                            f"generation_outputs/{dataset}/{subdir}/{run_id}/step=final/scut={shortcut_size}"
                        )

                    gen_cfg_copy = gen_cfg.infra.clone_obj(
                        {
                            "training_config_path": new_training_config_path,
                            "checkpoint_path": new_checkpoint_path,
                            "generation_shortcut_size": shortcut_size,
                            "denoising_step_size": shortcut_size,
                            "output_folder": new_output_folder,
                        }
                    )
                    array.append(gen_cfg_copy)

            except ValueError as e:
                logger.warning(f"Skipping {subdir}/{run_id}: {e}")
                continue

#!/usr/bin/env python3
"""
Script to generate checkpoint list files for each run in a dataset directory.
Creates individual files containing all checkpoints for each run.
Supports filtering by experiment type (e.g., scut, baseline, etc.)
"""

import argparse
import re
from pathlib import Path


def discover_available_datasets():
    """Discover all available datasets by scanning the checkpoints directory."""
    checkpoints_base = Path("checkpoints")
    if not checkpoints_base.exists():
        return []

    datasets = []
    for item in checkpoints_base.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Skip directories that look like individual runs (e.g., run_43u12buh)
            if not item.name.startswith("run_"):
                datasets.append(item.name)

    return sorted(datasets)


def extract_step_number(checkpoint_path):
    """Extract step number from checkpoint filename."""
    match = re.search(r"step=(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else 0


def find_all_checkpoints(run_dir):
    """Find all checkpoint files in a run directory and sort by step number."""
    checkpoint_files = []

    for ckpt_file in run_dir.rglob("*.ckpt"):
        if "step=" in str(ckpt_file):
            checkpoint_files.append(ckpt_file)

    # Sort by step number
    checkpoint_files.sort(key=extract_step_number)
    return checkpoint_files


def generate_run_checkpoint_file(experiment_dir, output_dir, dataset="qqp"):
    """Generate checkpoint list file for a single experiment type (e.g., scut, baseline)."""
    experiment_name = experiment_dir.name
    checkpoints = find_all_checkpoints(experiment_dir)

    if not checkpoints:
        print(f"No checkpoints found in {experiment_dir}")
        return None

    # Create output filename
    output_file = output_dir / f"{experiment_name}_checkpoints.txt"

    with open(output_file, "w") as f:
        for ckpt_path in checkpoints:
            # Get relative path from project root
            try:
                rel_ckpt_path = ckpt_path.relative_to(Path.cwd())
            except ValueError:
                # If can't get relative path, use absolute
                rel_ckpt_path = ckpt_path

            # Extract run_id from checkpoint path
            # Path structure: checkpoints/{dataset}/{experiment}/{run_id}/checkpoint.ckpt
            # ckpt_path.parent gives us the run directory
            run_id = ckpt_path.parent.name

            # Look for training config
            config_path = ckpt_path.parent / "training_config.yaml"
            if config_path.exists():
                try:
                    rel_config_path = config_path.relative_to(Path.cwd())
                except ValueError:
                    rel_config_path = config_path
            else:
                rel_config_path = ""

            # Extract step number for the line
            step_num = extract_step_number(ckpt_path)

            # Write in the format: checkpoint_path|training_config_path|subdir|run_id|step
            line = f"{rel_ckpt_path}|{rel_config_path}|{experiment_name}|{run_id}|{step_num}\n"
            f.write(line)

    print(f"Generated {output_file} with {len(checkpoints)} checkpoints")
    return output_file


def scan_dataset_directories(dataset):
    """Scan dataset directory for runs and generate individual checkpoint files."""
    base_dir = Path(f"checkpoints/{dataset}")
    output_dir = Path(f"configs/generation/individual_runs/{dataset}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    all_checkpoints = []

    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        return

    # Find all subdirectories
    filtered_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    filter_desc = ""

    if not filtered_dirs:
        print(f"No directories found in {base_dir}{filter_desc}!")
        return

    print(f"Found {len(filtered_dirs)} directories in {dataset}{filter_desc}")

    for experiment_dir in sorted(filtered_dirs):
        print(f"Processing {experiment_dir.name}...")
        output_file = generate_run_checkpoint_file(experiment_dir, output_dir, dataset)

        if output_file:
            generated_files.append(output_file)

            # Also collect for master list
            checkpoints = find_all_checkpoints(experiment_dir)
            # For each checkpoint, extract both experiment_name and run_id
            experiment_name = experiment_dir.name
            for ckpt_path in checkpoints:
                run_id = ckpt_path.parent.name
                all_checkpoints.append((ckpt_path, run_id, experiment_name))

    # Generate master list file
    if all_checkpoints:
        master_file = output_dir / "all_master_list.txt"

        with open(master_file, "w") as f:
            for ckpt_path, run_id, experiment_name in all_checkpoints:
                try:
                    rel_ckpt_path = ckpt_path.relative_to(Path.cwd())
                except ValueError:
                    rel_ckpt_path = ckpt_path

                config_path = ckpt_path.parent / "training_config.yaml"
                if config_path.exists():
                    try:
                        rel_config_path = config_path.relative_to(Path.cwd())
                    except ValueError:
                        rel_config_path = config_path
                else:
                    rel_config_path = ""

                step_num = extract_step_number(ckpt_path)
                line = f"{rel_ckpt_path}|{rel_config_path}|{experiment_name}|{run_id}|{step_num}\n"
                f.write(line)

        print(f"Generated master list: {master_file} with {len(all_checkpoints)} total checkpoints")
        generated_files.append(master_file)

    print(f"\nGenerated {len(generated_files)} checkpoint list files:")
    for file in generated_files:
        print(f"  {file}")


def main():
    # Discover available datasets
    available_datasets = discover_available_datasets()

    if not available_datasets:
        print("No datasets found in checkpoints directory!")
        return

    parser = argparse.ArgumentParser(description="Generate checkpoint list files for dataset experiments")
    parser.add_argument(
        "dataset",
        nargs="?",  # Make dataset optional
        choices=available_datasets + ["all"],
        help=f"Dataset to process. Available: {', '.join(available_datasets)}, or 'all' for all datasets",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets and exit")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        print("Available datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset}")
        return

    # If no dataset specified, show help and available datasets
    if not args.dataset:
        print("Available datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset}")
        print("\nUsage:")
        print(f"  python {Path(__file__).name} <dataset>")
        print(f"  python {Path(__file__).name} all")
        return

    # Process datasets
    if args.dataset == "all":
        print(f"Processing all {len(available_datasets)} datasets...")
        for dataset in available_datasets:
            print(f"\n{'=' * 50}")
            print(f"Processing dataset: {dataset}")
            print(f"{'=' * 50}")
            scan_dataset_directories(dataset)
    else:
        scan_dataset_directories(args.dataset)


if __name__ == "__main__":
    main()

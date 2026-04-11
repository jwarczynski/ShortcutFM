#!/usr/bin/env python3
"""
Script to generate separate checkpoint list files for each "scut" run in QQP directory.

This script scans the checkpoints/qqp directory for all subdirectories containing "scut"
and generates individual checkpoint list files for each run with ALL available checkpoints
in the required format.
"""

import re
from pathlib import Path


def extract_step_number(checkpoint_filename: str) -> int:
    """Extract step number from checkpoint filename."""
    # Pattern for epoch=X-step=Y.ckpt
    step_match = re.search(r'step=(\d+)\.ckpt$', checkpoint_filename)
    if step_match:
        return int(step_match.group(1))
    
    # For last.ckpt, return a very high number to sort it last
    if checkpoint_filename == "last.ckpt":
        return 999999
    
    return 0


def find_all_checkpoints(run_dir: Path) -> list[tuple[str, int]]:
    """
    Find all checkpoint files in a run directory.
    
    Returns:
        List of tuples (checkpoint_filename, step_number) sorted by step number
    """
    checkpoints = []
    
    # Find all .ckpt files
    for ckpt_file in run_dir.glob("*.ckpt"):
        step_number = extract_step_number(ckpt_file.name)
        checkpoints.append((ckpt_file.name, step_number))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    
    return checkpoints


def generate_run_checkpoint_file(scut_dir: Path, run_dir: Path, output_dir: Path) -> str:
    """
    Generate checkpoint list file for a specific run.
    
    Args:
        scut_dir: The scut experiment directory
        run_dir: The specific run directory
        output_dir: Directory to save the checkpoint list file
        
    Returns:
        Path to the generated file
    """
    subdir_name = scut_dir.name
    run_id = run_dir.name
    
    # Check if training_config.yaml exists
    training_config_path = run_dir / "training_config.yaml"
    if not training_config_path.exists():
        print(f"    Warning: training_config.yaml not found in {run_dir}")
        return None
    
    # Find all checkpoints in this run
    checkpoints = find_all_checkpoints(run_dir)
    
    if not checkpoints:
        print(f"    Warning: No checkpoints found in {run_dir}")
        return None
    
    print(f"  Processing {run_id} with {len(checkpoints)} checkpoints...")
    
    # Create output filename
    output_filename = f"qqp_{subdir_name}_{run_id}_checkpoints.txt"
    output_file = output_dir / output_filename
    
    # Write the checkpoint list file
    with open(output_file, 'w') as f:
        f.write(f"# QQP Shortcut Checkpoints - {subdir_name}/{run_id}\n")
        f.write("# Format: checkpoint_path|training_config_path|subdir|run_id|step\n")
        f.write(f"# Generated automatically by generate_scut_checkpoint_list.py\n\n")
        
        f.write(f"# {subdir_name}/{run_id}\n")
        
        for checkpoint_filename, step_number in checkpoints:
            checkpoint_path = f"checkpoints/qqp/{subdir_name}/{run_id}/{checkpoint_filename}"
            training_config_path_str = f"checkpoints/qqp/{subdir_name}/{run_id}/training_config.yaml"
            step_display = str(step_number if step_number != 999999 else "final")
            
            f.write(f"# {checkpoint_filename} - {step_display} steps\n")
            f.write(f"{checkpoint_path}|{training_config_path_str}|{subdir_name}|{run_id}|{step_display}\n")
            f.write("\n")
    
    print(f"    Generated: {output_file}")
    return str(output_file)


def scan_scut_directories_per_run() -> list[str]:
    """
    Scan checkpoints/qqp for all directories containing 'scut' and generate 
    separate checkpoint files for each run.
    
    Returns:
        List of generated checkpoint file paths
    """
    base_path = Path("checkpoints/qqp")
    if not base_path.exists():
        raise ValueError(f"QQP checkpoints directory not found: {base_path}")
    
    output_dir = Path("configs/generation/individual_runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Find all subdirectories containing 'scut'
    scut_dirs = [d for d in base_path.iterdir() if d.is_dir() and 'scut' in d.name.lower()]
    
    print(f"Found {len(scut_dirs)} directories containing 'scut':")
    for scut_dir in scut_dirs:
        print(f"  - {scut_dir.name}")
    
    for scut_dir in scut_dirs:
        print(f"\nScanning {scut_dir.name}...")
        
        # Find all run directories
        run_dirs = [d for d in scut_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        for run_dir in run_dirs:
            # Skip malformed directories
            if "bound method" in run_dir.name or run_dir.name.startswith("run_<"):
                print(f"  Skipping malformed directory: {run_dir.name}")
                continue
            
            generated_file = generate_run_checkpoint_file(scut_dir, run_dir, output_dir)
            if generated_file:
                generated_files.append(generated_file)
    
    return generated_files


def generate_master_config_file(generated_files: list[str], output_dir: Path):
    """Generate a master configuration file that lists all individual run files."""
    master_config = output_dir / "qqp_scut_master_list.txt"

    with open(master_config, 'w') as f:
        f.write("# Master list of all generated QQP scut checkpoint files\n")
        f.write("# Each file contains checkpoints for a specific run\n")
        f.write("# You can use any of these files with the generation pipeline\n\n")

        for file_path in sorted(generated_files):
            # Convert to relative path from project root
            abs_path = Path(file_path).resolve()
            try:
                relative_path = abs_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = Path(file_path)
            f.write(f"{relative_path}\n")

    print(f"\nMaster list generated: {master_config}")
def generate_checkpoint_list_files():
    """Generate individual checkpoint list files for all scut experiments."""
    
    print("Scanning for 'scut' directories in checkpoints/qqp...")
    generated_files = scan_scut_directories_per_run()
    
    if not generated_files:
        print("No checkpoint files generated!")
        return
    
    output_dir = Path("configs/generation/individual_runs")
    
    print(f"\nGenerated {len(generated_files)} checkpoint list files:")
    for file_path in generated_files:
        # Convert to absolute path first, then to relative
        abs_path = Path(file_path).resolve()
        try:
            relative_path = abs_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = abs_path
        print(f"  {relative_path}")
    
    # Generate master list
    generate_master_config_file(generated_files, output_dir)
    
    print(f"\nAll files saved to: {output_dir}")
    print("\nTo use any of these files, create a generation config with:")
    print("  checkpoint_list_file: \"configs/generation/individual_runs/qqp_<experiment>_<run>_checkpoints.txt\"")


if __name__ == "__main__":
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    generate_checkpoint_list_files()

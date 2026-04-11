#!/usr/bin/env python3
"""
Test script to validate the new checkpoint metrics checking logic.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shortcutfm.decoding.generation_runner import (
    check_existing_metrics_for_specific_config,
    load_checkpoint_list
)

def test_metrics_checking():
    """Test the new metrics checking logic."""
    
    # Test the specific config checking
    print("Testing check_existing_metrics_for_specific_config...")
    
    # This should return False since no metrics exist yet
    result = check_existing_metrics_for_specific_config(
        dataset="qqp",
        subdir="baseline",
        run_id="run_test",
        step="1000",
        shortcut_size=256,
        suffix=""
    )
    print(f"Non-existent metrics check: {result} (should be False)")
    
    # Test checkpoint list loading
    print("\nTesting checkpoint list loading...")
    try:
        checkpoint_file = "configs/generation/individual_runs/qqp/baseline_checkpoints.txt"
        if Path(checkpoint_file).exists():
            checkpoints = load_checkpoint_list(checkpoint_file)
            print(f"Loaded {len(checkpoints)} checkpoints from {checkpoint_file}")
            
            # Show first few checkpoints
            for i, checkpoint in enumerate(checkpoints[:3]):
                print(f"  Checkpoint {i+1}: {checkpoint}")
        else:
            print(f"Checkpoint file not found: {checkpoint_file}")
    except Exception as e:
        print(f"Error loading checkpoint list: {e}")

if __name__ == "__main__":
    test_metrics_checking()

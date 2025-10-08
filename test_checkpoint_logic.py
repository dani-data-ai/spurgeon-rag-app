"""
Test script to verify the refactored checkpointing logic in build_graph.py
"""

import os
from pathlib import Path

# Test configuration
CHECKPOINT_INTERVAL = 5

def simulate_checkpoint_behavior(num_files):
    """Simulate how checkpoints will be saved during processing."""
    print(f"Simulating processing of {num_files} files")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} files")
    print("=" * 80)

    files_completed_this_run = 0

    for file_num in range(1, num_files + 1):
        files_completed_this_run += 1

        # TIER 1: Always save processed files list
        print(f"File {file_num}: Saved to processed_files list (total: {files_completed_this_run})")

        # TIER 2: Periodically save graph checkpoint
        if files_completed_this_run % CHECKPOINT_INTERVAL == 0:
            print(f"  --> Graph checkpoint saved (every {CHECKPOINT_INTERVAL} files)")

    # FINAL SAVE
    if files_completed_this_run > 0:
        print(f"\nFinal checkpoint saved after processing {files_completed_this_run} files")

    print("=" * 80)
    print("\nSummary:")
    print(f"  Total files processed: {files_completed_this_run}")
    print(f"  Graph checkpoints saved: {files_completed_this_run // CHECKPOINT_INTERVAL + 1}")
    print(f"    - Periodic saves: {files_completed_this_run // CHECKPOINT_INTERVAL}")
    print(f"    - Final save: 1")

if __name__ == "__main__":
    print("Testing checkpoint behavior with different file counts:\n")

    # Test with different file counts
    test_cases = [3, 5, 7, 10, 12, 15]

    for num_files in test_cases:
        simulate_checkpoint_behavior(num_files)
        print("\n")

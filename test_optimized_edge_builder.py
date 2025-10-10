"""
Test script for optimized edge builder.

Tests individual components before running on real data.
"""

import sys
import psutil
import hashlib
from pathlib import Path

# Import components from the optimized script
from add_edges_memory_optimized import (
    MemoryMonitor, ProgressTracker, TelemetryLogger,
    stable_hash_item, EdgeDatabase
)


def test_memory_monitor():
    """Test byte-based memory monitoring."""
    print("Testing MemoryMonitor...")

    monitor = MemoryMonitor(max_ram_gb=19.0, headroom_gb=1.0)

    # Check basic functionality
    assert monitor.max_ram_bytes == int(19.0 * 1024**3), "Max RAM bytes incorrect"
    assert monitor.headroom_bytes == int(1.0 * 1024**3), "Headroom bytes incorrect"

    # Test current usage
    current_gb = monitor.gb_in_use()
    current_bytes = monitor.bytes_in_use()

    print(f"  Current usage: {current_gb:.2f}GB ({current_bytes:,} bytes)")
    assert current_gb > 0, "Memory usage should be > 0"
    assert current_bytes > 0, "Memory bytes should be > 0"

    # Test should_pause (should not pause at normal levels)
    should_pause = monitor.should_pause()
    print(f"  Should pause: {should_pause}")

    # Test batch size estimation
    sample_data = [
        ('node1', 'node2', 'entity', 'shared_item_1'),
        ('node3', 'node4', 'keyword', 'shared_item_2'),
    ] * 50

    batch_size = monitor.estimate_batch_size(sample_data, target_mb=100)
    print(f"  Estimated batch size for 100MB: {batch_size:,}")
    assert 1000 <= batch_size <= 50000, "Batch size should be in valid range"

    print("[PASS] MemoryMonitor tests passed\n")


def test_progress_tracker():
    """Test progress tracking and resume capability."""
    print("Testing ProgressTracker...")

    test_file = Path("test_progress.json")

    # Clean up if exists
    if test_file.exists():
        test_file.unlink()

    # Create tracker
    tracker = ProgressTracker(test_file)

    # Test phase setting
    tracker.set_phase('edgegen')
    assert tracker.data['phase'] == 'edgegen', "Phase not set correctly"

    # Test item completion marking
    tracker.mark_item_completed('entity', 'test_item_1')
    assert tracker.is_item_completed('entity', 'test_item_1'), "Item should be marked completed"
    assert not tracker.is_item_completed('entity', 'test_item_2'), "Item should not be completed"

    # Test progress update
    tracker.update_progress('keyword', 'test_item_2', 100, 200)
    assert tracker.data['current_item'] == 'keyword|test_item_2', "Current item not set"
    assert tracker.data['last_i'] == 100, "Last i not set"
    assert tracker.data['last_j'] == 200, "Last j not set"

    # Test resume point
    i, j = tracker.get_resume_point('keyword', 'test_item_2')
    assert i == 100 and j == 200, f"Resume point incorrect: {i}, {j}"

    # Test persistence
    tracker2 = ProgressTracker(test_file)
    assert tracker2.is_item_completed('entity', 'test_item_1'), "Persistence failed"

    # Cleanup
    test_file.unlink()

    print("[PASS] ProgressTracker tests passed\n")


def test_stable_hash():
    """Test stable hash partitioning."""
    print("Testing stable_hash_item...")

    num_workers = 7

    # Test that same item always goes to same worker
    item1_worker = stable_hash_item('entity', 'test_item', num_workers)
    assert 0 <= item1_worker < num_workers, "Worker ID out of range"

    # Verify stability (same input → same output)
    for _ in range(10):
        assert stable_hash_item('entity', 'test_item', num_workers) == item1_worker, \
            "Hash should be stable"

    # Test distribution (different items → different workers)
    worker_counts = [0] * num_workers
    for i in range(1000):
        worker_id = stable_hash_item('entity', f'item_{i}', num_workers)
        worker_counts[worker_id] += 1

    # Each worker should get roughly 1000/7 ≈ 143 items (allow 50% variance)
    expected = 1000 / num_workers
    for count in worker_counts:
        assert expected * 0.5 < count < expected * 1.5, \
            f"Distribution imbalanced: {worker_counts}"

    print(f"  Distribution across {num_workers} workers: {worker_counts}")
    print("[PASS] stable_hash_item tests passed\n")


def test_telemetry():
    """Test telemetry logging."""
    print("Testing TelemetryLogger...")

    test_file = Path("test_telemetry.csv")

    # Clean up if exists
    if test_file.exists():
        test_file.unlink()

    # Create logger
    telemetry = TelemetryLogger(test_file)

    # Log some batches
    telemetry.log_batch('entity', 'test_item', 1, 1000, 123.45, 5.5)
    telemetry.log_batch('keyword', 'another_item', 2, 2000, 234.56, 6.7, vram_gb=1.2)

    # Verify file exists and has content
    assert test_file.exists(), "Telemetry file should exist"

    with open(test_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3, f"Expected 3 lines (header + 2 rows), got {len(lines)}"
        assert 'timestamp' in lines[0], "Header should contain 'timestamp'"
        assert 'entity' in lines[1], "First row should contain 'entity'"
        assert 'keyword' in lines[2], "Second row should contain 'keyword'"

    # Cleanup
    test_file.unlink()

    print("[PASS] TelemetryLogger tests passed\n")


def test_edge_database():
    """Test EdgeDatabase basic operations."""
    print("Testing EdgeDatabase...")

    test_db = Path("test_edges.db")

    # Clean up if exists
    if test_db.exists():
        test_db.unlink()

    # Create database
    db = EdgeDatabase(test_db, delete_on_close=True)

    # Add some edges
    edges = [
        ('node1', 'node2', 'entity', 'item1'),
        ('node1', 'node3', 'entity', 'item1'),
        ('node2', 'node3', 'keyword', 'item2'),
    ]

    db.add_edge_batch(edges)

    # Check count
    count = db.get_edge_count()
    assert count == 3, f"Expected 3 edges, got {count}"

    # Test aggregation
    aggregated = list(db.get_aggregated_edges(min_overlap=1, max_shared=5))
    print(f"  Aggregated edges: {len(aggregated)}")

    # Close without deletion (preserve for inspection)
    db.close(success=False)
    assert test_db.exists(), "DB should exist after close(success=False)"

    # Reopen and delete
    db2 = EdgeDatabase(test_db, delete_on_close=True)
    db2.close(success=True)
    assert not test_db.exists(), "DB should be deleted after close(success=True)"

    print("[PASS] EdgeDatabase tests passed\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING COMPONENT TESTS")
    print("=" * 70)
    print()

    try:
        test_memory_monitor()
        test_progress_tracker()
        test_stable_hash()
        test_telemetry()
        test_edge_database()

        print("=" * 70)
        print("[PASS] ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The optimized edge builder components are working correctly.")
        print("Ready to run on real checkpoint files.")
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

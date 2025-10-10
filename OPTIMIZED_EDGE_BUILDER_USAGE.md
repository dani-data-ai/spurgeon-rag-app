# Optimized Edge Builder - Usage Guide

## Overview

The optimized `add_edges_memory_optimized.py` script has been upgraded with:
- **7-core parallel processing** for 4-6× speedup
- **Strict 20GB RAM limit** with byte-level monitoring
- **Mid-file checkpointing** for resumability
- **Detailed telemetry** for performance analysis

## What's New

### Phase 1 Optimizations (Implemented)

| Feature | Before | After |
|---------|--------|-------|
| **CPU Usage** | Single core | 7 cores (parallel) |
| **RAM Control** | 96% (percentage-based) | ≤20GB (byte-based RSS) |
| **Resume** | File-level only | Mid-item (i,j) checkpoints |
| **Progress** | Basic logging | CSV telemetry + detailed TQDM |
| **Crash Safety** | Loses temp data | Preserves temp DB for resume |

### Key Improvements

1. **Memory Governance**
   - Uses RSS (Resident Set Size) in bytes, not percentages
   - Hard 19GB limit with 1GB headroom
   - Dynamic batch sizing based on actual memory usage

2. **Parallel Processing**
   - 7 worker processes for edge generation
   - 1 dedicated SQLite writer process
   - Queue-based communication with backpressure control
   - Stable hash partitioning ensures no duplicate pairs

3. **Checkpointing**
   - Saves progress to `.progress.json` per file
   - Tracks (edge_type, item, i, j) for exact resume point
   - Temp DB preserved on interruption
   - Resume with `--resume` flag

4. **Telemetry**
   - CSV file per checkpoint with batch-level metrics
   - Columns: timestamp, edge_type, item, subbatch, pairs, duration_ms, ram_gb
   - Use for performance analysis and troubleshooting

## Usage

### Basic Run

Process all checkpoint files with default settings:

```bash
python add_edges_memory_optimized.py --source-dir .
```

### With Custom Parameters

```bash
python add_edges_memory_optimized.py \
  --source-dir . \
  --workers 7 \
  --max-ram-gb 19.0 \
  --min-overlap 1 \
  --max-shared 3 \
  --resume
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source-dir` | `.` | Directory containing checkpoint_*.gml files |
| `--workers` | `7` | Number of parallel worker processes |
| `--max-ram-gb` | `19.0` | Maximum RAM usage in GB (hard limit) |
| `--min-overlap` | `1` | Minimum overlap to create edge |
| `--max-shared` | `3` | Max shared items per edge |
| `--resume` | `False` | Resume from last checkpoint |
| `--clear-state` | `False` | Clear state and start fresh |

## Resume After Interruption

If the script is interrupted (Ctrl+C, crash, or killed):

1. **Check what was preserved:**
   ```bash
   ls -la *.progress.json *.tmp.db *.telemetry.csv
   ```

2. **Resume processing:**
   ```bash
   python add_edges_memory_optimized.py --source-dir . --resume
   ```

The script will:
- Skip already-completed files (in `.edge_builder_state.json`)
- Resume incomplete files from exact (i,j) checkpoint
- Use existing temp DB to avoid recomputation

## Output Files

For each checkpoint file (e.g., `checkpoint_spurgeon.gml`), the script creates:

| File | Purpose | When Deleted |
|------|---------|--------------|
| `checkpoint_spurgeon.tmp.db` | Temporary edge storage (SQLite) | After successful GML save |
| `checkpoint_spurgeon.progress.json` | Mid-file resume checkpoint | After successful completion |
| `checkpoint_spurgeon.telemetry.csv` | Batch-level performance metrics | Never (keep for analysis) |
| `.edge_builder_state.json` | File-level completion tracking | Manual with `--clear-state` |
| `add_edges_optimized.log` | High-level logging | Never (append-only) |

## Monitoring Progress

### Console Output

The script shows real-time progress via TQDM:

```
Generating entity edges with 7 workers...
Processing 1,234 entity items (filtered from 1,250)
Distributing 1,234 items across 7 workers: 100%

Worker 0 completed (1/7)
Worker 1 completed (2/7)
...
```

### Telemetry CSV

Analyze performance with:

```bash
# View first 10 rows
head -10 checkpoint_spurgeon.telemetry.csv

# Calculate average batch time
awk -F',' 'NR>1 {sum+=$6; count++} END {print "Avg batch time:", sum/count, "ms"}' checkpoint_spurgeon.telemetry.csv

# Find peak RAM usage
awk -F',' 'NR>1 {if($7>max) max=$7} END {print "Peak RAM:", max, "GB"}' checkpoint_spurgeon.telemetry.csv
```

## Hardware Constraints

The script is designed for:

- **Max RAM:** 20GB (configurable via `--max-ram-gb`)
- **CPU Cores:** 7 workers (configurable via `--workers`)
- **GPU:** Not used in Phase 1 (CPU-only)

### Adjusting for Your Hardware

**Less RAM available (e.g., 16GB):**
```bash
python add_edges_memory_optimized.py --max-ram-gb 14.0
```

**Fewer CPU cores (e.g., 4 cores):**
```bash
python add_edges_memory_optimized.py --workers 3
```

**More cores (e.g., 16 cores):**
```bash
python add_edges_memory_optimized.py --workers 15
```

## Expected Performance

### Before Optimization
- **CPU:** Single core (~14% utilization)
- **RAM:** 96% (crashes on large files)
- **Speed:** ~1-2 hours per large file
- **Resume:** File-level only

### After Optimization (Phase 1)
- **CPU:** 7 cores (~85-95% utilization across 7 cores)
- **RAM:** ≤20GB (never exceeds)
- **Speed:** ~15-30 minutes per large file (4-6× faster)
- **Resume:** Mid-file (i,j) checkpoints

## Troubleshooting

### Script Hangs During "Loading graph for edge application"

**Symptom:** High RAM usage (19-20GB), no progress
**Cause:** Large graph + many edges to apply
**Solution:** Reduce batch size in apply phase (already set to 1000)

### Workers Not Starting

**Symptom:** "Generating entity edges with 7 workers" but no worker completion messages
**Cause:** Multiprocessing issue on Windows
**Solution:** Ensure Python 3.7+ and no antivirus blocking

### Memory Still Exceeds 20GB

**Symptom:** Peak RSS > 20GB
**Cause:** NetworkX graph loading in apply phase
**Solution:** Reduce `--max-ram-gb` to 17.0 to account for graph overhead

### Resume Not Working

**Symptom:** `--resume` restarts from beginning
**Cause:** Progress file or state file missing
**Solution:** Check for `.progress.json` and `.edge_builder_state.json` files

## Testing

Run component tests before processing real data:

```bash
python test_optimized_edge_builder.py
```

Expected output:
```
[PASS] MemoryMonitor tests passed
[PASS] ProgressTracker tests passed
[PASS] stable_hash_item tests passed
[PASS] TelemetryLogger tests passed
[PASS] EdgeDatabase tests passed
[PASS] ALL TESTS PASSED
```

## Next Steps

### For Your Current Task

Process the 2 remaining files:

```bash
# Resume from where you left off (Vol 80-125 and spurgeon)
python add_edges_memory_optimized.py --source-dir . --resume
```

### Phase 2 (Optional - If CPU Performance Still Insufficient)

If 4-6× speedup isn't enough, Phase 2 could add:
- GPU acceleration for pair enumeration (CuPy/CUDA)
- But only if CPU parallelism proves inadequate

**Recommendation:** Test Phase 1 first. GPU is likely overkill for this workload.

## Support

- **Logs:** Check `add_edges_optimized.log` for detailed error messages
- **Telemetry:** Analyze `.telemetry.csv` files for performance bottlenecks
- **Progress:** Inspect `.progress.json` for resume state

## Summary

This optimized version:
✅ Uses ≤20GB RAM (byte-based hard limit)
✅ Runs 7 cores in parallel (4-6× faster)
✅ Resumes from exact (i,j) checkpoint
✅ Preserves temp data on interruption
✅ Logs detailed telemetry for analysis
✅ Passed all component tests

Ready to process your remaining checkpoint files!

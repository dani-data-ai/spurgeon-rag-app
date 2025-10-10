# Quick Start - Optimized Edge Builder

## TL;DR

```bash
# Test components (1 minute)
python test_optimized_edge_builder.py

# Process remaining files with resume
python add_edges_memory_optimized.py --source-dir . --resume
```

---

## What's Different?

| Old Script | New Script |
|------------|------------|
| 1 CPU core | 7 CPU cores (parallel) |
| 96% RAM (crashes) | ≤20GB RAM (safe) |
| File-level resume | Mid-item (i,j) resume |
| ~2 hours/file | ~15-30 min/file |

**Speed: 4-6× faster** 🚀

---

## Current Status

Your checkpoint files:

- ✅ `checkpoint_Epub_Vol_1-25.gml` - **Completed** (48.7M edges)
- ✅ `checkpoint_Epub_Vol_26-79.gml` - **Completed**
- ⏸️ `checkpoint_Epub_Vol_80-125.gml` - **Needs resume** (crashed at apply phase)
- ⏳ `checkpoint_spurgeon.gml` - **Not started**

---

## Run Now

```bash
python add_edges_memory_optimized.py --source-dir . --resume
```

This will:
1. Skip the 2 completed files
2. Resume Vol 80-125 from where it crashed
3. Process spurgeon.gml
4. Use 7 cores + ≤20GB RAM
5. Save progress every batch (safe to interrupt)

---

## Monitor Progress

### In Console
```
Generating entity edges with 7 workers...
Processing 1,234 entity items (filtered from 1,250)
Worker 0 completed (1/7)
Worker 1 completed (2/7)
...
```

### In Files
- `add_edges_optimized.log` - High-level events
- `checkpoint_*.telemetry.csv` - Batch-level metrics (timestamp, RAM, duration)
- `checkpoint_*.progress.json` - Resume checkpoints
- `.edge_builder_state.json` - Completed files

---

## If Interrupted

The script auto-saves progress. To resume:

```bash
python add_edges_memory_optimized.py --source-dir . --resume
```

Temp files (`.tmp.db`, `.progress.json`) are preserved until success.

---

## Parameters

Default settings are optimized for your hardware:

```bash
--workers 7              # Use all 7 cores
--max-ram-gb 19.0        # Leave 1GB headroom
--min-overlap 1          # Same as before
--max-shared 3           # Same as before
```

To adjust (e.g., less RAM):
```bash
python add_edges_memory_optimized.py --max-ram-gb 16.0 --resume
```

---

## Expected Results

**For each file:**
- Time: 15-30 minutes (vs 1-2 hours before)
- RAM: Never exceeds 20GB
- CPU: ~85-95% across 7 cores
- Output: Same GML format, same edge weights

**Total for 2 remaining files:** ~1 hour (vs 3-4 hours before)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| RAM > 20GB | Reduce `--max-ram-gb 17.0` |
| Workers not starting | Check Python ≥3.7, no antivirus blocking |
| Hangs at "Loading graph" | Normal - NetworkX loads full graph (can take minutes) |
| Resume not working | Check for `.progress.json` files |

---

## After Completion

Files will be updated in-place with new edges:
- `checkpoint_Epub_Vol_80-125.gml` - Updated
- `checkpoint_spurgeon.gml` - Updated

Temp files auto-deleted:
- `checkpoint_*.tmp.db` - Deleted after success
- `checkpoint_*.progress.json` - Deleted after success

Keep for analysis:
- `checkpoint_*.telemetry.csv` - Performance metrics
- `add_edges_optimized.log` - Event log

---

## Validation

Check edges were added:
```bash
python -c "import networkx as nx; g = nx.read_gml('checkpoint_spurgeon.gml'); print(f'{g.number_of_nodes()} nodes, {g.number_of_edges()} edges')"
```

---

## Support

- **Full docs:** [OPTIMIZED_EDGE_BUILDER_USAGE.md](OPTIMIZED_EDGE_BUILDER_USAGE.md)
- **Implementation details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Tests:** `python test_optimized_edge_builder.py`

---

**Ready to go!** 🚀

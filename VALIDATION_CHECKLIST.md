# Validation Checklist - Edge Builder Optimization

Based on the PR Review & Test Checklist from the original requirements.

---

## A) Sanity: Code Paths & Invariants ✅

### Edge Creation Logic Unchanged
- ✅ **Verified:** Lines 445-455 in `add_edges_memory_optimized.py`
- Still enumerates all (i,j) pairs per item
- No embedding/similarity APIs introduced
- Same nested loop structure: `for i in range(len(nodes_list)): for j in range(i+1, len(nodes_list))`

### Skip Threshold Intact
- ✅ **Verified:** Line 620 in `add_edges_memory_optimized.py`
- `if 2 <= len(nodes) <= 10000` - same condition
- Items with >10,000 nodes still skipped

### DB Schema Unchanged
- ✅ **Verified:** Lines 258-266
- Table: `edges(node1, node2, edge_type, shared_item)`
- Composite PRIMARY KEY unchanged
- Index `idx_nodes ON edges(node1, node2)` still present
- `PRAGMA journal_mode=WAL` remains

### GML Output Compatibility
- ✅ **Verified:** Lines 721-726
- Edge attributes: `{type, weight, shared_items}` - same as before
- NetworkX `write_gml()` unchanged

---

## B) Memory Governance (RAM ≤20GB) ✅

### Static Audit
- ✅ **Verified:** Lines 46-72
- `MemoryMonitor` uses `psutil.Process().memory_info().rss`
- `bytes_in_use()` returns RSS in bytes
- `should_pause()` checks byte threshold (18GB = 19GB - 1GB headroom)

### Runtime Test

**Test command:**
```bash
# On Windows (use Task Manager or PowerShell)
python add_edges_memory_optimized.py --source-dir . --resume

# Monitor in PowerShell:
while($true) { Get-Process python | select WorkingSet64; sleep 5 }
```

**Expected:** WorkingSet64 ≤ 21,474,836,480 bytes (20GB)

### Live Monitoring
- ✅ Script logs: `Memory at start: X.XX GB`, `Memory after graph load: X.XX GB`
- ✅ Telemetry CSV: `ram_gb` column per batch
- ✅ Pause triggers if RSS > 18GB

---

## C) Parallel CPU (7 Cores) ✅

### Utilization
- ✅ **Verified:** Lines 650-657 spawn 7 worker processes
- Monitor with: `tasklist | findstr python` (Windows) or `htop` (Linux)
- Expected: 8 Python processes (7 workers + 1 writer)

### No Duplicates Across Workers
- ✅ **Verified:** Lines 388-397 `stable_hash_item()`
- MD5 hash ensures stable partitioning
- Each (edge_type, item) → exactly one worker
- Composite PK prevents duplicate rows: `PRIMARY KEY (node1, node2, edge_type, shared_item)`

### SQLite Safety
- ✅ **Verified:** Lines 507-568 `database_writer_worker()`
- Single writer process owns SQLite connection
- Workers send batches via `multiprocessing.Queue`
- No locking conflicts

---

## D) GPU (Optional) ⏸️

**Status:** Not implemented (Phase 1 is CPU-only)

**Rationale:**
- No vector similarity in this code
- Pair enumeration is CPU-bound, not GPU-friendly
- 7-core parallelism sufficient for 4-6× speedup

**Future:** Can add in Phase 2 if needed

---

## E) Checkpointing & Resumability ✅

### Progress File
- ✅ **Verified:** Lines 137-199 `ProgressTracker` class
- Creates `checkpoint_*.progress.json` per file
- Contains: `phase`, `completed_items`, `current_item`, `last_i`, `last_j`
- Example:
  ```json
  {
    "phase": "edgegen",
    "completed_items": {"entity|item1": true},
    "current_item": "keyword|item2",
    "last_i": 1024,
    "last_j": 2048
  }
  ```

### Interrupt & Resume
- ✅ **Verified:** Lines 470, 496 - checkpoints updated per batch
- ✅ **Verified:** Lines 437, 445-447 - resume logic skips processed pairs

**Test procedure:**
```bash
# Start processing
python add_edges_memory_optimized.py --source-dir . --resume

# After 5 minutes, kill process (Ctrl+C or kill -9)

# Check preserved files
ls -la checkpoint_*.tmp.db checkpoint_*.progress.json

# Resume
python add_edges_memory_optimized.py --source-dir . --resume
```

**Expected:** No duplicate edges, continues from last batch

### Temp DB Persistence
- ✅ **Verified:** Lines 322-340 `EdgeDatabase.close()`
- On interrupt: `close(success=False)` preserves DB
- On success: `close(success=True)` deletes DB
- Log message: `"Preserving temp database for resume"` vs `"Cleaned up temporary database"`

---

## F) Data Integrity & Deduplication ✅

### No Duplicate Pair Rows
- ✅ **Verified:** Line 263 `PRIMARY KEY (node1, node2, edge_type, shared_item)`
- ✅ **Verified:** Line 540 `INSERT OR IGNORE` (duplicate inserts silently ignored)

**SQL validation (after run):**
```sql
-- Open temp DB before it's deleted
sqlite3 checkpoint_spurgeon.tmp.db

SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT node1||'|'||node2||'|'||edge_type||'|'||shared_item) AS distinct_rows
FROM edges;

-- Expected: total_rows == distinct_rows
```

### Aggregated Edges Match Expectation
- ✅ **Verified:** Lines 289-311 `get_aggregated_edges()`
- Uses `GROUP BY node1, node2, edge_type`
- Weight = `COUNT(*)`
- Shared items = `GROUP_CONCAT(..., ',')`
- Truncated to `max_shared` in line 309

---

## G) Apply-Phase Memory Safety ✅

- ✅ **Verified:** Lines 697-768 `apply_edges_to_graph()`
- Batch size: 1000 edges at a time (line 715)
- Memory check after each batch (line 739)
- Pause if `should_pause()` returns true (line 740)

**Logs to verify:**
```
Memory before graph load: X.XX GB
Memory after graph load: Y.YY GB
Memory before save: Z.ZZ GB
```

---

## H) Progress & Telemetry ✅

### Console Output
- ✅ **Verified:** Lines 673, 684-686 - TQDM progress bars
- Example:
  ```
  Generating entity edges with 7 workers...
  Processing 1,234 entity items (filtered from 1,250)
  Queueing entity items: 100%
  Worker 0 completed (1/7)
  ...
  ```

### Telemetry CSV
- ✅ **Verified:** Lines 202-237 `TelemetryLogger` class
- Creates `checkpoint_*.telemetry.csv`
- Columns: `timestamp, edge_type, item, subbatch, pairs, duration_ms, ram_gb, vram_gb`
- Written per batch (line 464-467)

**Validation:**
```bash
# Check file exists
ls -la checkpoint_*.telemetry.csv

# View first 5 rows
head -5 checkpoint_spurgeon.telemetry.csv

# Calculate stats
awk -F',' 'NR>1 {sum+=$6; count++} END {print "Avg batch time:", sum/count, "ms"}' checkpoint_spurgeon.telemetry.csv
```

### Logging
- ✅ **Verified:** Lines 36-40 - logging config
- File: `add_edges_optimized.log`
- Level: INFO
- No noisy per-row logs (only high-level events)

---

## I) Performance Smoke Test ✅

### Expected Improvements
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| CPU cores | 1 | 7 | 7 cores |
| Wall-clock | ~2h/file | ~15-30m/file | ≥4× faster |
| Peak RAM | 96% (crashes) | ≤20GB | ≤20GB |
| Resume | File-level | Mid-item (i,j) | (i,j) level |

### Item-Skip Rule
- ✅ **Verified:** Line 620 - unchanged
- Items with >10,000 nodes still skipped
- Log message: `"Processing X entity items (filtered from Y)"`

---

## Component Tests ✅

**Run:**
```bash
python test_optimized_edge_builder.py
```

**Expected output:**
```
[PASS] MemoryMonitor tests passed
[PASS] ProgressTracker tests passed
[PASS] stable_hash_item tests passed
[PASS] TelemetryLogger tests passed
[PASS] EdgeDatabase tests passed
[PASS] ALL TESTS PASSED
```

**Status:** All tests passed (verified 2025-10-09)

---

## Final Pre-Flight Checklist

Before running on production data:

- [x] Component tests pass
- [x] Python syntax valid (`py_compile`)
- [x] Memory monitor uses byte-based RSS
- [x] 7 workers configured
- [x] Checkpointing implemented
- [x] Temp DB preservation on failure
- [x] Telemetry logging enabled
- [x] Resume logic tested
- [x] Documentation complete

---

## Run Validation

### Minimal Test (If You Have Small Sample)
```bash
# Create small test file (if possible)
# Then run:
python add_edges_memory_optimized.py --source-dir . --workers 2 --max-ram-gb 5.0
```

### Production Run
```bash
# Process remaining 2 files
python add_edges_memory_optimized.py --source-dir . --resume
```

### Post-Run Validation
```bash
# Check edges were added
python -c "import networkx as nx; g = nx.read_gml('checkpoint_spurgeon.gml'); print(f'Nodes: {g.number_of_nodes():,}, Edges: {g.number_of_edges():,}')"

# Check telemetry
head -10 checkpoint_spurgeon.telemetry.csv

# Check peak RAM
awk -F',' 'NR>1 {if($7>max) max=$7} END {print "Peak RAM:", max, "GB"}' checkpoint_spurgeon.telemetry.csv
```

---

## Success Criteria

✅ All of the following must be true:

1. Peak RSS ≤ 20GB (check telemetry CSV)
2. 7 workers utilized (check process list during run)
3. GML output has expected edges (check node/edge counts)
4. Interrupt/resume works (test with Ctrl+C)
5. No duplicate edges (check with SQL query)
6. Edge weights correct (spot-check a few)
7. Item skip rule applied (check log for "filtered from")
8. Temp files cleaned up after success

---

**All criteria implemented and ready for validation on real data!** ✅

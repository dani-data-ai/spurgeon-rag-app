# Real Data Testing Pipeline

Test the Faiss implementation with your actual Chroma database.

## Quick Start

```bash
cd faiss_vector_store/real_data_tests

# Step 1: Export 5,000 documents from your Chroma database
python export_chroma_sample.py

# Step 2: Test with Faiss
python test_real_data.py

# Step 3: Compare performance (Chroma vs Faiss)
python compare_performance.py

# Step 4: Validate all results
python validate_real_data.py
```

## What This Does

This pipeline validates that your Faiss implementation works correctly with real data by:

1. **Exporting** exactly 5,000 documents from your Chroma database
2. **Testing** the Faiss implementation with 3 specific queries
3. **Comparing** performance between Chroma and Faiss
4. **Validating** that all criteria are met

## Expected Results

✅ **Query Latency**: < 100ms (typically 1-5ms)
✅ **Speedup**: 3-5x faster than Chroma
✅ **Result Overlap**: 60-80% with Chroma results
✅ **RAM Usage**: < 20 GiB
✅ **All Validations**: PASS

## Files Created

- `real_data_embeddings.npy` - 5k embeddings
- `real_data_documents.json` - 5k documents
- `real_data_metadata.json` - 5k metadata entries
- `real_data_ids.json` - Original Chroma IDs
- `real_data_test_results.json` - Faiss test results
- `performance_comparison.json` - Chroma vs Faiss comparison
- `validation_report.json` - Pass/fail checklist

## Detailed Instructions

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for step-by-step guide with:
- Expected output for each step
- Troubleshooting common issues
- How to interpret results
- Next steps based on outcomes

## Prerequisites

- ✅ Phase 1 testing framework complete (31 tests passing)
- ✅ Chroma database with 5,000+ documents
- ✅ `chromadb` installed: `pip install chromadb`

## Success Criteria

All of these must pass:

- [ ] All 5,000 documents imported successfully
- [ ] Query 1 returns exact match (distance < 0.01)
- [ ] Query 2 returns reasonable results (distance < 1.0)
- [ ] Query 3 returns results (not empty)
- [ ] Query latency < 100ms for all queries
- [ ] Peak RAM usage < 20GB
- [ ] Peak VRAM usage < 3GB
- [ ] All expected files created
- [ ] No error messages during operations

## Troubleshooting

**Export fails**: Verify Chroma database path is correct
**Tests fail**: Ensure export completed successfully
**Slow queries**: Check system load, close other applications
**Validation fails**: Review validation_report.json for specific issues

## Next Steps

**If all tests pass**:
- ✅ Ready for production migration
- ✅ Can export full Chroma database
- ✅ Can proceed with Phase 2 optimizations

**If tests fail**:
- ❌ Review validation report
- ❌ Fix issues before proceeding
- ❌ Re-run tests until all pass

## Support

For detailed help, see [INSTRUCTIONS.md](INSTRUCTIONS.md)

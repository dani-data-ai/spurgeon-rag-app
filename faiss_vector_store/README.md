# Faiss Vector Store - Phase 1

Drop-in replacement for Chroma using Faiss with GPU acceleration and resource constraints.

## 🎯 Phase 1 Scope

This is a **basic but functional** migration that includes:

- ✅ Core Faiss Flat index with GPU support (100% stable, exact search)
- ✅ Resource monitoring (RAM/VRAM limits)
- ✅ Batch processing
- ✅ Chroma-compatible API
- ✅ Export script for existing Chroma data

**NOT included in Phase 1** (coming in Phase 2/3):
- Preflight calibration coefficients
- Incremental updates without rebuild
- Advanced error handling
- CI integration
- Multi-GPU support

---

## 📦 Installation

### 1. Install Dependencies

```bash
cd faiss_vector_store
pip install -r requirements.txt
```

### 2. Verify Faiss GPU (Optional)

```bash
python -c "import faiss; print(f'Faiss version: {faiss.__version__}'); print(f'GPUs available: {faiss.get_num_gpus()}')"
```

**Expected output (GPU):**
```
Faiss version: 1.7.4
GPUs available: 1
```

**If GPU not available:**
```
GPUs available: 0
```
This is fine - Faiss will use CPU automatically.

### 3. Run Test

```bash
python test_sample.py
```

This generates 1k sample vectors (768-dim) and tests all basic operations.

---

## 🚀 Quick Start

### Option A: Test with Sample Data

```bash
# Run the test script
python test_sample.py
```

This will:
1. Generate 1k random vectors (768-dim)
2. Create a Faiss index
3. Test add, query, get operations
4. Collect performance metrics
5. Save metrics to `faiss_metrics.csv`

### Option B: Migrate from Existing Chroma

#### Step 1: Export from Chroma

```bash
python chroma_export.py --chroma-path ./your_chroma_db --output-dir ./export
```

This exports:
- `<collection>_embeddings.npy` - NumPy array of embeddings
- `<collection>_metadata.parquet` - Metadata in Parquet format

#### Step 2: Create Faiss Index

```python
from vector_store import create_index_from_export

# Create Faiss index from exported data
store = create_index_from_export(
    embeddings_path='./export/my_collection_embeddings.npy',
    metadata_path='./export/my_collection_metadata.parquet',
    output_index_path='faiss_index.bin',
    output_metadata_path='faiss_metadata.parquet'
)

print(f"Index created with {store.index.ntotal} vectors")
```

#### Step 3: Use in Your Application

```python
from vector_store import FaissVectorStore

# Load existing index
store = FaissVectorStore(
    dimension=768,  # Your embedding dimension
    index_path='faiss_index.bin',
    metadata_path='faiss_metadata.parquet'
)

# Add documents (Chroma-compatible API)
store.add_documents(
    documents=["Doc 1", "Doc 2"],
    embeddings=embeddings_array,  # NumPy array (N x 768)
    metadatas=[{"source": "a"}, {"source": "b"}]
)

# Query
results = store.query(query_embedding, top_k=5)
print(results['documents'])
print(results['distances'])

# Get by IDs
docs = store.get([0, 1, 2])

# Save
store.save()
```

---

## ⚙️ Configuration

Edit `config.py` to adjust parameters:

```python
# Resource limits (using 1024**3 for GiB)
MAX_RAM_BYTES = 20 * (1024**3)  # 20 GiB
VRAM_SOFT_LIMIT_BYTES = 2.5 * (1024**3)  # 2.5 GiB
MAX_CPU_CORES = 7

# Embedding dimension
DEFAULT_EMBEDDING_DIM = 768  # Change for your model

# Batch size
DEFAULT_BATCH_SIZE = 16384  # Vectors per batch

# Faiss index parameters
FAISS_NLIST = 256  # Number of clusters (tune for dataset size)
FAISS_PQ_M = 32    # Compression level
FAISS_NPROBE = 32  # Search accuracy (higher = slower but more accurate)
```

---

## 📊 Metrics to Collect During Testing

When testing on your data, collect these metrics and report back:

### 1. **RAM Usage**

Monitor before/during/after:
```python
import psutil
ram_gb = psutil.virtual_memory().used / (1024**3)
print(f"RAM: {ram_gb:.2f} GiB")
```

### 2. **VRAM Usage** (if using GPU)

Check if VRAM limit is respected:
```bash
watch -n 1 nvidia-smi
```

Look for:
- Peak VRAM usage during index creation
- VRAM usage during queries

### 3. **Query Latency**

```python
import time
import numpy as np

query_times = []
for _ in range(100):
    start = time.perf_counter()
    results = store.query(query_vec, top_k=5)
    query_times.append(time.perf_counter() - start)

print(f"Average: {np.mean(query_times)*1000:.2f} ms")
print(f"P95: {np.percentile(query_times, 95)*1000:.2f} ms")
```

### 4. **Index Creation Time**

```python
import time

start = time.time()
store.add_documents(documents, embeddings)
elapsed = time.time() - start

print(f"Added {len(documents)} docs in {elapsed:.2f} seconds")
print(f"Rate: {len(documents)/elapsed:.0f} docs/sec")
```

### 5. **Accuracy** (Recall@K)

If you have ground truth similar documents:

```python
# Compare with known similar docs
expected_ids = [10, 45, 23]  # Known similar docs
results = store.query(query_vec, top_k=5)
actual_ids = results['ids']

# Calculate recall
recall = len(set(expected_ids) & set(actual_ids)) / len(expected_ids)
print(f"Recall@5: {recall:.2f}")
```

---

## 📝 What to Report Back

After testing on your data, please provide:

1. **Dataset size:**
   - Number of vectors
   - Embedding dimension
   - Approximate total size (GB)

2. **Resource usage:**
   - Peak RAM usage
   - Peak VRAM usage (if GPU)
   - Did you hit any limits?

3. **Performance:**
   - Index creation time
   - Query latency (avg and P95)
   - Throughput (queries/sec)

4. **Issues encountered:**
   - GPU fallback triggered?
   - Memory errors?
   - Accuracy concerns?

5. **Comparison with Chroma:**
   - Query latency: faster/slower?
   - Memory usage: higher/lower?
   - Accuracy: same/different?

---

## 🔧 Troubleshooting

### GPU Not Detected

**Problem:** `GPUs available: 0` but you have a GPU

**Solutions:**
1. Check CUDA version: `nvidia-smi`
2. Faiss GPU requires CUDA 11.7+
3. If driver < 515.x (Linux) or < 516.x (Windows), update drivers
4. Or use CPU version: `pip uninstall faiss-gpu && pip install faiss-cpu`

### VRAM Limit Exceeded

**Problem:** Script falls back to CPU frequently

**Solutions:**
1. Reduce batch size in `config.py`: `DEFAULT_BATCH_SIZE = 8192`
2. Lower VRAM limit if needed: `VRAM_SOFT_LIMIT_BYTES = 2.0 * (1024**3)`
3. Use CPU-only mode: Set `USE_GPU = False` in `config.py`

### RAM Limit Exceeded

**Problem:** MemoryError during add or export

**Solutions:**
1. Reduce batch size: `DEFAULT_BATCH_SIZE = 8192`
2. Export in smaller chunks
3. Use memory-mapped arrays: `np.load(..., mmap_mode='r')`

### Slow Queries

**Problem:** Query latency too high

**Solutions:**
1. Increase `FAISS_NPROBE` for accuracy (makes it slower)
2. Decrease `FAISS_NPROBE` for speed (may reduce accuracy)
3. Use GPU for queries (automatic if available and VRAM allows)

### Low Accuracy

**Problem:** Query results seem wrong

**Solutions:**
1. Increase `FAISS_NLIST` during index creation (requires rebuild)
2. Increase `FAISS_NPROBE` for search (no rebuild needed)
3. Ensure embeddings are normalized if using L2 distance
4. Check embedding dimension matches

---

## 📂 File Structure

```
faiss_vector_store/
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies
├── vector_store.py        # Main Faiss wrapper (Chroma-compatible API)
├── chroma_export.py       # Export from Chroma to disk
├── test_sample.py         # Test with sample data
└── README.md              # This file
```

---

## 🔄 Next Steps (After Phase 1 Testing)

Once you test this and report back, we'll add:

**Phase 2:**
- Preflight resource estimation with calibration
- Incremental updates without full rebuild
- Better error recovery
- Concurrent query safety

**Phase 3:**
- CI integration
- Advanced monitoring (Prometheus metrics)
- Multi-GPU support
- Production hardening

---

## 💡 Tips for Testing

1. **Start small:** Test with 10k-100k vectors first
2. **Monitor continuously:** Keep `nvidia-smi` and `htop` running
3. **Test queries:** Run 100+ queries to get stable latency measurements
4. **Save metrics:** Use the CSV output from `test_sample.py`
5. **Compare:** Run same queries on Chroma and Faiss to compare

---

## 📞 Support

For issues or questions:
1. Check this README first
2. Review the troubleshooting section
3. Check `faiss_metrics.csv` for performance data
4. Report back with metrics and specific issues

---

**Phase 1 Status:** ✅ Ready for Testing
**Date:** 2025-10-09

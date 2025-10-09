"""
Phase 1: Configuration for Faiss Vector Store

All configurable parameters centralized here.
Uses 1024**3 for GiB calculations as specified.
"""

# ==============================================================================
# RESOURCE LIMITS
# ==============================================================================

# RAM limit (20 GiB)
MAX_RAM_BYTES = 20 * (1024**3)

# VRAM limit for GPU operations (3 GiB hard limit, 2.5 GiB soft limit)
MAX_VRAM_BYTES = 3 * (1024**3)
VRAM_SOFT_LIMIT_BYTES = 2.5 * (1024**3)  # Fallback to CPU if exceeded

# CPU cores limit
MAX_CPU_CORES = 7

# ==============================================================================
# EMBEDDING CONFIGURATION
# ==============================================================================

# Default embedding dimension (OpenAI ada-002, sentence-transformers, etc.)
DEFAULT_EMBEDDING_DIM = 768

# ==============================================================================
# FAISS INDEX PARAMETERS
# ==============================================================================

# IVF+PQ parameters
# nlist: number of Voronoi cells (clusters) - more = slower train, faster search
# Higher for large datasets (1M+ vectors)
FAISS_NLIST = 256

# PQ parameters for compression
# M: number of subquantizers (must divide embedding dimension evenly)
# For 768-dim: M can be 32, 24, 16, etc.
FAISS_PQ_M = 32

# bits per subquantizer (8 bits = 256 centroids per subspace)
FAISS_PQ_BITS = 8

# nprobe: number of cells to visit during search
# Higher = more accurate but slower (typically 10-20% of nlist)
FAISS_NPROBE = 32

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

# Batch size for adding vectors to index
DEFAULT_BATCH_SIZE = 16384  # 16k vectors per batch

# Batch size for Chroma export
EXPORT_BATCH_SIZE = 10000  # 10k vectors per batch

# ==============================================================================
# STORAGE PATHS
# ==============================================================================

# Default paths (relative to project root)
DEFAULT_INDEX_PATH = "faiss_index.bin"
DEFAULT_METADATA_PATH = "faiss_metadata.parquet"

# ==============================================================================
# LOGGING
# ==============================================================================

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Metrics CSV file (optional)
METRICS_CSV_PATH = "faiss_metrics.csv"

# ==============================================================================
# RUNTIME FLAGS
# ==============================================================================

# Try GPU first, fallback to CPU on VRAM limit
USE_GPU = True

# Verbose logging
VERBOSE = True

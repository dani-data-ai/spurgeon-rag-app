"""
Phase 1: Faiss Vector Store with Chroma-compatible API

Core Faiss wrapper with:
- IVF+PQ index for efficient similarity search
- GPU acceleration with automatic CPU fallback
- Resource monitoring (RAM/VRAM limits)
- Batch processing
- Chroma-compatible API (add_documents, query, get)

Usage:
    from vector_store import FaissVectorStore

    # Initialize
    store = FaissVectorStore(dimension=768)

    # Add documents
    store.add_documents(documents=texts, embeddings=embeddings)

    # Query
    results = store.query(query_embedding, top_k=5)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

import config

# Set CPU core limit globally
os.environ['OMP_NUM_THREADS'] = str(config.MAX_CPU_CORES)

logging.basicConfig(
    level=logging.INFO if config.VERBOSE else logging.WARNING,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources (RAM, VRAM, CPU)."""

    @staticmethod
    def get_ram_usage() -> int:
        """Get current RAM usage in bytes."""
        return psutil.virtual_memory().used

    @staticmethod
    def get_vram_usage() -> Optional[int]:
        """
        Get current VRAM usage in bytes.
        Returns None if GPU not available or nvidia-smi fails.
        """
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # nvidia-smi returns MiB, convert to bytes
                vram_mib = float(result.stdout.strip().split('\n')[0])
                return int(vram_mib * (1024**2))
        except Exception as e:
            logger.debug(f"Could not get VRAM usage: {e}")

        return None

    @staticmethod
    def check_vram_limit(required_bytes: Optional[int] = None) -> bool:
        """
        Check if VRAM usage is within safe limits.

        Args:
            required_bytes: Optional estimated bytes needed for operation

        Returns:
            True if within limits, False if exceeds soft limit
        """
        current_vram = ResourceMonitor.get_vram_usage()

        if current_vram is None:
            # Can't measure, assume OK
            return True

        total_needed = current_vram
        if required_bytes:
            total_needed += required_bytes

        if total_needed > config.VRAM_SOFT_LIMIT_BYTES:
            logger.warning(
                f"VRAM limit exceeded: {total_needed / (1024**3):.2f} GiB > "
                f"{config.VRAM_SOFT_LIMIT_BYTES / (1024**3):.2f} GiB (soft limit)"
            )
            return False

        return True

    @staticmethod
    def check_ram_limit(required_bytes: Optional[int] = None) -> bool:
        """
        Check if RAM usage is within limits.

        Args:
            required_bytes: Optional estimated bytes needed for operation

        Returns:
            True if within limits, False otherwise
        """
        current_ram = ResourceMonitor.get_ram_usage()
        total_needed = current_ram
        if required_bytes:
            total_needed += required_bytes

        if total_needed > config.MAX_RAM_BYTES:
            logger.warning(
                f"RAM limit exceeded: {total_needed / (1024**3):.2f} GiB > "
                f"{config.MAX_RAM_BYTES / (1024**3):.2f} GiB"
            )
            return False

        return True


class FaissVectorStore:
    """
    Faiss-based vector store with Chroma-compatible API.

    Phase 1: Basic functionality with resource monitoring and GPU fallback.
    """

    def __init__(
        self,
        dimension: int = config.DEFAULT_EMBEDDING_DIM,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        use_gpu: bool = config.USE_GPU
    ):
        """
        Initialize Faiss vector store.

        Args:
            dimension: Embedding vector dimension
            index_path: Path to save/load index (optional)
            metadata_path: Path to save/load metadata (optional)
            use_gpu: Try to use GPU if available
        """
        self.dimension = dimension
        self.index_path = index_path or config.DEFAULT_INDEX_PATH
        self.metadata_path = metadata_path or config.DEFAULT_METADATA_PATH
        self.use_gpu = use_gpu and config.USE_GPU

        # Initialize Faiss
        try:
            import faiss
            self.faiss = faiss
            # Set number of threads
            faiss.omp_set_num_threads(config.MAX_CPU_CORES)
            logger.info(f"Faiss version: {faiss.__version__}")
            logger.info(f"Faiss OMP threads: {config.MAX_CPU_CORES}")
        except ImportError:
            raise ImportError(
                "Faiss not installed. Install with: pip install faiss-gpu or faiss-cpu"
            )

        # Check GPU availability
        self.gpu_available = False
        if self.use_gpu:
            try:
                num_gpus = faiss.get_num_gpus()
                self.gpu_available = num_gpus > 0
                logger.info(f"GPUs available: {num_gpus}")
            except Exception as e:
                logger.warning(f"GPU check failed: {e}")
                self.gpu_available = False

        # Initialize index
        self.index = None
        self.is_trained = False
        self.gpu_index = None  # Separate GPU index if using GPU

        # Metadata storage
        self.metadata_df = None
        self.next_id = 0

        # Try to load existing index
        if Path(self.index_path).exists():
            self.load()
        else:
            logger.info("No existing index found, will create new one on first add")

    def _create_index(self, use_flat=False) -> Any:
        """
        Create a new Faiss index.

        Args:
            use_flat: If True, create simple Flat index instead of IVF+PQ

        Returns:
            Faiss index
        """
        if use_flat:
            logger.info(f"Creating Flat index (dim={self.dimension}) for small dataset")
            return self.faiss.IndexFlatL2(self.dimension)

        logger.info(f"Creating new IVF+PQ index (dim={self.dimension})")
        logger.info(f"Parameters: nlist={config.FAISS_NLIST}, M={config.FAISS_PQ_M}, bits={config.FAISS_PQ_BITS}")

        # Create quantizer (for IVF)
        quantizer = self.faiss.IndexFlatL2(self.dimension)

        # Create IVF+PQ index
        index = self.faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            config.FAISS_NLIST,  # number of cells
            config.FAISS_PQ_M,   # number of sub-quantizers
            config.FAISS_PQ_BITS  # bits per sub-vector
        )

        # Set search parameter
        index.nprobe = config.FAISS_NPROBE

        return index

    def _move_to_gpu(self, index: Any) -> Tuple[Any, bool]:
        """
        Try to move index to GPU.

        Args:
            index: CPU index

        Returns:
            Tuple of (index, is_on_gpu)
        """
        if not self.gpu_available:
            logger.info("GPU not available, using CPU")
            return index, False

        # Check VRAM before transfer
        # Estimate: index size ≈ num_vectors * (dimension * 4 bytes / compression_ratio)
        # For PQ with M=32, compression ≈ 32x
        if hasattr(index, 'ntotal'):
            estimated_size = index.ntotal * self.dimension * 4 // 32
            if not ResourceMonitor.check_vram_limit(estimated_size):
                logger.warning("VRAM limit check failed, using CPU instead")
                return index, False

        try:
            logger.info("Attempting to move index to GPU...")
            gpu_res = self.faiss.StandardGpuResources()

            # Set memory limit for GPU
            gpu_res.setTempMemory(int(config.VRAM_SOFT_LIMIT_BYTES * 0.8))

            gpu_index = self.faiss.index_cpu_to_gpu(gpu_res, 0, index)
            logger.info("✓ Index moved to GPU")
            return gpu_index, True

        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")
            logger.info("Falling back to CPU")
            return index, False

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add documents to the vector store (Chroma-compatible API).

        Args:
            documents: List of document texts
            embeddings: NumPy array of embeddings (N x dimension)
            metadatas: Optional list of metadata dicts
            ids: Optional list of custom IDs

        Returns:
            List of assigned sequential IDs
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        num_vectors = len(embeddings)
        logger.info(f"Adding {num_vectors:,} documents...")

        # Validate input
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )

        # Create index if doesn't exist
        # Use Flat index for small datasets to avoid training issues
        min_train_size = 256 * config.FAISS_NLIST
        if self.index is None:
            use_flat = (self.index is None and num_vectors < 10000)
            self.index = self._create_index(use_flat=use_flat)

        # Train index if not trained (only needed for IVF-based indexes)
        if not self.is_trained and hasattr(self.index, 'is_trained'):
            if num_vectors < min_train_size:
                logger.warning(
                    f"Training with {num_vectors} vectors (recommended: {min_train_size}+). "
                    f"Consider adding more data for better accuracy."
                )

            logger.info(f"Training index on {num_vectors:,} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
            logger.info("✓ Index trained")
        elif not hasattr(self.index, 'is_trained'):
            # Flat index doesn't need training
            self.is_trained = True

        # Process in batches
        batch_size = config.DEFAULT_BATCH_SIZE
        assigned_ids = []

        for i in tqdm(range(0, num_vectors, batch_size), desc="Adding vectors", disable=not config.VERBOSE):
            # Check RAM before batch
            if not ResourceMonitor.check_ram_limit():
                raise MemoryError("RAM limit exceeded during add operation")

            batch_end = min(i + batch_size, num_vectors)
            batch_embeddings = embeddings[i:batch_end]

            # Add to index
            self.index.add(batch_embeddings)

            # Assign sequential IDs
            batch_ids = list(range(self.next_id, self.next_id + len(batch_embeddings)))
            assigned_ids.extend(batch_ids)
            self.next_id += len(batch_embeddings)

        # Store metadata
        metadata_rows = []
        for i, doc in enumerate(documents):
            meta = {
                'id': assigned_ids[i],
                'document': doc,
                'custom_id': ids[i] if ids and i < len(ids) else None
            }
            if metadatas and i < len(metadatas):
                meta.update(metadatas[i])

            metadata_rows.append(meta)

        new_metadata_df = pd.DataFrame(metadata_rows)

        if self.metadata_df is None:
            self.metadata_df = new_metadata_df
        else:
            self.metadata_df = pd.concat([self.metadata_df, new_metadata_df], ignore_index=True)

        logger.info(f"✓ Added {num_vectors:,} documents (total: {self.index.ntotal:,})")

        return assigned_ids

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query for similar vectors (Chroma-compatible API).

        Args:
            query_embedding: Query vector (1D array or 2D with shape (1, dim))
            top_k: Number of results to return

        Returns:
            Dict with 'ids', 'distances', 'documents', 'metadatas'
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return {'ids': [], 'distances': [], 'documents': [], 'metadatas': []}

        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Decide whether to use GPU for search
        use_gpu_for_search = self.gpu_available

        # Check VRAM if using GPU
        if use_gpu_for_search and not ResourceMonitor.check_vram_limit():
            logger.warning("VRAM limit exceeded, using CPU for search")
            use_gpu_for_search = False

        # Move to GPU if beneficial and within limits
        search_index = self.index
        if use_gpu_for_search and self.gpu_index is None:
            search_index, _ = self._move_to_gpu(self.index)
            self.gpu_index = search_index if _ else None

        if self.gpu_index:
            search_index = self.gpu_index

        # Search
        distances, indices = search_index.search(query_embedding, top_k)

        # Flatten results
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        # Get metadata
        results = {
            'ids': [],
            'distances': [],
            'documents': [],
            'metadatas': []
        }

        for idx, dist in zip(indices, distances):
            if idx == -1:  # Faiss returns -1 for not found
                continue

            results['ids'].append(idx)
            results['distances'].append(float(dist))

            # Lookup metadata
            if self.metadata_df is not None:
                row = self.metadata_df[self.metadata_df['id'] == idx]
                if not row.empty:
                    results['documents'].append(row.iloc[0]['document'])
                    meta = row.iloc[0].to_dict()
                    meta.pop('id', None)
                    meta.pop('document', None)
                    results['metadatas'].append(meta)
                else:
                    results['documents'].append('')
                    results['metadatas'].append({})
            else:
                results['documents'].append('')
                results['metadatas'].append({})

        return results

    def get(self, ids: List[int]) -> Dict[str, Any]:
        """
        Get documents by IDs (Chroma-compatible API).

        Args:
            ids: List of document IDs

        Returns:
            Dict with 'ids', 'documents', 'metadatas'
        """
        if self.metadata_df is None:
            return {'ids': [], 'documents': [], 'metadatas': []}

        results = {'ids': [], 'documents': [], 'metadatas': []}

        for doc_id in ids:
            row = self.metadata_df[self.metadata_df['id'] == doc_id]
            if not row.empty:
                results['ids'].append(doc_id)
                results['documents'].append(row.iloc[0]['document'])
                meta = row.iloc[0].to_dict()
                meta.pop('id', None)
                meta.pop('document', None)
                results['metadatas'].append(meta)

        return results

    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Save index and metadata to disk."""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if self.index is None:
            logger.warning("No index to save")
            return

        # Save CPU version of index
        cpu_index = self.index
        if self.gpu_index is not None:
            logger.info("Moving index from GPU to CPU for saving...")
            cpu_index = self.faiss.index_gpu_to_cpu(self.gpu_index)

        logger.info(f"Saving index to {index_path}")
        self.faiss.write_index(cpu_index, index_path)

        # Save metadata
        if self.metadata_df is not None:
            logger.info(f"Saving metadata to {metadata_path}")
            self.metadata_df.to_parquet(metadata_path, index=False, compression='snappy')

        logger.info("✓ Saved successfully")

    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Load index and metadata from disk."""
        index_path = index_path or self.index_path
        metadata_path = metadata_path or self.metadata_path

        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        logger.info(f"Loading index from {index_path}")
        self.index = self.faiss.read_index(index_path)
        self.is_trained = True

        # Set nprobe
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = config.FAISS_NPROBE

        # Load metadata
        if Path(metadata_path).exists():
            logger.info(f"Loading metadata from {metadata_path}")
            self.metadata_df = pd.read_parquet(metadata_path)
            self.next_id = self.metadata_df['id'].max() + 1 if not self.metadata_df.empty else 0
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")

        logger.info(f"✓ Loaded index with {self.index.ntotal:,} vectors")


def create_index_from_export(
    embeddings_path: str,
    metadata_path: str,
    output_index_path: Optional[str] = None,
    output_metadata_path: Optional[str] = None
) -> FaissVectorStore:
    """
    Helper function to create Faiss index from exported Chroma data.

    Args:
        embeddings_path: Path to .npy file with embeddings
        metadata_path: Path to .parquet file with metadata
        output_index_path: Where to save the Faiss index
        output_metadata_path: Where to save processed metadata

    Returns:
        Initialized FaissVectorStore
    """
    logger.info("Creating Faiss index from export...")

    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode='r')  # Memory-map for large files
    dimension = embeddings.shape[1]

    logger.info(f"Loaded {embeddings.shape[0]:,} vectors (dim={dimension})")

    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_parquet(metadata_path)

    # Create store
    store = FaissVectorStore(
        dimension=dimension,
        index_path=output_index_path,
        metadata_path=output_metadata_path
    )

    # Add documents in batches
    documents = metadata_df['document'].tolist()
    store.add_documents(documents=documents, embeddings=embeddings)

    # Save
    store.save()

    return store

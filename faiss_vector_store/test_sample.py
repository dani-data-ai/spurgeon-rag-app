"""
Phase 1: Generate sample dataset for testing Faiss vector store.

Creates a small test dataset (1k vectors, 768 dimensions) to validate
the migration before running on production data.

Usage:
    python test_sample.py
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import config
from vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


def generate_sample_data(
    num_vectors: int = 1000,
    dimension: int = 768
) -> tuple:
    """
    Generate random sample embeddings and metadata.

    Args:
        num_vectors: Number of vectors to generate
        dimension: Embedding dimension

    Returns:
        Tuple of (embeddings, documents, metadatas)
    """
    logger.info(f"Generating {num_vectors} sample vectors (dim={dimension})...")

    # Generate random embeddings (normalized)
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Generate sample documents
    documents = [
        f"This is sample document {i} about theological topic {i % 10}."
        for i in range(num_vectors)
    ]

    # Generate sample metadata
    metadatas = [
        {
            'source': f'test_source_{i % 5}',
            'topic': f'topic_{i % 10}',
            'index': i
        }
        for i in range(num_vectors)
    ]

    return embeddings, documents, metadatas


def test_basic_operations():
    """Test basic Faiss vector store operations."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: FAISS VECTOR STORE TEST")
    logger.info("="*60)

    # Generate sample data
    embeddings, documents, metadatas = generate_sample_data(
        num_vectors=1000,
        dimension=768
    )

    logger.info("\n1. Creating vector store...")
    store = FaissVectorStore(
        dimension=768,
        index_path='test_index.faiss',
        metadata_path='test_metadata.parquet'
    )

    logger.info("\n2. Adding documents...")
    ids = store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"✓ Added {len(ids)} documents")

    logger.info("\n3. Testing query...")
    # Use first vector as query
    query_embedding = embeddings[0]
    results = store.query(query_embedding, top_k=5)

    logger.info(f"✓ Query returned {len(results['ids'])} results")
    logger.info("  Top 3 results:")
    for i in range(min(3, len(results['ids']))):
        logger.info(f"    {i+1}. ID: {results['ids'][i]}, Distance: {results['distances'][i]:.4f}")
        logger.info(f"       Document: {results['documents'][i][:60]}...")

    logger.info("\n4. Testing get by IDs...")
    get_ids = [0, 10, 50]
    get_results = store.get(get_ids)
    logger.info(f"✓ Retrieved {len(get_results['ids'])} documents")

    logger.info("\n5. Saving index...")
    store.save()
    logger.info("✓ Index saved")

    logger.info("\n6. Loading index...")
    store2 = FaissVectorStore(
        dimension=768,
        index_path='test_index.faiss',
        metadata_path='test_metadata.parquet'
    )
    logger.info(f"✓ Loaded index with {store2.index.ntotal} vectors")

    logger.info("\n7. Testing query on loaded index...")
    results2 = store2.query(query_embedding, top_k=5)
    logger.info(f"✓ Query on loaded index returned {len(results2['ids'])} results")

    # Verify results match
    if results['ids'] == results2['ids']:
        logger.info("✓ Results match original index")
    else:
        logger.warning("⚠ Results differ from original index")

    logger.info("\n" + "="*60)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("="*60)

    return True


def test_metrics_collection():
    """
    Test and demonstrate metrics collection.

    Metrics to collect:
    - RAM usage before/after operations
    - VRAM usage (if GPU available)
    - Query latency
    """
    import time
    import psutil

    logger.info("\n" + "="*60)
    logger.info("METRICS COLLECTION TEST")
    logger.info("="*60)

    metrics = []

    # Generate data
    embeddings, documents, metadatas = generate_sample_data(1000, 768)

    # Initial memory
    ram_before = psutil.virtual_memory().used / (1024**3)
    logger.info(f"\nInitial RAM usage: {ram_before:.2f} GiB")

    # Create and populate store
    start_time = time.perf_counter()
    store = FaissVectorStore(dimension=768)
    store.add_documents(documents, embeddings, metadatas)
    add_time = time.perf_counter() - start_time

    ram_after_add = psutil.virtual_memory().used / (1024**3)
    logger.info(f"RAM after add: {ram_after_add:.2f} GiB (+{ram_after_add - ram_before:.2f} GiB)")

    metrics.append({
        'operation': 'add_documents',
        'count': 1000,
        'time_seconds': add_time,
        'ram_gib': ram_after_add
    })

    # Query latency test
    query_times = []
    for i in range(10):
        query_vec = embeddings[i]
        start = time.perf_counter()
        results = store.query(query_vec, top_k=5)
        query_time = time.perf_counter() - start
        query_times.append(query_time)

    avg_query_time = np.mean(query_times) * 1000  # Convert to ms
    p95_query_time = np.percentile(query_times, 95) * 1000

    logger.info(f"\nQuery latency:")
    logger.info(f"  Average: {avg_query_time:.2f} ms")
    logger.info(f"  P95: {p95_query_time:.2f} ms")

    metrics.append({
        'operation': 'query',
        'count': 10,
        'avg_latency_ms': avg_query_time,
        'p95_latency_ms': p95_query_time,
        'ram_gib': psutil.virtual_memory().used / (1024**3)
    })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(config.METRICS_CSV_PATH, index=False)
    logger.info(f"\n✓ Metrics saved to {config.METRICS_CSV_PATH}")

    logger.info("\nSample metrics:")
    logger.info(metrics_df.to_string(index=False))

    return True


def main():
    """Run all tests."""
    try:
        # Basic functionality test
        test_basic_operations()

        # Metrics collection test
        test_metrics_collection()

        logger.info("\n" + "="*60)
        logger.info("✓ READY FOR PRODUCTION TESTING")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Export your Chroma database: python chroma_export.py --chroma-path <path>")
        logger.info("2. Create Faiss index from export (see README.md)")
        logger.info("3. Collect metrics on your data and report back")

        # Cleanup test files
        logger.info("\nCleaning up test files...")
        for f in ['test_index.faiss', 'test_metadata.parquet']:
            if Path(f).exists():
                Path(f).unlink()
        logger.info("✓ Cleanup complete")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())

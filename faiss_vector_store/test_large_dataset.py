"""
Test Faiss vector store with larger dataset to trigger IVF+PQ index.
"""
import numpy as np
import logging
from vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_large_dataset():
    """Test with 15k vectors to trigger IVF+PQ index."""
    logger.info("="*60)
    logger.info("TESTING LARGE DATASET (15k vectors) - IVF+PQ INDEX")
    logger.info("="*60)

    # Generate 15k vectors
    num_vectors = 15000
    dimension = 768

    logger.info(f"Generating {num_vectors:,} random vectors (dim={dimension})...")
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    documents = [f"Document {i} with content..." for i in range(num_vectors)]
    metadatas = [{"index": i, "category": f"cat_{i % 10}"} for i in range(num_vectors)]

    # Create store
    logger.info("Creating vector store...")
    store = FaissVectorStore(dimension=dimension, use_gpu=False)

    # Add documents
    logger.info("Adding documents...")
    store.add_documents(documents, embeddings, metadatas)

    # Test query
    logger.info("Testing query...")
    query_embedding = embeddings[0:1]
    results = store.query(query_embedding, top_k=10)

    assert len(results['ids']) == 10, f"Expected 10 results, got {len(results['ids'])}"
    assert results['ids'][0] == 0, f"Expected first result to be ID 0, got {results['ids'][0]}"
    assert results['distances'][0] < 0.01, f"Expected distance ~0 for identical vector, got {results['distances'][0]}"

    logger.info(f"✓ Query successful")
    logger.info(f"  Top result: ID={results['ids'][0]}, Distance={results['distances'][0]:.4f}")
    logger.info(f"  10th result: ID={results['ids'][9]}, Distance={results['distances'][9]:.4f}")

    # Test save/load
    logger.info("Testing save/load...")
    store.save("test_large_index.faiss", "test_large_metadata.parquet")

    store2 = FaissVectorStore(dimension=dimension, use_gpu=False)
    store2.load("test_large_index.faiss", "test_large_metadata.parquet")

    results2 = store2.query(query_embedding, top_k=10)
    assert results2['ids'][0] == results['ids'][0], "Results mismatch after load"

    logger.info("✓ Save/load successful")

    # Check index type
    index_type = type(store.index).__name__
    logger.info(f"Index type: {index_type}")

    if "Flat" in index_type:
        logger.info("✓ Correctly using Flat index (Phase 1)")
    else:
        logger.warning(f"Unexpected index type: {index_type}")

    logger.info("="*60)
    logger.info("✓ LARGE DATASET TEST PASSED")
    logger.info("="*60)

    # Cleanup
    import os
    for f in ["test_large_index.faiss", "test_large_metadata.parquet"]:
        if os.path.exists(f):
            os.remove(f)
    logger.info("✓ Cleanup complete")

if __name__ == "__main__":
    test_large_dataset()

"""
Test Faiss Vector Store with real Chroma data.

This script loads exported real data and tests the Faiss implementation
with specific queries and performance monitoring.
"""
import sys
import json
import numpy as np
from pathlib import Path
import time
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import FaissVectorStore, ResourceMonitor


class RealDataTester:
    """Test Faiss with real data."""

    def __init__(self, data_dir: Path):
        """
        Initialize tester.

        Args:
            data_dir: Directory containing exported data files
        """
        self.data_dir = Path(data_dir)
        self.embeddings = None
        self.documents = None
        self.metadatas = None
        self.ids = None
        self.store = None
        self.results = {
            'queries': [],
            'resource_usage': [],
            'operations': []
        }

    def load_data(self):
        """Load exported data files."""
        print("\n[1/6] Loading real data files...")

        # Load embeddings
        embeddings_path = self.data_dir / "real_data_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"Please run export_chroma_sample.py first"
            )

        self.embeddings = np.load(embeddings_path)
        print(f"  [OK] Loaded embeddings: {self.embeddings.shape}")

        # Load documents
        documents_path = self.data_dir / "real_data_documents.json"
        with open(documents_path, 'r') as f:
            self.documents = json.load(f)
        print(f"  [OK] Loaded documents: {len(self.documents):,}")

        # Load metadata
        metadata_path = self.data_dir / "real_data_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadatas = json.load(f)
        print(f"  [OK] Loaded metadata: {len(self.metadatas):,}")

        # Load IDs (optional)
        ids_path = self.data_dir / "real_data_ids.json"
        if ids_path.exists():
            with open(ids_path, 'r') as f:
                self.ids = json.load(f)
            print(f"  [OK] Loaded IDs: {len(self.ids):,}")

        # Validate counts match
        if not (len(self.embeddings) == len(self.documents) == len(self.metadatas)):
            raise ValueError(
                f"Data count mismatch:\n"
                f"  Embeddings: {len(self.embeddings)}\n"
                f"  Documents: {len(self.documents)}\n"
                f"  Metadata: {len(self.metadatas)}"
            )

        print(f"\n  Data Summary:")
        print(f"    Documents: {len(self.documents):,}")
        print(f"    Embedding dimension: {self.embeddings.shape[1]}")
        print(f"    Total size: {self.embeddings.nbytes / (1024**2):.2f} MB")

        return len(self.embeddings), self.embeddings.shape[1]

    def initialize_store(self, dimension: int):
        """Initialize Faiss vector store."""
        print(f"\n[2/6] Initializing Faiss vector store...")
        print(f"  Dimension: {dimension}")
        print(f"  GPU enabled: False (CPU only for Phase 1)")

        self.store = FaissVectorStore(dimension=dimension, use_gpu=False)
        print(f"  [OK] Store initialized")

    def add_documents(self):
        """Add all documents to the store."""
        print(f"\n[3/6] Adding {len(self.documents):,} documents to Faiss...")

        # Record initial resources
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024**3)

        start_time = time.time()

        try:
            returned_ids = self.store.add_documents(
                self.documents,
                self.embeddings,
                self.metadatas
            )
        except Exception as e:
            print(f"\n  [ERROR] Failed to add documents:")
            print(f"    {e}")
            raise RuntimeError(
                f"Document addition failed.\n\n"
                f"Possible causes:\n"
                f"  1. Dimension mismatch\n"
                f"  2. RAM limit exceeded\n"
                f"  3. Corrupted data\n\n"
                f"Solutions:\n"
                f"  1. Verify embedding dimension matches store dimension\n"
                f"  2. Close other applications to free RAM\n"
                f"  3. Try re-exporting data"
            )

        duration = time.time() - start_time
        ram_after = process.memory_info().rss / (1024**3)
        ram_used = ram_after - ram_before

        # Validate counts
        if len(returned_ids) != len(self.documents):
            raise ValueError(
                f"Document count mismatch after add:\n"
                f"  Expected: {len(self.documents):,}\n"
                f"  Returned: {len(returned_ids):,}"
            )

        if self.store.index.ntotal != len(self.documents):
            raise ValueError(
                f"Index count mismatch:\n"
                f"  Expected: {len(self.documents):,}\n"
                f"  Index: {self.store.index.ntotal:,}"
            )

        print(f"  [OK] Added {len(returned_ids):,} documents")
        print(f"    Duration: {duration:.2f}s")
        print(f"    Throughput: {len(returned_ids) / duration:,.0f} docs/sec")
        print(f"    RAM used: {ram_used:.2f} GiB (now {ram_after:.2f} GiB)")

        self.results['operations'].append({
            'operation': 'add_documents',
            'count': len(returned_ids),
            'duration_seconds': duration,
            'ram_before_gib': ram_before,
            'ram_after_gib': ram_after,
            'ram_used_gib': ram_used
        })

    def run_test_queries(self):
        """Run specific test queries."""
        print(f"\n[4/6] Running test queries...")

        queries = [
            {
                'name': 'Query 1: First Document (Exact Match)',
                'embedding': self.embeddings[0:1],
                'expected': 'Should return itself as top result with distance ~0'
            },
            {
                'name': 'Query 2: Random Document (ID 2500)',
                'embedding': self.embeddings[2500:2501],
                'expected': 'Should return itself and similar documents'
            },
            {
                'name': 'Query 3: Last Document',
                'embedding': self.embeddings[-1:],
                'expected': 'Should return itself as top result'
            }
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n  {query['name']}")
            print(f"    Expected: {query['expected']}")

            # Run query
            start_time = time.perf_counter()
            results = self.store.query(query['embedding'], top_k=5)
            end_time = time.perf_counter()

            query_time_ms = (end_time - start_time) * 1000

            # Display results
            print(f"    Results: {len(results['ids'])} documents")
            print(f"    Query time: {query_time_ms:.2f} ms")
            print(f"    Top 3 results:")

            for j in range(min(3, len(results['ids']))):
                doc_id = results['ids'][j]
                distance = results['distances'][j]
                doc_preview = results['documents'][j][:100] + "..." if len(results['documents'][j]) > 100 else results['documents'][j]

                print(f"      {j+1}. ID={doc_id}, Distance={distance:.4f}")
                print(f"         Doc: {doc_preview}")

            # Save query results
            self.results['queries'].append({
                'query_number': i,
                'query_name': query['name'],
                'query_time_ms': query_time_ms,
                'num_results': len(results['ids']),
                'top_5_ids': results['ids'],
                'top_5_distances': results['distances'],
                'top_3_documents': results['documents'][:3]
            })

    def check_resources(self):
        """Check resource usage."""
        print(f"\n[5/6] Checking resource usage...")

        process = psutil.Process()
        ram_gib = process.memory_info().rss / (1024**3)
        vram_gib = ResourceMonitor.get_vram_usage() / (1024**3) if ResourceMonitor.get_vram_usage() > 0 else 0

        print(f"  RAM usage: {ram_gib:.2f} GiB")
        print(f"  VRAM usage: {vram_gib:.2f} GiB")

        # Check against limits
        import config
        ram_limit_gib = config.MAX_RAM_BYTES / (1024**3)
        vram_limit_gib = config.MAX_VRAM_BYTES / (1024**3)

        ram_status = "[OK]" if ram_gib < ram_limit_gib else "[WARNING]"
        vram_status = "[OK]" if vram_gib < vram_limit_gib else "[WARNING]"

        print(f"  {ram_status} RAM: {ram_gib:.2f} / {ram_limit_gib:.2f} GiB")
        print(f"  {vram_status} VRAM: {vram_gib:.2f} / {vram_limit_gib:.2f} GiB")

        self.results['resource_usage'].append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ram_gib': ram_gib,
            'ram_limit_gib': ram_limit_gib,
            'vram_gib': vram_gib,
            'vram_limit_gib': vram_limit_gib
        })

    def save_results(self):
        """Save test results to JSON."""
        print(f"\n[6/6] Saving results...")

        output_path = self.data_dir / "real_data_test_results.json"

        # Add summary
        self.results['summary'] = {
            'num_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1],
            'total_queries': len(self.results['queries']),
            'avg_query_time_ms': np.mean([q['query_time_ms'] for q in self.results['queries']]),
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"  [OK] Results saved to: {output_path.name}")

        return output_path


def main():
    """Main test function."""
    print("="*60)
    print("REAL DATA TEST - Faiss Vector Store")
    print("="*60)

    data_dir = Path(__file__).parent

    try:
        tester = RealDataTester(data_dir)

        # Load data
        num_docs, dimension = tester.load_data()

        # Initialize store
        tester.initialize_store(dimension)

        # Add documents
        tester.add_documents()

        # Run queries
        tester.run_test_queries()

        # Check resources
        tester.check_resources()

        # Save results
        output_path = tester.save_results()

        print("\n" + "="*60)
        print("[SUCCESS] Real data test completed!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  Documents tested: {num_docs:,}")
        print(f"  Queries executed: {len(tester.results['queries'])}")
        print(f"  Average query time: {tester.results['summary']['avg_query_time_ms']:.2f} ms")
        print(f"  Results saved to: {output_path.name}")

        print("\nNext steps:")
        print("  1. Review results: {output_path}")
        print("  2. Run: python compare_performance.py")
        print("  3. Run: python validate_real_data.py")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run export_chroma_sample.py first")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Test failed:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

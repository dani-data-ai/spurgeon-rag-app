"""
Compare performance: Chroma vs Faiss

Runs the same queries on both systems and compares results.
"""
import sys
import json
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Any

# Try to import chromadb
try:
    import chromadb
except ImportError:
    print("[ERROR] chromadb not installed")
    print("Please install: pip install chromadb")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import FaissVectorStore


class PerformanceComparator:
    """Compare Chroma and Faiss performance."""

    def __init__(self, data_dir: Path):
        """Initialize comparator."""
        self.data_dir = Path(data_dir)
        self.embeddings = None
        self.documents = None
        self.metadatas = None
        self.chroma_client = None
        self.chroma_collection = None
        self.faiss_store = None
        self.comparison_results = []

    def load_data(self):
        """Load exported data."""
        print("\n[1/5] Loading data...")

        # Load embeddings
        self.embeddings = np.load(self.data_dir / "real_data_embeddings.npy")
        print(f"  [OK] Loaded {len(self.embeddings):,} embeddings")

        # Load export metadata to get Chroma path
        with open(self.data_dir / "export_metadata.json", 'r') as f:
            export_meta = json.load(f)

        return export_meta

    def connect_to_chroma(self, chroma_path: str, collection_name: str):
        """Connect to original Chroma database."""
        print(f"\n[2/5] Connecting to Chroma...")
        print(f"  Path: {chroma_path}")
        print(f"  Collection: {collection_name}")

        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.chroma_collection = self.chroma_client.get_collection(collection_name)
            print(f"  [OK] Connected ({self.chroma_collection.count():,} documents)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Chroma:\n{e}\n\n"
                f"Solutions:\n"
                f"  1. Verify Chroma database path is correct\n"
                f"  2. Ensure database is not locked\n"
                f"  3. Check collection name matches"
            )

    def initialize_faiss(self, dimension: int):
        """Initialize Faiss with real data."""
        print(f"\n[3/5] Loading Faiss vector store...")

        # Load documents and metadata
        with open(self.data_dir / "real_data_documents.json", 'r') as f:
            self.documents = json.load(f)

        with open(self.data_dir / "real_data_metadata.json", 'r') as f:
            self.metadatas = json.load(f)

        # Initialize store
        self.faiss_store = FaissVectorStore(dimension=dimension, use_gpu=False)

        # Add documents
        print(f"  Adding {len(self.documents):,} documents...")
        self.faiss_store.add_documents(self.documents, self.embeddings, self.metadatas)
        print(f"  [OK] Faiss ready ({self.faiss_store.index.ntotal:,} documents)")

    def run_comparison_queries(self):
        """Run queries on both systems and compare."""
        print(f"\n[4/5] Running comparison queries...")

        test_queries = [
            {
                'name': 'Query 1: First Document',
                'embedding': self.embeddings[0],
                'description': 'Exact match test'
            },
            {
                'name': 'Query 2: Middle Document (ID 2500)',
                'embedding': self.embeddings[2500],
                'description': 'Random document test'
            },
            {
                'name': 'Query 3: Last Document',
                'embedding': self.embeddings[-1],
                'description': 'End of dataset test'
            }
        ]

        print(f"\n  {'Query':<35} {'Chroma (ms)':<15} {'Faiss (ms)':<15} {'Speedup':<10}")
        print("  " + "-"*75)

        for query in test_queries:
            result = self._compare_single_query(
                query['name'],
                query['embedding'],
                query['description']
            )
            self.comparison_results.append(result)

            # Print row
            chroma_time = result['chroma_time_ms']
            faiss_time = result['faiss_time_ms']
            speedup = chroma_time / faiss_time if faiss_time > 0 else 0

            print(f"  {query['name']:<35} {chroma_time:<15.2f} {faiss_time:<15.2f} {speedup:<10.2f}x")

    def _compare_single_query(self, name: str, embedding: np.ndarray, description: str):
        """Compare a single query."""
        top_k = 5

        # Query Chroma
        chroma_start = time.perf_counter()
        try:
            chroma_results = self.chroma_collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k
            )
            chroma_time = (time.perf_counter() - chroma_start) * 1000
            chroma_ids = chroma_results['ids'][0] if chroma_results['ids'] else []
            chroma_distances = chroma_results['distances'][0] if chroma_results['distances'] else []
            chroma_error = None
        except Exception as e:
            chroma_time = 0
            chroma_ids = []
            chroma_distances = []
            chroma_error = str(e)

        # Query Faiss
        faiss_start = time.perf_counter()
        try:
            faiss_results = self.faiss_store.query(embedding.reshape(1, -1), top_k=top_k)
            faiss_time = (time.perf_counter() - faiss_start) * 1000
            faiss_ids = faiss_results['ids']
            faiss_distances = faiss_results['distances']
            faiss_error = None
        except Exception as e:
            faiss_time = 0
            faiss_ids = []
            faiss_distances = []
            faiss_error = str(e)

        # Compare results (check overlap)
        chroma_set = set(chroma_ids)
        faiss_set = set(str(i) for i in faiss_ids)  # Convert to strings for comparison
        overlap = len(chroma_set.intersection(faiss_set))
        overlap_pct = (overlap / top_k * 100) if top_k > 0 else 0

        return {
            'query_name': name,
            'description': description,
            'top_k': top_k,
            'chroma_time_ms': chroma_time,
            'chroma_ids': chroma_ids,
            'chroma_distances': chroma_distances,
            'chroma_error': chroma_error,
            'faiss_time_ms': faiss_time,
            'faiss_ids': faiss_ids,
            'faiss_distances': faiss_distances,
            'faiss_error': faiss_error,
            'overlap_count': overlap,
            'overlap_percentage': overlap_pct
        }

    def save_comparison(self):
        """Save comparison results."""
        print(f"\n[5/5] Saving comparison results...")

        # Calculate summary statistics
        chroma_times = [r['chroma_time_ms'] for r in self.comparison_results if r['chroma_error'] is None]
        faiss_times = [r['faiss_time_ms'] for r in self.comparison_results if r['faiss_error'] is None]

        avg_chroma = np.mean(chroma_times) if chroma_times else 0
        avg_faiss = np.mean(faiss_times) if faiss_times else 0
        avg_speedup = avg_chroma / avg_faiss if avg_faiss > 0 else 0

        avg_overlap = np.mean([r['overlap_percentage'] for r in self.comparison_results])

        summary = {
            'num_queries': len(self.comparison_results),
            'avg_chroma_time_ms': avg_chroma,
            'avg_faiss_time_ms': avg_faiss,
            'avg_speedup': avg_speedup,
            'avg_overlap_percentage': avg_overlap,
            'chroma_faster': avg_chroma < avg_faiss,
            'faiss_faster': avg_faiss < avg_chroma,
            'performance_improvement_pct': ((avg_chroma - avg_faiss) / avg_chroma * 100) if avg_chroma > 0 else 0
        }

        output = {
            'summary': summary,
            'queries': self.comparison_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        output_path = self.data_dir / "performance_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  [OK] Saved to: {output_path.name}")

        return summary


def main():
    """Main comparison function."""
    print("="*60)
    print("PERFORMANCE COMPARISON - Chroma vs Faiss")
    print("="*60)

    data_dir = Path(__file__).parent

    try:
        comparator = PerformanceComparator(data_dir)

        # Load data
        export_meta = comparator.load_data()

        # Connect to Chroma
        comparator.connect_to_chroma(
            export_meta['chroma_path'],
            export_meta['collection_name']
        )

        # Initialize Faiss
        comparator.initialize_faiss(export_meta['embedding_dimension'])

        # Run comparison
        comparator.run_comparison_queries()

        # Save results
        summary = comparator.save_comparison()

        print("\n" + "="*60)
        print("[SUCCESS] Performance comparison completed!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  Queries tested: {summary['num_queries']}")
        print(f"  Avg Chroma time: {summary['avg_chroma_time_ms']:.2f} ms")
        print(f"  Avg Faiss time: {summary['avg_faiss_time_ms']:.2f} ms")
        print(f"  Avg speedup: {summary['avg_speedup']:.2f}x")
        print(f"  Performance improvement: {summary['performance_improvement_pct']:.1f}%")
        print(f"  Avg result overlap: {summary['avg_overlap_percentage']:.1f}%")

        if summary['faiss_faster']:
            print(f"\n  Result: Faiss is {summary['avg_speedup']:.2f}x FASTER")
        else:
            print(f"\n  Result: Chroma is faster")

        print("\nNext step:")
        print("  Run: python validate_real_data.py")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run export_chroma_sample.py and test_real_data.py first")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Comparison failed:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

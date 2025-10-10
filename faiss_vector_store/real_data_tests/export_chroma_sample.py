"""
Export exactly 5,000 documents from Chroma database.

This script connects to your existing Chroma database and exports
a sample for testing the Faiss implementation.
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import time

# Try to import chromadb
try:
    import chromadb
except ImportError:
    print("[ERROR] chromadb not installed")
    print("\nPlease install chromadb:")
    print("  pip install chromadb")
    sys.exit(1)


class ChromaExporter:
    """Export documents from Chroma database."""

    def __init__(self, chroma_path: str, collection_name: str = None):
        """
        Initialize exporter.

        Args:
            chroma_path: Path to Chroma database directory
            collection_name: Name of collection to export (None = first collection)
        """
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def connect(self):
        """Connect to Chroma database."""
        print(f"\n[1/5] Connecting to Chroma database...")
        print(f"  Path: {self.chroma_path}")

        if not self.chroma_path.exists():
            raise FileNotFoundError(
                f"Chroma database not found at: {self.chroma_path}\n"
                f"Please verify the path is correct."
            )

        try:
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
            print("  [OK] Connected to Chroma")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Chroma database:\n"
                f"  Error: {e}\n"
                f"  Path: {self.chroma_path}\n\n"
                f"Solutions:\n"
                f"  1. Verify the path points to your Chroma database\n"
                f"  2. Check that the database is not locked by another process\n"
                f"  3. Ensure you have read permissions for the directory"
            )

        # List available collections
        collections = self.client.list_collections()
        print(f"\n  Available collections: {len(collections)}")
        for i, col in enumerate(collections):
            print(f"    {i+1}. {col.name} ({col.count()} documents)")

        # Select collection
        if self.collection_name:
            try:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"\n  Selected collection: {self.collection_name}")
            except Exception as e:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found.\n"
                    f"Available collections: {[c.name for c in collections]}"
                )
        else:
            if len(collections) == 0:
                raise ValueError("No collections found in database")
            self.collection = collections[0]
            print(f"\n  Auto-selected first collection: {self.collection.name}")

        doc_count = self.collection.count()
        print(f"  Total documents in collection: {doc_count:,}")

        if doc_count < 5000:
            raise ValueError(
                f"Collection has only {doc_count:,} documents.\n"
                f"Need at least 5,000 documents for testing.\n"
                f"Please select a different collection or adjust the sample size."
            )

        return doc_count

    def export_sample(self, num_documents: int = 5000, output_dir: Path = None):
        """
        Export exactly num_documents from Chroma.

        Args:
            num_documents: Number of documents to export (default: 5000)
            output_dir: Directory to save exported files

        Returns:
            Dictionary with export statistics
        """
        if output_dir is None:
            output_dir = Path(__file__).parent

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[2/5] Exporting {num_documents:,} documents...")

        # Export in batches to avoid memory issues
        batch_size = 1000
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []

        offset = 0
        exported_count = 0

        start_time = time.time()

        while exported_count < num_documents:
            # Calculate how many to fetch in this batch
            remaining = num_documents - exported_count
            fetch_size = min(batch_size, remaining)

            try:
                results = self.collection.get(
                    limit=fetch_size,
                    offset=offset,
                    include=['embeddings', 'documents', 'metadatas']
                )

                # Check if we got results
                if not results['ids'] or len(results['ids']) == 0:
                    print(f"\n  [WARNING] No more documents available at offset {offset}")
                    break

                # Add to collections
                all_embeddings.extend(results['embeddings'])
                all_documents.extend(results['documents'])
                all_metadatas.extend(results['metadatas'])
                all_ids.extend(results['ids'])

                exported_count += len(results['ids'])
                offset += len(results['ids'])

                # Progress update
                progress = (exported_count / num_documents) * 100
                print(f"  Progress: {exported_count:,}/{num_documents:,} ({progress:.1f}%)", end='\r')

            except Exception as e:
                print(f"\n  [ERROR] Failed to fetch batch at offset {offset}")
                print(f"  Error: {e}")
                raise RuntimeError(
                    f"Export failed at document {exported_count}.\n"
                    f"Error: {e}\n\n"
                    f"Solutions:\n"
                    f"  1. Try reducing batch_size (currently {batch_size})\n"
                    f"  2. Check if database is corrupted\n"
                    f"  3. Verify you have enough RAM available"
                )

        print()  # New line after progress

        export_duration = time.time() - start_time
        print(f"  [OK] Exported {exported_count:,} documents in {export_duration:.2f}s")

        # Validate export
        print(f"\n[3/5] Validating export integrity...")

        if exported_count != num_documents:
            print(f"  [WARNING] Exported {exported_count} documents, expected {num_documents}")

        # Check for missing fields
        issues = []

        if len(all_embeddings) != exported_count:
            issues.append(f"Embeddings count mismatch: {len(all_embeddings)} != {exported_count}")

        if len(all_documents) != exported_count:
            issues.append(f"Documents count mismatch: {len(all_documents)} != {exported_count}")

        if len(all_metadatas) != exported_count:
            issues.append(f"Metadata count mismatch: {len(all_metadatas)} != {exported_count}")

        # Check for None values
        none_embeddings = sum(1 for e in all_embeddings if e is None)
        none_documents = sum(1 for d in all_documents if d is None)
        none_metadatas = sum(1 for m in all_metadatas if m is None)

        if none_embeddings > 0:
            issues.append(f"{none_embeddings} embeddings are None")

        if none_documents > 0:
            issues.append(f"{none_documents} documents are None")

        if none_metadatas > 0:
            issues.append(f"{none_metadatas} metadata entries are None")

        if issues:
            print("  [ERROR] Validation failed:")
            for issue in issues:
                print(f"    - {issue}")
            raise ValueError("Export validation failed. Data may be incomplete.")

        print(f"  [OK] All {exported_count:,} documents have complete data")

        # Get embedding dimension
        embedding_dim = len(all_embeddings[0]) if all_embeddings else 0
        print(f"  Embedding dimension: {embedding_dim}")

        # Save to files
        print(f"\n[4/5] Saving to files...")

        # Save embeddings as NumPy array
        embeddings_path = output_dir / "real_data_embeddings.npy"
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        np.save(embeddings_path, embeddings_array)
        print(f"  [OK] Saved embeddings: {embeddings_path.name} ({embeddings_array.nbytes / (1024**2):.2f} MB)")

        # Save metadata as JSON
        metadata_path = output_dir / "real_data_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadatas, f, indent=2)
        print(f"  [OK] Saved metadata: {metadata_path.name}")

        # Save documents as JSON
        documents_path = output_dir / "real_data_documents.json"
        with open(documents_path, 'w') as f:
            json.dump(all_documents, f, indent=2)
        print(f"  [OK] Saved documents: {documents_path.name}")

        # Save IDs for reference
        ids_path = output_dir / "real_data_ids.json"
        with open(ids_path, 'w') as f:
            json.dump(all_ids, f, indent=2)
        print(f"  [OK] Saved IDs: {ids_path.name}")

        # Save export metadata
        export_metadata = {
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chroma_path': str(self.chroma_path),
            'collection_name': self.collection.name,
            'num_documents': exported_count,
            'embedding_dimension': embedding_dim,
            'export_duration_seconds': export_duration,
            'files': {
                'embeddings': str(embeddings_path),
                'metadata': str(metadata_path),
                'documents': str(documents_path),
                'ids': str(ids_path)
            }
        }

        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(export_metadata, f, indent=2)
        print(f"  [OK] Saved export metadata: {metadata_path.name}")

        print(f"\n[5/5] Export Summary")
        print("="*60)
        print(f"  Documents exported: {exported_count:,}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Export duration: {export_duration:.2f}s")
        print(f"  Output directory: {output_dir}")
        print("="*60)

        return export_metadata


def main():
    """Main export function."""
    print("="*60)
    print("CHROMA DATABASE EXPORT - 5,000 Documents")
    print("="*60)

    # Get Chroma database path from user
    print("\nPlease provide your Chroma database path:")
    print("Example: C:/Users/YourName/chroma_data")
    print("Example: ./chroma_db")
    print()

    chroma_path = input("Chroma database path: ").strip()

    if not chroma_path:
        print("\n[ERROR] No path provided. Exiting.")
        sys.exit(1)

    # Optional: specify collection name
    print("\nCollection name (press Enter to auto-select first collection):")
    collection_name = input("Collection name: ").strip()
    collection_name = collection_name if collection_name else None

    try:
        # Initialize exporter
        exporter = ChromaExporter(chroma_path, collection_name)

        # Connect
        doc_count = exporter.connect()

        # Confirm export
        print(f"\nReady to export 5,000 documents from collection '{exporter.collection.name}'")
        confirm = input("Proceed with export? (yes/no): ").strip().lower()

        if confirm not in ['yes', 'y']:
            print("\nExport cancelled.")
            sys.exit(0)

        # Export
        output_dir = Path(__file__).parent
        metadata = exporter.export_sample(num_documents=5000, output_dir=output_dir)

        print("\n" + "="*60)
        print("[SUCCESS] Export completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run: python test_real_data.py")
        print("  2. Run: python compare_performance.py")
        print("  3. Run: python validate_real_data.py")

    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Export interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Export failed:")
        print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

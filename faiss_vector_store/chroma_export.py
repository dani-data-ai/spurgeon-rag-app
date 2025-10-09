"""
Phase 1: Export embeddings and metadata from Chroma to disk format.

Exports Chroma database in batches to NumPy arrays (embeddings) and
Parquet files (metadata) for Faiss migration.

Usage:
    python chroma_export.py --chroma-path ./chroma_db --output-dir ./export
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

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
    """Simple resource monitoring for RAM usage."""

    @staticmethod
    def get_ram_usage():
        """Get current RAM usage in bytes."""
        return psutil.virtual_memory().used

    @staticmethod
    def check_ram_limit():
        """Check if RAM usage is approaching limit."""
        used = ResourceMonitor.get_ram_usage()
        limit = config.MAX_RAM_BYTES

        usage_pct = (used / limit) * 100

        if usage_pct > 80:
            logger.warning(
                f"RAM usage high: {used / (1024**3):.2f} GiB / "
                f"{limit / (1024**3):.2f} GiB ({usage_pct:.1f}%)"
            )

        return used < limit


def export_chroma_collection(
    chroma_path: str,
    output_dir: str,
    collection_name: Optional[str] = None
) -> bool:
    """
    Export Chroma database to NumPy + Parquet format.

    Args:
        chroma_path: Path to Chroma database directory
        output_dir: Output directory for exported files
        collection_name: Optional specific collection to export

    Returns:
        True if successful, False otherwise
    """
    try:
        import chromadb
    except ImportError:
        logger.error(
            "chromadb not installed. "
            "Install with: pip install chromadb==0.4.15"
        )
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Connect to Chroma
    logger.info(f"Connecting to Chroma at: {chroma_path}")
    try:
        client = chromadb.PersistentClient(path=chroma_path)
    except Exception as e:
        logger.error(f"Failed to connect to Chroma: {e}")
        return False

    # Get collections
    try:
        collections = client.list_collections()
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return False

    if not collections:
        logger.error("No collections found in Chroma database")
        return False

    logger.info(f"Found {len(collections)} collection(s): {[c.name for c in collections]}")

    # Process each collection
    for coll_obj in collections:
        coll_name = coll_obj.name

        # Skip if specific collection requested and this isn't it
        if collection_name and coll_name != collection_name:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Exporting collection: {coll_name}")
        logger.info(f"{'='*60}")

        try:
            collection = client.get_collection(coll_name)
        except Exception as e:
            logger.error(f"Failed to get collection {coll_name}: {e}")
            continue

        # Get total count
        try:
            total_count = collection.count()
            logger.info(f"Total vectors: {total_count:,}")
        except Exception as e:
            logger.warning(f"Could not get count: {e}")
            total_count = None

        # Export in batches
        all_embeddings = []
        all_metadata = []
        offset = 0
        batch_num = 0

        with tqdm(total=total_count, desc=f"Exporting {coll_name}", unit="vec") as pbar:
            while True:
                # Check RAM before batch
                if not ResourceMonitor.check_ram_limit():
                    logger.error("RAM limit exceeded, stopping export")
                    return False

                try:
                    # Fetch batch - try different API patterns for compatibility
                    try:
                        # Newer Chroma API (0.4+)
                        results = collection.get(
                            limit=config.EXPORT_BATCH_SIZE,
                            offset=offset,
                            include=['embeddings', 'metadatas', 'documents']
                        )
                    except TypeError:
                        # Older Chroma API (no offset parameter)
                        if batch_num > 0:
                            logger.warning(
                                "Chroma version doesn't support pagination. "
                                "All data will be fetched at once."
                            )
                            results = collection.get(
                                include=['embeddings', 'metadatas', 'documents']
                            )
                        else:
                            results = collection.get(
                                include=['embeddings', 'metadatas', 'documents']
                            )

                    if not results or not results.get('ids'):
                        break  # No more data

                    batch_size = len(results['ids'])
                    logger.debug(f"Batch {batch_num}: {batch_size} vectors")

                    # Collect embeddings
                    embeddings = results.get('embeddings', [])
                    if embeddings:
                        all_embeddings.extend(embeddings)

                    # Collect metadata
                    for i, (doc_id, metadata, document) in enumerate(zip(
                        results['ids'],
                        results.get('metadatas', [{}] * batch_size),
                        results.get('documents', [''] * batch_size)
                    )):
                        # Create metadata row with sequential ID
                        seq_id = offset + i
                        meta_row = {
                            'id': seq_id,
                            'chroma_id': doc_id,
                            'document': document or '',
                            **(metadata or {})  # Unpack original metadata
                        }
                        all_metadata.append(meta_row)

                    offset += batch_size
                    batch_num += 1
                    pbar.update(batch_size)

                    # Break if batch smaller than limit (last batch)
                    if batch_size < config.EXPORT_BATCH_SIZE:
                        break

                    # For old API without pagination, break after first batch
                    if batch_num > 0:
                        try:
                            # Test if offset works
                            _ = collection.get(limit=1, offset=1)
                        except TypeError:
                            break

                except Exception as e:
                    logger.error(f"Error in batch {batch_num}: {e}")
                    break

        # Save embeddings
        if all_embeddings:
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            embeddings_file = output_path / f"{coll_name}_embeddings.npy"

            logger.info(f"Saving {len(all_embeddings):,} embeddings to {embeddings_file.name}")
            np.save(embeddings_file, embeddings_array)

            # Save metadata as Parquet
            if all_metadata:
                metadata_df = pd.DataFrame(all_metadata)
                metadata_file = output_path / f"{coll_name}_metadata.parquet"

                logger.info(f"Saving metadata to {metadata_file.name}")
                metadata_df.to_parquet(metadata_file, index=False, compression='snappy')

            logger.info(f"\n✓ Export complete for '{coll_name}':")
            logger.info(f"  Embeddings: {embeddings_file.name} (shape: {embeddings_array.shape})")
            logger.info(f"  Metadata: {metadata_file.name} ({len(all_metadata)} rows)")
            logger.info(f"  Dimension: {embeddings_array.shape[1]}")
        else:
            logger.warning(f"No embeddings found in collection '{coll_name}'")

    logger.info(f"\n{'='*60}")
    logger.info("✓ Export completed successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_path.absolute()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Chroma database to Faiss-compatible format"
    )
    parser.add_argument(
        '--chroma-path',
        type=str,
        required=True,
        help='Path to Chroma database directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./chroma_export',
        help='Output directory for exported files (default: ./chroma_export)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Specific collection name to export (optional, exports all if not specified)'
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.chroma_path).exists():
        logger.error(f"Chroma path does not exist: {args.chroma_path}")
        return 1

    success = export_chroma_collection(
        args.chroma_path,
        args.output_dir,
        args.collection
    )

    if success:
        logger.info("\n✓ Ready for Faiss migration!")
        logger.info("Next step: Use vector_store.py to build Faiss index from exported data")
        return 0
    else:
        logger.error("\n✗ Export failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

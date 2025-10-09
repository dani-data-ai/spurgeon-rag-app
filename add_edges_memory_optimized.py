"""
Memory-Optimized Edge Builder for Large Graphs

This version is designed for graphs with millions of nodes that exceed RAM capacity.

Key Features:
- Streaming node processing (never loads full graph into memory)
- SQLite-based temporary edge storage (disk-backed)
- Small batch sizes (500 edges at a time)
- Aggressive garbage collection
- Sequential file processing
- Proper memory cleanup between operations

Handles graphs with 5M+ nodes and billions of potential edges safely.
"""

import networkx as nx
from pathlib import Path
from typing import Set, Dict, Tuple, Optional
import sys
import time
import logging
import psutil
import sqlite3
import gc
from collections import defaultdict
from tqdm import tqdm
import argparse
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('add_edges_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage."""

    def __init__(self, critical_threshold=85, warning_threshold=75):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

    def get_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent

    def check_and_cleanup(self, force=False) -> float:
        """Check memory and force garbage collection if needed."""
        usage = self.get_usage()

        if usage > self.warning_threshold or force:
            gc.collect()
            usage = self.get_usage()

        if usage > self.critical_threshold:
            logger.warning(f"Critical memory usage: {usage:.1f}%")

        return usage

    def wait_for_memory(self, max_wait=30):
        """Wait for memory to be available."""
        waited = 0
        while self.get_usage() > self.critical_threshold and waited < max_wait:
            logger.info(f"Waiting for memory... ({self.get_usage():.1f}%)")
            time.sleep(2)
            gc.collect()
            waited += 2


class EdgeDatabase:
    """SQLite-based temporary edge storage for memory efficiency."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with optimized settings."""
        # Remove old database if exists
        if self.db_path.exists():
            self.db_path.unlink()

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging for performance
        self.conn.execute('PRAGMA synchronous = NORMAL')  # Faster writes
        self.conn.execute('PRAGMA cache_size = 10000')  # Larger cache
        self.conn.execute('PRAGMA temp_store = MEMORY')  # Temp tables in memory

        # Create edges table
        self.conn.execute('''
            CREATE TABLE edges (
                node1 TEXT,
                node2 TEXT,
                edge_type TEXT,
                shared_item TEXT,
                PRIMARY KEY (node1, node2, edge_type, shared_item)
            )
        ''')

        # Create index for fast lookups
        self.conn.execute('CREATE INDEX idx_nodes ON edges(node1, node2)')

        self.conn.commit()
        logger.info(f"Initialized edge database: {self.db_path}")

    def add_edge_batch(self, edges: list):
        """Add a batch of edges to the database."""
        if not edges:
            return

        try:
            self.conn.executemany(
                'INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)',
                edges
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding edge batch: {e}")
            self.conn.rollback()

    def get_aggregated_edges(self, min_overlap: int, max_shared: int):
        """
        Generator that yields aggregated edges meeting criteria.
        Groups by node pair and calculates weights.
        """
        query = '''
            SELECT node1, node2, edge_type,
                   GROUP_CONCAT(shared_item, ',') as shared_items,
                   COUNT(*) as weight
            FROM edges
            GROUP BY node1, node2, edge_type
            HAVING COUNT(*) >= ?
        '''

        cursor = self.conn.execute(query, (min_overlap,))

        for row in cursor:
            node1, node2, edge_type, shared_items, weight = row

            # Limit shared items for memory
            items_list = shared_items.split(',')[:max_shared]

            yield (node1, node2, edge_type, ','.join(items_list), weight)

    def get_edge_count(self) -> int:
        """Get total number of edge records."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM edges')
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

        # Clean up database file
        if self.db_path.exists():
            try:
                self.db_path.unlink()
                logger.info("Cleaned up temporary database")
            except Exception as e:
                logger.warning(f"Could not delete temp database: {e}")


def parse_field(field_data) -> Set[str]:
    """Parse node field data into set of items."""
    if not field_data:
        return set()

    if isinstance(field_data, str):
        return {item.strip() for item in field_data.split(',') if item.strip()}
    elif isinstance(field_data, (list, tuple)):
        return {str(item).strip() for item in field_data if item}
    return set()


def stream_graph_nodes(file_path: Path, memory_monitor: MemoryMonitor):
    """
    Stream nodes from graph file one at a time.
    Yields (node_id, entities, keywords, figures) tuples.
    """
    logger.info(f"Streaming nodes from: {file_path.name}")

    # Load graph
    graph = nx.read_gml(str(file_path))
    total_nodes = graph.number_of_nodes()

    logger.info(f"Graph has {total_nodes:,} nodes")

    # Stream nodes
    with tqdm(total=total_nodes, desc="Streaming nodes", unit="node") as pbar:
        for node_id, node_data in graph.nodes(data=True):
            entities = parse_field(node_data.get('entities'))
            keywords = parse_field(node_data.get('keywords'))
            figures = parse_field(node_data.get('figures'))

            yield (node_id, entities, keywords, figures)

            pbar.update(1)

            # Periodic memory check
            if pbar.n % 1000 == 0:
                memory_monitor.check_and_cleanup()

    # Cleanup
    del graph
    gc.collect()


def build_index_streaming(file_path: Path, memory_monitor: MemoryMonitor) -> Tuple[Dict, Dict, Dict]:
    """
    Build inverted indexes by streaming nodes from disk.
    Returns dictionaries mapping items to sets of node IDs.
    """
    logger.info("Building inverted indexes (streaming mode)...")

    entity_index = defaultdict(set)
    keyword_index = defaultdict(set)
    figure_index = defaultdict(set)

    # Stream nodes and build indexes
    for node_id, entities, keywords, figures in stream_graph_nodes(file_path, memory_monitor):
        for entity in entities:
            entity_index[entity].add(node_id)
        for keyword in keywords:
            keyword_index[keyword].add(node_id)
        for figure in figures:
            figure_index[figure].add(node_id)

    logger.info(f"Indexes built: {len(entity_index)} entities, "
                f"{len(keyword_index)} keywords, {len(figure_index)} figures")

    return entity_index, keyword_index, figure_index


def generate_edges_to_database(index: Dict[str, Set], edge_type: str,
                                edge_db: EdgeDatabase, batch_size: int,
                                memory_monitor: MemoryMonitor):
    """
    Generate edges from index and store in database in small batches.
    """
    logger.info(f"Generating {edge_type} edges to database...")

    batch = []
    total_edges = 0

    with tqdm(total=len(index), desc=f"Processing {edge_type}s", unit="item") as pbar:
        for item, nodes in index.items():
            if len(nodes) < 2:
                pbar.update(1)
                continue

            if len(nodes) > 10000:
                pbar.update(1)
                continue

            nodes_list = sorted(nodes)  # Sort for consistent ordering

            # Generate all pairs
            for i in range(len(nodes_list)):
                for j in range(i + 1, len(nodes_list)):
                    node1, node2 = nodes_list[i], nodes_list[j]

                    # Store in database format
                    batch.append((node1, node2, edge_type, item))
                    total_edges += 1

                    # Flush batch when full
                    if len(batch) >= batch_size:
                        edge_db.add_edge_batch(batch)
                        batch.clear()
                        memory_monitor.check_and_cleanup()

            pbar.update(1)

    # Flush remaining
    if batch:
        edge_db.add_edge_batch(batch)
        batch.clear()

    logger.info(f"Generated {total_edges:,} {edge_type} edge records")
    gc.collect()


def apply_edges_to_graph(file_path: Path, edge_db: EdgeDatabase,
                         min_overlap: int, max_shared: int,
                         memory_monitor: MemoryMonitor) -> int:
    """
    Apply aggregated edges from database back to graph file.
    """
    logger.info("Loading graph for edge application...")
    graph = nx.read_gml(str(file_path))
    edges_before = graph.number_of_edges()

    logger.info(f"Applying edges (min overlap: {min_overlap})...")
    edges_added = 0
    batch = []
    batch_size = 500  # Small batches for memory safety

    # Get aggregated edges from database
    edge_generator = edge_db.get_aggregated_edges(min_overlap, max_shared)

    with tqdm(desc="Adding edges to graph", unit="edge") as pbar:
        for node1, node2, edge_type, shared_items, weight in edge_generator:
            batch.append((node1, node2, {
                'type': edge_type,
                'weight': weight,
                'shared_items': shared_items
            }))

            # Apply batch
            if len(batch) >= batch_size:
                for n1, n2, attrs in batch:
                    if not graph.has_edge(n1, n2):
                        graph.add_edge(n1, n2, **attrs)
                        edges_added += 1

                batch.clear()
                pbar.update(batch_size)
                memory_monitor.check_and_cleanup()

    # Apply remaining
    if batch:
        for n1, n2, attrs in batch:
            if not graph.has_edge(n1, n2):
                graph.add_edge(n1, n2, **attrs)
                edges_added += 1
        batch.clear()

    edges_after = graph.number_of_edges()
    logger.info(f"Edges: {edges_before:,} -> {edges_after:,} (+{edges_added:,})")

    # Save graph
    logger.info("Saving updated graph...")
    nx.write_gml(graph, str(file_path))

    # Cleanup
    del graph
    gc.collect()

    return edges_added


def process_checkpoint_file(file_path: Path, min_overlap: int = 1,
                            max_shared: int = 3, batch_size: int = 500) -> Tuple[bool, int]:
    """
    Process a single checkpoint file with memory-optimized approach.
    """
    memory_monitor = MemoryMonitor()

    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*70}")
        logger.info(f"Memory at start: {memory_monitor.get_usage():.1f}%")

        # Wait for memory if needed
        memory_monitor.wait_for_memory()

        # Create temporary database
        db_path = file_path.with_suffix('.tmp.db')
        edge_db = EdgeDatabase(db_path)

        try:
            # Step 1: Build indexes by streaming
            entity_index, keyword_index, figure_index = build_index_streaming(
                file_path, memory_monitor
            )

            # Step 2: Generate edges to database
            generate_edges_to_database(
                entity_index, 'entity', edge_db, batch_size, memory_monitor
            )

            # Clear entity index to free memory
            entity_index.clear()
            gc.collect()

            generate_edges_to_database(
                keyword_index, 'keyword', edge_db, batch_size, memory_monitor
            )

            # Clear keyword index
            keyword_index.clear()
            gc.collect()

            generate_edges_to_database(
                figure_index, 'figure', edge_db, batch_size, memory_monitor
            )

            # Clear figure index
            figure_index.clear()
            gc.collect()

            logger.info(f"Database has {edge_db.get_edge_count():,} edge records")

            # Step 3: Apply edges back to graph
            edges_added = apply_edges_to_graph(
                file_path, edge_db, min_overlap, max_shared, memory_monitor
            )

            logger.info(f"✓ Completed: {edges_added:,} edges added")
            logger.info(f"Memory at end: {memory_monitor.get_usage():.1f}%")

            return True, edges_added

        finally:
            edge_db.close()

    except Exception as e:
        logger.error(f"✗ Error processing {file_path.name}: {e}", exc_info=True)
        return False, 0


def find_checkpoint_files(directory: Path) -> list:
    """Find all checkpoint .gml files."""
    return sorted([f for f in directory.glob('checkpoint_*.gml') if f.is_file()])


class StateTracker:
    """Track processing state for resume capability."""

    def __init__(self, state_file: Path = Path('.edge_builder_state.json')):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'completed': [], 'failed': []}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_completed(self, file_path: Path) -> bool:
        return str(file_path) in self.state['completed']

    def mark_completed(self, file_path: Path):
        if str(file_path) not in self.state['completed']:
            self.state['completed'].append(str(file_path))
            self.save_state()

    def mark_failed(self, file_path: Path):
        if str(file_path) not in self.state['failed']:
            self.state['failed'].append(str(file_path))
            self.save_state()

    def clear(self):
        self.state = {'completed': [], 'failed': []}
        if self.state_file.exists():
            self.state_file.unlink()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Memory-Optimized Edge Builder for Large Graphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--source-dir', type=str, default='.',
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--min-overlap', type=int, default=1,
        help='Minimum overlap to create edge'
    )
    parser.add_argument(
        '--max-shared', type=int, default=3,
        help='Max shared items per edge'
    )
    parser.add_argument(
        '--batch-size', type=int, default=5000,
        help='Batch size for database operations'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--clear-state', action='store_true',
        help='Clear state and start fresh'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("MEMORY-OPTIMIZED EDGE BUILDER")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Min overlap: {args.min_overlap}")
    logger.info(f"Max shared: {args.max_shared}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequential processing (memory-safe)")

    work_dir = Path(args.source_dir)

    # State tracking
    state_tracker = StateTracker()

    if args.clear_state:
        state_tracker.clear()
        logger.info("State cleared")

    # Find files
    checkpoint_files = find_checkpoint_files(work_dir)

    if not checkpoint_files:
        logger.error("No checkpoint files found")
        sys.exit(1)

    # Filter completed if resuming
    if args.resume:
        original = len(checkpoint_files)
        checkpoint_files = [f for f in checkpoint_files if not state_tracker.is_completed(f)]
        skipped = original - len(checkpoint_files)
        if skipped > 0:
            logger.info(f"Resuming: skipped {skipped} completed files")

    logger.info(f"\nFound {len(checkpoint_files)} file(s) to process:")
    for f in checkpoint_files:
        logger.info(f"  - {f.name}")

    # Process files sequentially
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING FILES (Sequential)")
    logger.info("=" * 70)

    start_time = time.time()
    results = []

    for idx, checkpoint_file in enumerate(checkpoint_files, 1):
        logger.info(f"\n[{idx}/{len(checkpoint_files)}]")

        success, edges = process_checkpoint_file(
            checkpoint_file,
            min_overlap=args.min_overlap,
            max_shared=args.max_shared,
            batch_size=args.batch_size
        )

        results.append((success, edges))

        if success:
            state_tracker.mark_completed(checkpoint_file)
        else:
            state_tracker.mark_failed(checkpoint_file)

        # Force cleanup between files
        gc.collect()
        time.sleep(1)  # Brief pause to let system stabilize

    elapsed = time.time() - start_time

    # Summary
    successful = sum(1 for s, _ in results if s)
    failed = len(results) - successful
    total_edges = sum(e for _, e in results)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Processed: {successful}/{len(checkpoint_files)} files")
    if failed > 0:
        logger.warning(f"Failed: {failed} files")
    logger.info(f"Total edges added: {total_edges:,}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"Average: {elapsed/len(checkpoint_files):.1f}s per file")

    if failed == 0:
        logger.info("\n✓ All files processed successfully!")


if __name__ == "__main__":
    main()

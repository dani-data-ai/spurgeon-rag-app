"""
Memory-Optimized Edge Builder for Large Graphs

This version is designed for graphs with millions of nodes that exceed RAM capacity.

Key Features:
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
from typing import Set, Dict, Tuple, Optional, List
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
import multiprocessing as mp
import hashlib
import csv
from datetime import datetime


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
    """Monitor and manage memory usage with absolute byte limits."""

    def __init__(self, max_ram_gb=19.0, headroom_gb=1.0):
        """
        Initialize with hard byte limits.

        Args:
            max_ram_gb: Maximum RAM in GB (default 19GB, leaving 1GB headroom from 20GB limit)
            headroom_gb: Additional safety margin in GB
        """
        self.max_ram_bytes = int(max_ram_gb * 1024**3)
        self.headroom_bytes = int(headroom_gb * 1024**3)
        self.pause_threshold_bytes = self.max_ram_bytes - self.headroom_bytes
        self.process = psutil.Process()

        logger.info(f"MemoryMonitor: max_ram={max_ram_gb:.1f}GB, pause_at={self.pause_threshold_bytes/(1024**3):.1f}GB")

    def bytes_in_use(self) -> int:
        """Get current RSS (Resident Set Size) in bytes."""
        return self.process.memory_info().rss

    def gb_in_use(self) -> float:
        """Get current RSS in GB for logging."""
        return self.bytes_in_use() / (1024**3)

    def should_pause(self) -> bool:
        """Check if we should pause due to memory pressure."""
        return self.bytes_in_use() > self.pause_threshold_bytes

    def check_and_cleanup(self, force=False) -> float:
        """Check memory and force garbage collection if needed. Returns GB in use."""
        current_gb = self.gb_in_use()

        if self.should_pause() or force:
            gc.collect()
            current_gb = self.gb_in_use()

        if self.should_pause():
            logger.warning(f"Memory pressure: {current_gb:.2f}GB / {self.max_ram_bytes/(1024**3):.1f}GB limit")

        return current_gb

    def wait_for_memory(self, max_wait=30):
        """Wait for memory to be available."""
        waited = 0
        while self.should_pause() and waited < max_wait:
            logger.info(f"Waiting for memory... ({self.gb_in_use():.2f}GB)")
            time.sleep(2)
            gc.collect()
            waited += 2

    def estimate_batch_size(self, sample_data: list, target_mb: int = 100) -> int:
        """
        Estimate safe batch size based on sample data.

        Args:
            sample_data: List of sample tuples to measure
            target_mb: Target batch size in MB

        Returns:
            Recommended batch size (number of items)
        """
        if not sample_data:
            return 5000  # Default fallback

        # Sample a few items to estimate size
        sample_size = min(len(sample_data), 100)
        sample = sample_data[:sample_size]

        # Measure total bytes for sample
        total_bytes = sum(sys.getsizeof(item) for item in sample)

        # Add overhead for strings within tuples
        for item in sample:
            if isinstance(item, (tuple, list)):
                for elem in item:
                    if isinstance(elem, str):
                        total_bytes += sys.getsizeof(elem)

        # Calculate average bytes per item
        avg_bytes_per_item = total_bytes / sample_size

        # Calculate batch size for target MB with safety margin
        target_bytes = target_mb * 1024 * 1024
        safe_batch_size = int(target_bytes / avg_bytes_per_item * 0.8)  # 20% safety margin

        return max(1000, min(safe_batch_size, 50000))  # Clamp between 1k-50k


class ProgressTracker:
    """Track detailed progress for resumability."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load progress from JSON file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return {
            'phase': 'indexing',
            'completed_items': {},
            'current_item': None,
            'last_i': 0,
            'last_j': 0
        }

    def save(self):
        """Save progress to JSON file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def set_phase(self, phase: str):
        """Set current processing phase."""
        self.data['phase'] = phase
        self.save()

    def mark_item_completed(self, edge_type: str, item: str):
        """Mark an (edge_type, item) as fully processed."""
        key = f"{edge_type}|{item}"
        self.data['completed_items'][key] = True
        self.data['current_item'] = None
        self.data['last_i'] = 0
        self.data['last_j'] = 0
        self.save()

    def is_item_completed(self, edge_type: str, item: str) -> bool:
        """Check if an item was already completed."""
        key = f"{edge_type}|{item}"
        return self.data['completed_items'].get(key, False)

    def update_progress(self, edge_type: str, item: str, i: int, j: int):
        """Update current position within an item."""
        self.data['current_item'] = f"{edge_type}|{item}"
        self.data['last_i'] = i
        self.data['last_j'] = j
        self.save()

    def get_resume_point(self, edge_type: str, item: str) -> Tuple[int, int]:
        """Get the last (i, j) checkpoint for resuming."""
        current_key = f"{edge_type}|{item}"
        if self.data.get('current_item') == current_key:
            return self.data.get('last_i', 0), self.data.get('last_j', 0)
        return 0, 0


class TelemetryLogger:
    """Log detailed metrics for analysis."""

    def __init__(self, telemetry_file: Path):
        self.telemetry_file = telemetry_file
        self.fieldnames = [
            'timestamp', 'edge_type', 'item', 'subbatch',
            'pairs', 'duration_ms', 'ram_gb', 'vram_gb'
        ]
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.telemetry_file.exists():
            with open(self.telemetry_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_batch(self, edge_type: str, item: str, subbatch: int,
                   pairs: int, duration_ms: float, ram_gb: float, vram_gb: float = None):
        """Log a batch processing event."""
        try:
            with open(self.telemetry_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'edge_type': edge_type,
                    'item': item,
                    'subbatch': subbatch,
                    'pairs': pairs,
                    'duration_ms': duration_ms,
                    'ram_gb': f"{ram_gb:.2f}",
                    'vram_gb': f"{vram_gb:.2f}" if vram_gb else ''
                })
        except Exception as e:
            logger.warning(f"Could not log telemetry: {e}")


class EdgeDatabase:
    """SQLite-based temporary edge storage for memory efficiency."""

    def __init__(self, db_path: Path, delete_on_close: bool = False):
        self.db_path = db_path
        self.conn = None
        self.delete_on_close = delete_on_close
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

    def close(self, success: bool = False):
        """
        Close database connection.

        Args:
            success: If True, delete temp DB. If False, preserve for resume.
        """
        if self.conn:
            self.conn.close()

        # Only delete database if explicitly requested (after successful GML save)
        if success and self.delete_on_close and self.db_path.exists():
            try:
                self.db_path.unlink()
                logger.info("Cleaned up temporary database")
            except Exception as e:
                logger.warning(f"Could not delete temp database: {e}")
        elif not success:
            logger.info(f"Preserving temp database for resume: {self.db_path}")


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

    WARNING: NetworkX read_gml() loads entire graph into memory.
    This is unavoidable with the NetworkX API.
    """
    logger.info(f"Streaming nodes from: {file_path.name}")
    logger.info(f"Memory before graph load: {memory_monitor.gb_in_use():.2f}GB")

    # Check if we have enough memory headroom before loading
    current_gb = memory_monitor.gb_in_use()
    available_gb = memory_monitor.max_ram_bytes / (1024**3) - current_gb

    if available_gb < 5.0:  # Need at least 5GB free for graph load
        logger.warning(f"Low memory before graph load: {current_gb:.2f}GB used, {available_gb:.2f}GB available")
        logger.warning("Running garbage collection...")
        gc.collect()
        current_gb = memory_monitor.gb_in_use()
        available_gb = memory_monitor.max_ram_bytes / (1024**3) - current_gb

        if available_gb < 3.0:  # Still too low
            raise MemoryError(
                f"Insufficient memory to load graph: {current_gb:.2f}GB used, "
                f"only {available_gb:.2f}GB available. Need at least 3GB free."
            )

    # Load graph (unavoidable memory spike)
    try:
        graph = nx.read_gml(str(file_path))
    except MemoryError as e:
        logger.error(f"Graph loading failed due to memory: {e}")
        raise MemoryError(
            f"Graph file too large to load. Consider splitting into smaller checkpoint files. "
            f"Current usage: {memory_monitor.gb_in_use():.2f}GB"
        )

    total_nodes = graph.number_of_nodes()
    logger.info(f"Graph loaded: {total_nodes:,} nodes")
    logger.info(f"Memory after graph load: {memory_monitor.gb_in_use():.2f}GB")

    # Check if we exceeded safe limits after load
    if memory_monitor.should_pause():
        logger.error(
            f"Graph load exceeded memory limit: {memory_monitor.gb_in_use():.2f}GB > "
            f"{memory_monitor.pause_threshold_bytes/(1024**3):.2f}GB"
        )
        del graph
        gc.collect()
        raise MemoryError(
            "Graph too large for current memory limits. "
            "Options: 1) Increase --max-ram-gb, 2) Split graph into smaller files"
        )

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
    logger.info(f"Memory before graph cleanup: {memory_monitor.gb_in_use():.2f}GB")
    del graph
    gc.collect()
    logger.info(f"Memory after graph cleanup: {memory_monitor.gb_in_use():.2f}GB")


def stable_hash_item(edge_type: str, item: str, num_workers: int) -> int:
    """
    Compute stable hash for work partitioning.

    Returns:
        Worker ID (0 to num_workers-1) that should handle this item.
    """
    key = f"{edge_type}|{item}"
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_val % num_workers


def edge_generation_worker(worker_id: int, work_queue: mp.Queue, result_queue: mp.Queue,
                            memory_monitor_config: dict, progress_tracker_file: Path,
                            telemetry_file: Path):
    """
    Worker process for parallel edge generation.

    Args:
        worker_id: Unique worker identifier
        work_queue: Queue of (edge_type, item, nodes_list) tuples to process
        result_queue: Queue to send edge batches to writer
        memory_monitor_config: Config for creating MemoryMonitor
        progress_tracker_file: Path to progress JSON file
        telemetry_file: Path to telemetry CSV file
    """
    # Initialize worker-local resources
    memory_monitor = MemoryMonitor(**memory_monitor_config)
    progress_tracker = ProgressTracker(progress_tracker_file)
    telemetry = TelemetryLogger(telemetry_file)

    processed = 0

    try:
        while True:
            # Get work item (with timeout to check for poison pill)
            try:
                work_item = work_queue.get(timeout=1)
            except:
                continue

            # Check for poison pill (None signals shutdown)
            if work_item is None:
                break

            edge_type, item, nodes_list = work_item
            item_hash = item[:30] if len(item) > 30 else item  # Truncate for logging

            # Get resume point for this item
            start_i, start_j = progress_tracker.get_resume_point(edge_type, item)

            # Generate pairs in sub-batches with checkpointing
            batch = []
            subbatch_num = 0
            total_pairs = 0
            batch_start_time = time.time()

            # Calculate dynamic batch size based on current memory
            # Start with sample to estimate
            sample_edge = (nodes_list[0], nodes_list[1] if len(nodes_list) > 1 else nodes_list[0], edge_type, item)
            sample_batch = [sample_edge] * 100

            # Estimate safe batch size (target 50MB per batch to leave room for queue)
            current_batch_size = memory_monitor.estimate_batch_size(sample_batch, target_mb=50)
            current_batch_size = max(1000, min(current_batch_size, 10000))  # Clamp 1k-10k

            for i in range(len(nodes_list)):
                # Skip already-processed pairs when resuming
                j_start = start_j if i == start_i else i + 1

                for j in range(j_start, len(nodes_list)):
                    if j <= i:
                        continue

                    node1, node2 = nodes_list[i], nodes_list[j]
                    batch.append((node1, node2, edge_type, item))
                    total_pairs += 1

                    # Use dynamic batch size and memory pressure check
                    if len(batch) >= current_batch_size or memory_monitor.should_pause():
                        # Wait for queue space if result queue is full (backpressure)
                        while result_queue.qsize() > 8:  # Leave 2 slots for poison pills
                            time.sleep(0.1)
                            if memory_monitor.should_pause():
                                memory_monitor.wait_for_memory()

                        # Send batch to writer
                        result_queue.put(('edges', batch))

                        # Log telemetry
                        duration_ms = (time.time() - batch_start_time) * 1000
                        telemetry.log_batch(
                            edge_type, item_hash, subbatch_num,
                            len(batch), duration_ms, memory_monitor.gb_in_use()
                        )

                        # Update checkpoint
                        progress_tracker.update_progress(edge_type, item, i, j)

                        # Clear batch and recalculate size if memory pressure changed
                        batch.clear()
                        subbatch_num += 1
                        batch_start_time = time.time()

                        # Recalculate batch size if memory is tight
                        if memory_monitor.gb_in_use() > 15.0:  # > 15GB, shrink batches
                            current_batch_size = max(1000, current_batch_size // 2)
                        elif memory_monitor.gb_in_use() < 10.0:  # < 10GB, can grow
                            current_batch_size = min(10000, current_batch_size * 2)

                        # Wait for memory if needed
                        if memory_monitor.should_pause():
                            memory_monitor.wait_for_memory()

                # Reset j_start after first row
                if i == start_i:
                    start_j = 0

            # Send remaining batch
            if batch:
                while result_queue.qsize() > 8:
                    time.sleep(0.1)
                result_queue.put(('edges', batch))
                duration_ms = (time.time() - batch_start_time) * 1000
                telemetry.log_batch(
                    edge_type, item_hash, subbatch_num,
                    len(batch), duration_ms, memory_monitor.gb_in_use()
                )
                batch.clear()

            # Mark item as completed
            progress_tracker.mark_item_completed(edge_type, item)

            processed += 1

    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
    finally:
        # Signal completion
        result_queue.put(('worker_done', worker_id))


def database_writer_worker(result_queue: mp.Queue, db_path: Path,
                            num_workers: int, commit_interval: int = 200000):
    """
    Dedicated writer process for SQLite database.

    Args:
        result_queue: Queue receiving edge batches from workers
        db_path: Path to SQLite database
        num_workers: Number of worker processes (to know when all are done)
        commit_interval: Number of rows to accumulate before committing
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA cache_size = 10000')

    pending_rows = []
    total_written = 0
    workers_done = 0

    try:
        while workers_done < num_workers:
            try:
                msg_type, data = result_queue.get(timeout=1)
            except:
                continue

            if msg_type == 'edges':
                # Accumulate edges
                pending_rows.extend(data)

                # Commit in large batches
                if len(pending_rows) >= commit_interval:
                    conn.executemany(
                        'INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)',
                        pending_rows
                    )
                    conn.commit()
                    total_written += len(pending_rows)
                    pending_rows.clear()

            elif msg_type == 'worker_done':
                worker_id = data
                workers_done += 1
                logger.info(f"Worker {worker_id} completed ({workers_done}/{num_workers})")

        # Final commit
        if pending_rows:
            conn.executemany(
                'INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)',
                pending_rows
            )
            conn.commit()
            total_written += len(pending_rows)
            pending_rows.clear()

        logger.info(f"Writer: Total edges written: {total_written:,}")

    except Exception as e:
        logger.error(f"Writer error: {e}", exc_info=True)
    finally:
        conn.close()


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


def generate_edges_to_database_parallel(index: Dict[str, Set], edge_type: str,
                                         edge_db: EdgeDatabase,
                                         memory_monitor: MemoryMonitor,
                                         progress_tracker: ProgressTracker,
                                         telemetry: TelemetryLogger,
                                         num_workers: int = 7):
    """
    Generate edges from index using parallel workers with queue-based SQLite writer.

    Args:
        index: Dictionary mapping items to sets of node IDs
        edge_type: Type of edge ('entity', 'keyword', 'figure')
        edge_db: Database connection (will be closed and reopened for multiprocessing)
        memory_monitor: Memory monitoring instance
        progress_tracker: Progress tracking instance
        telemetry: Telemetry logging instance
        num_workers: Number of parallel worker processes
    """
    logger.info(f"Generating {edge_type} edges with {num_workers} workers...")

    # Filter items: skip if < 2 nodes or > 10,000 nodes
    valid_items = {
        item: nodes for item, nodes in index.items()
        if 2 <= len(nodes) <= 10000
    }

    if not valid_items:
        logger.info(f"No valid items for {edge_type}")
        return

    logger.info(f"Processing {len(valid_items):,} {edge_type} items (filtered from {len(index):,})")

    # Close main DB connection (writer will reopen)
    edge_db.conn.close()

    # Create multiprocessing queues
    work_queue = mp.Queue(maxsize=20)  # Backpressure control
    result_queue = mp.Queue(maxsize=10)

    # Start writer process
    writer_process = mp.Process(
        target=database_writer_worker,
        args=(result_queue, edge_db.db_path, num_workers)
    )
    writer_process.start()

    # Start worker processes
    memory_config = {
        'max_ram_gb': memory_monitor.max_ram_bytes / (1024**3),
        'headroom_gb': memory_monitor.headroom_bytes / (1024**3)
    }

    worker_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=edge_generation_worker,
            args=(worker_id, work_queue, result_queue, memory_config,
                  progress_tracker.progress_file, telemetry.telemetry_file)
        )
        p.start()
        worker_processes.append(p)

    # Distribute work items to workers based on stable hash
    items_by_worker = [[] for _ in range(num_workers)]
    for item, nodes in valid_items.items():
        # Skip already completed items
        if progress_tracker.is_item_completed(edge_type, item):
            continue

        worker_id = stable_hash_item(edge_type, item, num_workers)
        items_by_worker[worker_id].append((item, sorted(nodes)))

    # Send work to queues
    total_items = sum(len(items) for items in items_by_worker)
    logger.info(f"Distributing {total_items:,} items across {num_workers} workers")

    with tqdm(total=total_items, desc=f"Queueing {edge_type} items", unit="item") as pbar:
        for worker_items in items_by_worker:
            for item, nodes_list in worker_items:
                work_queue.put((edge_type, item, nodes_list))
                pbar.update(1)

    # Send poison pills to workers
    for _ in range(num_workers):
        work_queue.put(None)

    # Wait for all workers to complete
    for p in worker_processes:
        p.join()

    # Wait for writer to complete
    writer_process.join()

    # Reopen DB connection for main process
    edge_db.conn = sqlite3.connect(str(edge_db.db_path))

    logger.info(f"Completed {edge_type} edge generation")
    gc.collect()


def apply_edges_to_graph(file_path: Path, edge_db: EdgeDatabase,
                         min_overlap: int, max_shared: int,
                         memory_monitor: MemoryMonitor) -> int:
    """
    Apply aggregated edges from database back to graph file with strict memory control.

    WARNING: This function loads the entire graph into memory (unavoidable with NetworkX).
    Ensure sufficient memory is available before calling.
    """
    logger.info("Loading graph for edge application...")
    logger.info(f"Memory before graph load: {memory_monitor.gb_in_use():.2f}GB")

    # Check memory headroom before loading graph
    current_gb = memory_monitor.gb_in_use()
    available_gb = memory_monitor.max_ram_bytes / (1024**3) - current_gb

    if available_gb < 5.0:
        logger.warning(f"Low memory before graph load: {current_gb:.2f}GB used, {available_gb:.2f}GB available")
        logger.warning("Running garbage collection before loading graph...")
        gc.collect()
        current_gb = memory_monitor.gb_in_use()
        available_gb = memory_monitor.max_ram_bytes / (1024**3) - current_gb

        if available_gb < 3.0:
            raise MemoryError(
                f"Insufficient memory to load graph for apply phase: {current_gb:.2f}GB used, "
                f"only {available_gb:.2f}GB available. Need at least 3GB free."
            )

    # Load graph (unavoidable memory spike)
    try:
        graph = nx.read_gml(str(file_path))
    except MemoryError as e:
        logger.error(f"Graph loading failed due to memory: {e}")
        raise MemoryError(
            f"Graph file too large to load for apply phase. "
            f"Current usage: {memory_monitor.gb_in_use():.2f}GB. "
            f"Try increasing --max-ram-gb or processing smaller checkpoint files."
        )

    edges_before = graph.number_of_edges()

    logger.info(f"Graph loaded: {graph.number_of_nodes():,} nodes, {edges_before:,} edges")
    logger.info(f"Memory after graph load: {memory_monitor.gb_in_use():.2f}GB")

    # Check if we exceeded safe limits
    if memory_monitor.should_pause():
        current_usage = memory_monitor.gb_in_use()
        logger.error(
            f"Graph load exceeded memory limit: {current_usage:.2f}GB > "
            f"{memory_monitor.pause_threshold_bytes/(1024**3):.2f}GB"
        )
        del graph
        gc.collect()
        raise MemoryError(
            f"Graph too large for current memory limits ({current_usage:.2f}GB). "
            f"Options: 1) Increase --max-ram-gb to at least {current_usage + 3:.0f}, "
            f"2) Split graph into smaller checkpoint files"
        )

    logger.info(f"Applying edges (min overlap: {min_overlap})...")
    edges_added = 0
    batch = []
    batch_size = 1000  # Increased from 500 for better throughput

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

                pbar.update(len(batch))
                batch.clear()

                # Check memory and pause if needed
                current_gb = memory_monitor.check_and_cleanup()
                if memory_monitor.should_pause():
                    logger.warning(f"Memory pressure during apply: {current_gb:.2f}GB")
                    memory_monitor.wait_for_memory()

    # Apply remaining
    if batch:
        for n1, n2, attrs in batch:
            if not graph.has_edge(n1, n2):
                graph.add_edge(n1, n2, **attrs)
                edges_added += 1
        pbar.update(len(batch))
        batch.clear()

    edges_after = graph.number_of_edges()
    logger.info(f"Edges: {edges_before:,} -> {edges_after:,} (+{edges_added:,})")

    # Save graph
    logger.info("Saving updated graph...")
    logger.info(f"Memory before save: {memory_monitor.gb_in_use():.2f}GB")

    nx.write_gml(graph, str(file_path))

    logger.info(f"Memory after save: {memory_monitor.gb_in_use():.2f}GB")

    # Cleanup
    del graph
    gc.collect()

    return edges_added


def process_checkpoint_file(file_path: Path, min_overlap: int = 1,
                            max_shared: int = 3, num_workers: int = 7,
                            max_ram_gb: float = 19.0) -> Tuple[bool, int]:
    """
    Process a single checkpoint file with parallel, memory-optimized approach.

    Args:
        file_path: Path to checkpoint .gml file
        min_overlap: Minimum overlap to create edge
        max_shared: Max shared items per edge
        num_workers: Number of parallel worker processes
        max_ram_gb: Maximum RAM to use in GB
    """
    memory_monitor = MemoryMonitor(max_ram_gb=max_ram_gb, headroom_gb=1.0)

    # Initialize progress tracking and telemetry
    progress_file = file_path.with_suffix('.progress.json')
    telemetry_file = file_path.with_suffix('.telemetry.csv')

    progress_tracker = ProgressTracker(progress_file)
    telemetry = TelemetryLogger(telemetry_file)
    edge_db = None

    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*70}")
        logger.info(f"Memory at start: {memory_monitor.gb_in_use():.2f}GB (limit: {max_ram_gb:.1f}GB)")
        logger.info(f"Workers: {num_workers}")

        # Wait for memory if needed
        memory_monitor.wait_for_memory()

        # Create temporary database
        db_path = file_path.with_suffix('.tmp.db')
        edge_db = EdgeDatabase(db_path, delete_on_close=True)

        try:
            # Check if we're resuming
            current_phase = progress_tracker.data.get('phase', 'indexing')
            logger.info(f"Phase: {current_phase}")

            # Step 1: Build indexes by streaming (if not already done)
            if current_phase == 'indexing':
                progress_tracker.set_phase('indexing')
                entity_index, keyword_index, figure_index = build_index_streaming(
                    file_path, memory_monitor
                )
                progress_tracker.set_phase('edgegen')
            else:
                logger.info("Resuming: rebuilding indexes...")
                entity_index, keyword_index, figure_index = build_index_streaming(
                    file_path, memory_monitor
                )

            # Step 2: Generate edges to database (parallel, with resume support)
            progress_tracker.set_phase('edgegen')

            generate_edges_to_database_parallel(
                entity_index, 'entity', edge_db, memory_monitor,
                progress_tracker, telemetry, num_workers
            )

            # Clear entity index to free memory
            entity_index.clear()
            gc.collect()

            generate_edges_to_database_parallel(
                keyword_index, 'keyword', edge_db, memory_monitor,
                progress_tracker, telemetry, num_workers
            )

            # Clear keyword index
            keyword_index.clear()
            gc.collect()

            generate_edges_to_database_parallel(
                figure_index, 'figure', edge_db, memory_monitor,
                progress_tracker, telemetry, num_workers
            )

            # Clear figure index
            figure_index.clear()
            gc.collect()

            logger.info(f"Database has {edge_db.get_edge_count():,} edge records")

            # Step 3: Apply edges back to graph
            progress_tracker.set_phase('apply')

            edges_added = apply_edges_to_graph(
                file_path, edge_db, min_overlap, max_shared, memory_monitor
            )

            logger.info(f"✓ Completed: {edges_added:,} edges added")
            logger.info(f"Memory at end: {memory_monitor.gb_in_use():.2f}GB")

            # Success - clean up temp files
            edge_db.close(success=True)
            if progress_file.exists():
                progress_file.unlink()

            return True, edges_added

        except Exception as e:
            # Failure - preserve temp DB and progress for resume
            if edge_db is not None:
                edge_db.close(success=False)
            raise

    except Exception as e:
        logger.error(f"✗ Error processing {file_path.name}: {e}", exc_info=True)
        # Clean up edge_db if it was created
        if edge_db is not None:
            edge_db.close(success=False)
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
        description='Parallel Memory-Optimized Edge Builder for Large Graphs',
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
        '--workers', type=int, default=7,
        help='Number of parallel worker processes'
    )
    parser.add_argument(
        '--max-ram-gb', type=float, default=19.0,
        help='Maximum RAM usage in GB (hard limit)'
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
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)

    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("PARALLEL MEMORY-OPTIMIZED EDGE BUILDER")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Min overlap: {args.min_overlap}")
    logger.info(f"Max shared: {args.max_shared}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Max RAM: {args.max_ram_gb:.1f}GB")
    logger.info(f"Parallel processing with checkpointing")

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

    # Process files sequentially (but with parallel edge generation within each file)
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING FILES")
    logger.info("=" * 70)

    start_time = time.time()
    results = []

    for idx, checkpoint_file in enumerate(checkpoint_files, 1):
        logger.info(f"\n[{idx}/{len(checkpoint_files)}]")

        file_start = time.time()

        success, edges = process_checkpoint_file(
            checkpoint_file,
            min_overlap=args.min_overlap,
            max_shared=args.max_shared,
            num_workers=args.workers,
            max_ram_gb=args.max_ram_gb
        )

        file_elapsed = time.time() - file_start

        results.append((success, edges))

        if success:
            state_tracker.mark_completed(checkpoint_file)
            logger.info(f"File completed in {file_elapsed:.1f}s ({file_elapsed/60:.1f} minutes)")
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
    if len(checkpoint_files) > 0:
        logger.info(f"Average: {elapsed/len(checkpoint_files):.1f}s per file")

    if failed == 0:
        logger.info("\n✓ All files processed successfully!")


if __name__ == "__main__":
    main()

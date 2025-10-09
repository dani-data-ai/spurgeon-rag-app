
"""
Optimized edge builder for checkpoint graph files - Version 2.2
Production-ready for 5M+ node graphs with billions of edges.

New in v2.2:
- Parallel file processing with ProcessPoolExecutor
- Optimized edge attributes (removed redundant shared_count)
- Configurable max_shared default (3 for large graphs)
- Better batch sizing for memory efficiency
- Worker isolation with proper pickling

Maintains all v2.1 quality: bug fixes, validation, state tracking.
"""

import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Iterator
import sys
import time
import logging
import psutil
import argparse
import json
from collections import defaultdict
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('add_edges.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage with automatic garbage collection."""
    
    def __init__(self, threshold_percent=80, gc_threshold=75):
        self.threshold_percent = threshold_percent
        self.gc_threshold = gc_threshold
    
    def check_memory(self, force_gc=False) -> bool:
        """Check memory and trigger GC if needed."""
        memory = psutil.virtual_memory()
        
        if memory.percent > self.gc_threshold or force_gc:
            gc.collect()
            memory = psutil.virtual_memory()
        
        if memory.percent > self.threshold_percent:
            logger.warning(f"Memory usage high: {memory.percent:.1f}%")
            return False
        return True
    
    def get_memory_usage(self) -> float:
        return psutil.virtual_memory().percent


class StateTracker:
    """Track processing state for resume capability."""
    
    def __init__(self, state_file: Path = Path('.edge_builder_state.json')):
        self.state_file = state_file
        self.state = self._load_state()
        self._lock_file = state_file.with_suffix('.lock')
    
    def _load_state(self) -> dict:
        """Load existing state or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {'completed': [], 'failed': []}
    
    def save_state(self):
        """Save current state to disk with file locking."""
        try:
            # Simple file lock for multiprocessing
            lock_attempts = 0
            while self._lock_file.exists() and lock_attempts < 50:
                time.sleep(0.1)
                lock_attempts += 1
            
            self._lock_file.touch()
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
            finally:
                if self._lock_file.exists():
                    self._lock_file.unlink()
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def is_completed(self, file_path: Path) -> bool:
        """Check if file already processed."""
        return str(file_path) in self.state['completed']
    
    def mark_completed(self, file_path: Path):
        """Mark file as completed."""
        if str(file_path) not in self.state['completed']:
            self.state['completed'].append(str(file_path))
            self.save_state()
    
    def mark_failed(self, file_path: Path):
        """Mark file as failed."""
        file_str = str(file_path)
        if file_str not in self.state['failed']:
            self.state['failed'].append(file_str)
            self.save_state()
    
    def clear(self):
        """Clear state file."""
        self.state = {'completed': [], 'failed': []}
        if self.state_file.exists():
            self.state_file.unlink()
        if self._lock_file.exists():
            self._lock_file.unlink()


def find_checkpoint_files(directory: Path) -> List[Path]:
    """Find all checkpoint .gml files."""
    checkpoint_files = sorted([
        f for f in directory.glob('checkpoint_*.gml')
        if f.is_file()
    ])
    return checkpoint_files


def parse_field_optimized(field_data) -> Set[str]:
    """Optimized field parsing with minimal string operations."""
    if not field_data:
        return set()
    
    if isinstance(field_data, str):
        return {item for item in (s.strip() for s in field_data.split(',')) if item}
    elif isinstance(field_data, (list, tuple)):
        return {str(item).strip() for item in field_data if item}
    return set()


def extract_entities_from_node(node_data: Dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract entities, keywords, and figures efficiently."""
    return (
        parse_field_optimized(node_data.get('entities')),
        parse_field_optimized(node_data.get('keywords')),
        parse_field_optimized(node_data.get('figures'))
    )


def build_inverted_index(graph: nx.Graph, memory_monitor: MemoryMonitor) -> Tuple[Dict, Dict, Dict]:
    """Build inverted indexes with progress tracking and memory management."""
    logger.info("Building inverted indexes...")
    
    entity_index = defaultdict(set)
    keyword_index = defaultdict(set)
    figure_index = defaultdict(set)
    
    total_nodes = graph.number_of_nodes()
    
    with tqdm(total=total_nodes, desc="Indexing nodes", unit="node") as pbar:
        for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
            entities, keywords, figures = extract_entities_from_node(node_data)
            
            for entity in entities:
                entity_index[entity].add(node_id)
            for keyword in keywords:
                keyword_index[keyword].add(node_id)
            for figure in figures:
                figure_index[figure].add(node_id)
            
            if i % 5000 == 0 and i > 0:
                memory_monitor.check_memory()
            
            pbar.update(1)
    
    logger.info(f"Indexed: {len(entity_index)} entities, {len(keyword_index)} keywords, {len(figure_index)} figures")
    return entity_index, keyword_index, figure_index


def generate_edges_from_index(index: Dict[str, Set], edge_type: str) -> Iterator[Tuple]:
    """
    Generator that yields edges one at a time for memory efficiency.
    """
    total_groups = len(index)
    
    with tqdm(total=total_groups, desc=f"Processing {edge_type}s", unit="group") as pbar:
        for item, nodes in index.items():
            if len(nodes) < 2:
                pbar.update(1)
                continue
            
            nodes_list = list(nodes)
            
            for i in range(len(nodes_list)):
                for j in range(i + 1, len(nodes_list)):
                    node1, node2 = nodes_list[i], nodes_list[j]
                    edge_key = tuple(sorted((node1, node2)))
                    
                    yield (edge_key, edge_type, item)
            
            pbar.update(1)


def add_edges_in_batches(graph: nx.Graph, edge_generator: Iterator, 
                         batch_size: int, min_overlap: int, 
                         max_shared: int, memory_monitor: MemoryMonitor) -> int:
    """
    Add edges in batches to minimize memory usage.
    Optimized: Removed redundant shared_count (can be derived from shared_items).
    """
    logger.info("Adding edges in batches...")
    
    edge_data = defaultdict(lambda: {'weight': 0, 'types': set(), 'shared': []})
    edges_added = 0
    batch_count = 0
    
    for edge_key, edge_type, item in edge_generator:
        # Aggregate edge data
        edge_data[edge_key]['weight'] += 1
        edge_data[edge_key]['types'].add(edge_type)
        
        # Cap shared items for memory efficiency
        if len(edge_data[edge_key]['shared']) < max_shared:
            edge_data[edge_key]['shared'].append(item)
        
        batch_count += 1
        
        # Process batch when size reached
        if batch_count >= batch_size:
            edges_added += flush_edge_batch(graph, edge_data, min_overlap)
            edge_data.clear()
            batch_count = 0
            memory_monitor.check_memory(force_gc=True)
    
    # Flush remaining edges
    if edge_data:
        edges_added += flush_edge_batch(graph, edge_data, min_overlap)
        edge_data.clear()
    
    return edges_added


def flush_edge_batch(graph: nx.Graph, edge_data: Dict, min_overlap: int) -> int:
    """
    Flush a batch of edges to the graph, filtering by min_overlap.
    Optimized: Removed shared_count attribute (redundant).
    """
    added = 0
    
    for (node1, node2), data in edge_data.items():
        if data['weight'] >= min_overlap:
            # Store only: weight, type, shared_items
            # shared_count can be derived: len(shared_items.split(','))
            graph.add_edge(
                node1, node2,
                weight=data['weight'],
                type=','.join(sorted(data['types'])),
                shared_items=','.join(data['shared'])
            )
            added += 1
    
    return added


def create_edges_optimized(graph: nx.Graph, min_overlap: int = 1, 
                          max_shared_items: int = 3, 
                          batch_size: int = 100000,
                          memory_monitor: Optional[MemoryMonitor] = None) -> int:
    """
    Create edges using optimized batching and filtering.
    Keeps indexes in memory until ALL edge generation is complete.
    """
    if memory_monitor is None:
        memory_monitor = MemoryMonitor()
    
    # Build all indexes FIRST - keep them in memory
    entity_index, keyword_index, figure_index = build_inverted_index(graph, memory_monitor)
    
    edges_added = 0
    
    # Process entity edges
    logger.info("Generating entity edges...")
    entity_gen = generate_edges_from_index(entity_index, 'entity')
    edges_added += add_edges_in_batches(
        graph, entity_gen, batch_size, min_overlap, max_shared_items, memory_monitor
    )
    
    # Process keyword edges
    logger.info("Generating keyword edges...")
    keyword_gen = generate_edges_from_index(keyword_index, 'keyword')
    edges_added += add_edges_in_batches(
        graph, keyword_gen, batch_size, min_overlap, max_shared_items, memory_monitor
    )
    
    # Process figure edges
    logger.info("Generating figure edges...")
    figure_gen = generate_edges_from_index(figure_index, 'figure')
    edges_added += add_edges_in_batches(
        graph, figure_gen, batch_size, min_overlap, max_shared_items, memory_monitor
    )
    
    # Safe to clear indexes after ALL generation is complete
    entity_index.clear()
    keyword_index.clear()
    figure_index.clear()
    gc.collect()
    
    logger.info(f"Added {edges_added:,} edges total")
    return edges_added


def validate_graph_basic(graph: nx.Graph) -> Tuple[bool, str]:
    """Basic graph validation."""
    if graph.number_of_nodes() == 0:
        return False, "Empty graph"
    
    # Check for required attributes on some nodes
    has_attrs = any(
        any(attr in data for attr in ['entities', 'keywords', 'figures'])
        for _, data in list(graph.nodes(data=True))[:100]
    )
    
    if not has_attrs:
        return False, "No nodes have required attributes"
    
    return True, "OK"


def process_checkpoint_file(file_path: Path, min_overlap: int = 1, 
                           max_shared_items: int = 3,
                           batch_size: int = 100000) -> Tuple[bool, int, int]:
    """
    Process a single checkpoint file.
    Worker function for parallel processing - must be picklable.
    """
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'='*70}")
        
        # Load graph
        logger.info("Loading graph...")
        graph = nx.read_gml(str(file_path))
        
        nodes_count = graph.number_of_nodes()
        edges_before = graph.number_of_edges()
        
        # Basic validation
        is_valid, msg = validate_graph_basic(graph)
        if not is_valid:
            logger.warning(f"Validation: {msg}")
            if nodes_count == 0:
                return True, 0, 0
        
        logger.info(f"Loaded: {nodes_count:,} nodes, {edges_before:,} existing edges")
        
        # Create edges
        memory_monitor = MemoryMonitor()
        edges_added = create_edges_optimized(
            graph, min_overlap, max_shared_items, batch_size, memory_monitor
        )
        
        edges_after = graph.number_of_edges()
        
        # Save graph
        logger.info("Saving updated graph...")
        nx.write_gml(graph, str(file_path))
        
        logger.info(f"✓ Completed: {edges_added:,} edges added (total: {edges_after:,})")
        
        # Cleanup
        del graph
        gc.collect()
        
        return True, nodes_count, edges_added
        
    except Exception as e:
        logger.error(f"✗ Error processing {file_path.name}: {e}", exc_info=True)
        return False, 0, 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimized edge builder for checkpoint graph files (v2.2)',
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
        help='Max shared items per edge (reduced for large graphs)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100000,
        help='Edge batch size (increased for better efficiency)'
    )
    parser.add_argument(
        '--workers', type=int, default=7,
        help='Parallel workers (0=auto, 1=sequential)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint (skip completed files)'
    )
    parser.add_argument(
        '--clear-state', action='store_true',
        help='Clear state and start fresh'
    )
    
    return parser.parse_args()


def main():
    """Main execution with parallel processing support."""
    args = parse_arguments()
    
    logger.info("=" * 70)
    logger.info("OPTIMIZED EDGE BUILDER v2.2")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Min overlap: {args.min_overlap}")
    logger.info(f"Max shared items: {args.max_shared}")
    logger.info(f"Batch size: {args.batch_size:,}")
    
    work_dir = Path(args.source_dir)
    
    # State tracking for resume
    state_tracker = StateTracker()
    
    if args.clear_state:
        state_tracker.clear()
        logger.info("State cleared")
    
    # Find files
    logger.info("\nSearching for checkpoint files...")
    checkpoint_files = find_checkpoint_files(work_dir)
    
    if not checkpoint_files:
        logger.error("No checkpoint files found (checkpoint_*.gml)")
        sys.exit(1)
    
    # Filter already completed if resuming
    if args.resume:
        original_count = len(checkpoint_files)
        checkpoint_files = [f for f in checkpoint_files if not state_tracker.is_completed(f)]
        skipped = original_count - len(checkpoint_files)
        if skipped > 0:
            logger.info(f"Resuming: skipped {skipped} completed files")
    
    logger.info(f"Found {len(checkpoint_files)} file(s) to process")
    for f in checkpoint_files:
        logger.info(f"  - {f.name}")
    
    # Determine number of workers
    if args.workers == 0:
        workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    else:
        workers = args.workers
    
    logger.info(f"Using {workers} worker(s)")
    
    # Process files
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING FILES")
    logger.info("=" * 70)
    
    start_time = time.time()
    results = []
    
    if workers == 1:
        # Sequential processing
        for checkpoint_file in checkpoint_files:
            result = process_checkpoint_file(
                checkpoint_file,
                min_overlap=args.min_overlap,
                max_shared_items=args.max_shared,
                batch_size=args.batch_size
            )
            
            success, nodes, edges = result
            results.append(result)
            
            # Track state
            if success:
                state_tracker.mark_completed(checkpoint_file)
            else:
                state_tracker.mark_failed(checkpoint_file)
    else:
        # Parallel processing with ProcessPoolExecutor
        logger.info("Starting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(
                    process_checkpoint_file,
                    f,
                    args.min_overlap,
                    args.max_shared,
                    args.batch_size
                ): f for f in checkpoint_files
            }
            
            # Process results as they complete
            with tqdm(total=len(checkpoint_files), desc="Overall progress", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    checkpoint_file = future_to_file[future]
                    try:
                        result = future.result()
                        success, nodes, edges = result
                        results.append(result)
                        
                        # Track state
                        if success:
                            state_tracker.mark_completed(checkpoint_file)
                        else:
                            state_tracker.mark_failed(checkpoint_file)
                        
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Worker exception for {checkpoint_file.name}: {e}")
                        results.append((False, 0, 0))
                        state_tracker.mark_failed(checkpoint_file)
                        pbar.update(1)
    
    end_time = time.time()
    
    # Summary
    successful = sum(1 for s, _, _ in results if s)
    failed = len(results) - successful
    total_nodes = sum(n for _, n, _ in results)
    total_edges = sum(e for _, _, e in results)
    elapsed = end_time - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Processed: {successful}/{len(checkpoint_files)} files")
    if failed > 0:
        logger.warning(f"Failed: {failed} files")
    logger.info(f"Total nodes: {total_nodes:,}")
    logger.info(f"Total edges added: {total_edges:,}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Average: {elapsed / len(checkpoint_files):.1f}s/file")
    
    if workers > 1:
        logger.info(f"Speedup: ~{len(checkpoint_files) * (elapsed / len(checkpoint_files)) / elapsed:.1f}x with {workers} workers")
    
    if failed == 0:
        logger.info("\n✓ All files processed successfully!")
        logger.info("Next: Run merge script to create master graph")
    else:
        logger.warning("\nSome files failed. Use --resume to retry")


if __name__ == "__main__":
    main()
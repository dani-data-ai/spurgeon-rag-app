"""
Orchestration module for RAG pipeline.
Manages workflow, parallel processing, and checkpointing.
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import spacy

import config
from text_processor import TextProcessor
from graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

# Global spaCy model shared across workers
_nlp_model = None


def _init_worker():
    """Initialize worker process with shared spaCy model."""
    global _nlp_model
    if _nlp_model is None:
        logger.info(f"Loading spaCy model in worker process: {config.SPACY_MODEL}")
        _nlp_model = spacy.load(config.SPACY_MODEL)


def _get_nlp_model():
    """Get the shared spaCy model instance."""
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load(config.SPACY_MODEL)
    return _nlp_model


def process_single_file(file_path: Path) -> Dict:
    """
    Process a single file and return its graph data.

    This function runs in a worker process and must be picklable.

    Args:
        file_path: Path to file to process

    Returns:
        Dictionary with graph data and metadata
    """
    result = {
        "file_path": str(file_path),
        "success": False,
        "error": None,
        "nodes": [],
        "edges": [],
        "chunks_processed": 0
    }

    try:
        # Get shared spaCy model
        nlp = _get_nlp_model()

        # Initialize processors
        text_processor = TextProcessor()
        graph_builder = GraphBuilder(nlp)

        file_name = file_path.name
        chunk_counter = 0

        # Process file chunks using generator (memory efficient)
        for parent_chunk, child_chunks in text_processor.process_file_chunks(file_path):
            chunk_counter += 1

            # Create parent chunk ID
            parent_id = f"{file_name}_parent_{chunk_counter}"

            # Extract entities from parent chunk
            parent_entities = graph_builder.extract_entities_from_text(parent_chunk)

            # Add parent chunk node
            graph_builder.add_chunk_node(
                chunk_id=parent_id,
                chunk_text=parent_chunk,
                chunk_type="parent",
                source_file=file_name
            )

            # Add entity relationships for parent
            graph_builder.add_entity_relationships(parent_id, parent_entities)

            # Add entity co-occurrence
            graph_builder.add_entity_cooccurrence(parent_entities)

            # Process child chunks
            for child_idx, child_chunk in enumerate(child_chunks, 1):
                child_id = f"{parent_id}_child_{child_idx}"

                # Extract entities from child chunk
                child_entities = graph_builder.extract_entities_from_text(child_chunk)

                # Add child chunk node
                graph_builder.add_chunk_node(
                    chunk_id=child_id,
                    chunk_text=child_chunk,
                    chunk_type="child",
                    source_file=file_name,
                    parent_id=parent_id
                )

                # Add entity relationships for child
                graph_builder.add_entity_relationships(child_id, child_entities)

                # Add entity co-occurrence
                graph_builder.add_entity_cooccurrence(child_entities)

        # Extract graph data for serialization
        result["nodes"] = [
            (node, data) for node, data in graph_builder.graph.nodes(data=True)
        ]
        result["edges"] = [
            (u, v, data) for u, v, data in graph_builder.graph.edges(data=True)
        ]
        result["chunks_processed"] = chunk_counter
        result["success"] = True

        logger.info(f"Successfully processed: {file_name} ({chunk_counter} chunks)")

    except Exception as e:
        error_msg = f"Error processing {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        result["error"] = error_msg

    return result


class ProcessingOrchestrator:
    """Orchestrates the entire RAG pipeline processing."""

    def __init__(self, collection_name: str):
        """
        Initialize the orchestrator.

        Args:
            collection_name: Name of the collection being processed
        """
        self.collection_name = collection_name
        self.checkpoint_path = Path(f"{config.CHECKPOINT_PREFIX}{collection_name}.gml")
        self.final_graph_path = Path(f"{config.FINAL_GRAPH_PREFIX}{collection_name}.gml")
        self.processed_files_path = Path(f"processed_files_{collection_name}.txt")

        # Initialize main process spaCy model
        logger.info(f"Loading spaCy model in main process: {config.SPACY_MODEL}")
        self.nlp = spacy.load(config.SPACY_MODEL)

        # Initialize graph builder
        self.graph_builder = GraphBuilder(self.nlp)

        # Track processed files
        self.processed_files: Set[str] = set()

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint and processed files list if they exist."""
        # Load graph checkpoint
        if self.checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {self.checkpoint_path}")
            if self.graph_builder.load_checkpoint(self.checkpoint_path):
                logger.info(f"Checkpoint loaded successfully")

        # Load processed files list
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r') as f:
                self.processed_files = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.processed_files)} processed files")

    def _save_checkpoint(self):
        """Save checkpoint and processed files list."""
        try:
            # Save graph checkpoint
            self.graph_builder.save_checkpoint(self.checkpoint_path)

            # Save processed files list
            with open(self.processed_files_path, 'w') as f:
                for file_path in sorted(self.processed_files):
                    f.write(f"{file_path}\n")

            logger.info(f"Checkpoint saved: {len(self.processed_files)} files processed")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def get_files_to_process(self, directory: Path) -> List[Path]:
        """
        Get list of files to process, excluding already processed ones.

        Args:
            directory: Directory to scan for files

        Returns:
            List of file paths to process
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        all_files = []
        for ext in config.SUPPORTED_EXTENSIONS:
            all_files.extend(directory.rglob(f"*{ext}"))

        # Filter out already processed files
        files_to_process = [
            f for f in all_files
            if str(f) not in self.processed_files
        ]

        logger.info(f"Found {len(all_files)} total files, {len(files_to_process)} to process")
        return sorted(files_to_process)

    def process_files_parallel(self, files: List[Path]):
        """
        Process files in parallel using multiprocessing.

        Args:
            files: List of file paths to process
        """
        if not files:
            logger.info("No files to process")
            return

        total_files = len(files)
        processed_count = 0
        failed_count = 0

        logger.info(f"Starting parallel processing of {total_files} files with {config.MAX_WORKERS} workers")

        # Use ProcessPoolExecutor with initializer for shared spaCy model
        with ProcessPoolExecutor(
            max_workers=config.MAX_WORKERS,
            initializer=_init_worker
        ) as executor:

            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, file_path): file_path
                for file_path in files
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    result = future.result()

                    if result["success"]:
                        # Merge result into main graph
                        self._merge_result(result)

                        # Mark as processed
                        self.processed_files.add(str(file_path))
                        processed_count += 1

                        logger.info(
                            f"Progress: {processed_count}/{total_files} - "
                            f"{result['chunks_processed']} chunks from {file_path.name}"
                        )

                    else:
                        failed_count += 1
                        logger.error(f"Failed to process {file_path}: {result['error']}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception processing {file_path}: {e}", exc_info=True)

                # Save checkpoint periodically
                if processed_count % config.CHECKPOINT_INTERVAL == 0:
                    logger.info(f"Saving checkpoint at {processed_count} files...")
                    self._save_checkpoint()

                    stats = self.graph_builder.get_stats()
                    logger.info(
                        f"Current graph stats: {stats['nodes']} nodes, "
                        f"{stats['edges']} edges, {stats['entities']} entities"
                    )

        # Final checkpoint save
        logger.info("Saving final checkpoint...")
        self._save_checkpoint()

        # Summary
        logger.info(f"Processing complete: {processed_count} succeeded, {failed_count} failed")
        stats = self.graph_builder.get_stats()
        logger.info(
            f"Final graph stats: {stats['nodes']} nodes, "
            f"{stats['edges']} edges, {stats['entities']} entities"
        )

    def _merge_result(self, result: Dict):
        """
        Merge processing result into main graph.

        Args:
            result: Result dictionary from worker process
        """
        # Add nodes
        for node, data in result["nodes"]:
            if node in self.graph_builder.graph:
                # Update existing node attributes
                self.graph_builder.graph.nodes[node].update(data)
            else:
                self.graph_builder.graph.add_node(node, **data)

        # Add edges
        for u, v, data in result["edges"]:
            if self.graph_builder.graph.has_edge(u, v):
                # Update existing edge attributes
                self.graph_builder.graph.edges[u, v].update(data)
            else:
                self.graph_builder.graph.add_edge(u, v, **data)

    def finalize(self):
        """Finalize processing and save final graph."""
        logger.info("Finalizing graph...")

        # Save final graph
        self.graph_builder.save_final_graph(self.final_graph_path)

        # Print final statistics
        stats = self.graph_builder.get_stats()
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Files processed: {len(self.processed_files)}")
        logger.info(f"Total nodes: {stats['nodes']}")
        logger.info(f"Total edges: {stats['edges']}")
        logger.info(f"Entity nodes: {stats['entities']}")
        logger.info(f"Output file: {self.final_graph_path}")
        logger.info("=" * 60)

    def run(self, source_directory: Path):
        """
        Run the complete pipeline.

        Args:
            source_directory: Directory containing files to process
        """
        start_time = datetime.now()
        logger.info(f"Starting pipeline for collection: {self.collection_name}")
        logger.info(f"Source directory: {source_directory}")

        # Get files to process
        files = self.get_files_to_process(source_directory)

        if not files:
            logger.info("No new files to process")
        else:
            # Process files
            self.process_files_parallel(files)

        # Finalize
        self.finalize()

        elapsed = datetime.now() - start_time
        logger.info(f"Total elapsed time: {elapsed}")

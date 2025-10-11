"""
GML Graph Merger for RAG System
Merges 5 large GML files into a single master graph with memory-efficient streaming.
"""

import networkx as nx
import os
import time
import gc
from pathlib import Path
from typing import Set, Dict, Tuple

# Input files in processing order
INPUT_FILES = [
    "checkpoint_Epub_Vol_1-25.gml",
    "checkpoint_Epub_Vol_26-79.gml",
    "checkpoint_Epub_Vol_80-125.gml",
    "checkpoint_spurgeon.gml",
    "checkpoint_reference.gml"
]

# File size estimates for time calculation (in GB)
FILE_SIZES = {
    "checkpoint_Epub_Vol_1-25.gml": 4.1,
    "checkpoint_Epub_Vol_26-79.gml": 5.9,
    "checkpoint_Epub_Vol_80-125.gml": 0.743,
    "checkpoint_spurgeon.gml": 0.315,
    "checkpoint_reference.gml": 1.0  # Assumed
}

# Output files
OUTPUT_GRAPH = "master_graph.gml"
SUMMARY_FILE = "merge_summary.txt"
LOG_FILE = "merge_log.txt"


def log_message(message: str, log_file_path: str):
    """Write message to both console and log file."""
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def parse_gml_line_by_line(file_path: str, log_file_path: str) -> Tuple[list, list]:
    """
    Fallback parser: Read GML file line by line and extract nodes/edges.
    Returns: (nodes_list, edges_list) where each is a list of dicts
    """
    log_message(f"Using line-by-line fallback parser for {file_path}", log_file_path)

    nodes = []
    edges = []
    current_section = None
    current_item = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect section start
            if line == 'node [':
                current_section = 'node'
                current_item = {}
            elif line == 'edge [':
                current_section = 'edge'
                current_item = {}
            elif line == ']' and current_section:
                # Section end
                if current_section == 'node' and 'id' in current_item:
                    nodes.append(current_item)
                elif current_section == 'edge' and 'source' in current_item and 'target' in current_item:
                    edges.append(current_item)
                current_section = None
                current_item = {}
            elif current_section and line:
                # Parse attribute: key value
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    # Remove quotes from string values
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    # Try to convert to int/float
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except (ValueError, AttributeError):
                        pass
                    current_item[key] = value

    log_message(f"Fallback parser extracted {len(nodes)} nodes and {len(edges)} edges", log_file_path)
    return nodes, edges


def load_and_process_file(file_path: str, master_graph: nx.Graph, existing_node_ids: Set[str],
                          log_file_path: str, edge_weights: Dict[Tuple[str, str], Tuple[float, dict]]) -> Tuple[int, int]:
    """
    Load one GML file, check for ID conflicts, apply prefixes if needed, add to master graph.
    Returns: (nodes_added, edges_added_before_dedup)
    """
    if not os.path.exists(file_path):
        log_message(f"ERROR: File not found: {file_path}", log_file_path)
        return 0, 0

    file_name = Path(file_path).stem
    log_message(f"Loading file: {file_path}", log_file_path)

    loaded_graph = None
    nodes_data = []
    edges_data = []

    # Attempt to load with NetworkX
    try:
        loaded_graph = nx.read_gml(file_path)
        log_message(f"Successfully loaded with nx.read_gml: {loaded_graph.number_of_nodes()} nodes, {loaded_graph.number_of_edges()} edges", log_file_path)
    except Exception as e:
        log_message(f"nx.read_gml() failed for {file_path}: {e}", log_file_path)

        # Fallback to line-by-line parsing
        try:
            nodes_data, edges_data = parse_gml_line_by_line(file_path, log_file_path)
        except Exception as fallback_error:
            log_message(f"ERROR: Fallback parser also failed for {file_path}: {fallback_error}", log_file_path)
            log_message(f"Skipping file {file_path} due to parsing errors", log_file_path)
            return 0, 0

    # Check for ID conflicts
    if loaded_graph:
        file_node_ids = set(str(node) for node in loaded_graph.nodes())
    else:
        file_node_ids = set(str(node['id']) for node in nodes_data)

    conflicts = file_node_ids & existing_node_ids
    needs_prefix = len(conflicts) > 0

    if needs_prefix:
        log_message(f"Conflict detected: {len(conflicts)} node IDs overlap in {file_path}", log_file_path)
        log_message(f"Applying prefix '{file_name}' to all nodes from this file", log_file_path)
        prefix = f"{file_name}_"
    else:
        log_message(f"No ID conflicts detected for {file_path}", log_file_path)
        prefix = ""

    nodes_added = 0
    edges_added = 0

    # Add nodes to master graph
    if loaded_graph:
        for node_id, node_data in loaded_graph.nodes(data=True):
            new_id = f"{prefix}{node_id}"
            master_graph.add_node(new_id, **node_data)
            existing_node_ids.add(new_id)
            nodes_added += 1

        # Add edges to master graph with deduplication tracking
        for source, target, edge_data in loaded_graph.edges(data=True):
            new_source = f"{prefix}{source}"
            new_target = f"{prefix}{target}"
            # Ensure weight attribute exists
            if 'weight' not in edge_data:
                edge_data['weight'] = 1

            # Track edge with highest weight
            edge_key = tuple(sorted([new_source, new_target]))
            current_weight = edge_data['weight']

            if edge_key in edge_weights:
                existing_weight, existing_data = edge_weights[edge_key]
                if current_weight > existing_weight:
                    # Replace with higher weight edge
                    edge_weights[edge_key] = (current_weight, edge_data)
                # else: keep existing higher weight edge
            else:
                edge_weights[edge_key] = (current_weight, edge_data)

            edges_added += 1

        # Free memory
        del loaded_graph
    else:
        # Using fallback parsed data
        for node in nodes_data:
            node_id = str(node.pop('id'))
            new_id = f"{prefix}{node_id}"
            master_graph.add_node(new_id, **node)
            existing_node_ids.add(new_id)
            nodes_added += 1

        for edge in edges_data:
            source = str(edge.pop('source'))
            target = str(edge.pop('target'))
            new_source = f"{prefix}{source}"
            new_target = f"{prefix}{target}"
            if 'weight' not in edge:
                edge['weight'] = 1

            # Track edge with highest weight
            edge_key = tuple(sorted([new_source, new_target]))
            current_weight = edge['weight']

            if edge_key in edge_weights:
                existing_weight, existing_data = edge_weights[edge_key]
                if current_weight > existing_weight:
                    edge_weights[edge_key] = (current_weight, edge)
            else:
                edge_weights[edge_key] = (current_weight, edge)

            edges_added += 1

    gc.collect()
    log_message(f"Added {nodes_added} nodes and {edges_added} edges from {file_path}", log_file_path)
    return nodes_added, edges_added


def deduplicate_edges(graph: nx.Graph, log_file_path: str) -> int:
    """
    Remove duplicate edges, keeping the one with highest weight.
    NetworkX Graph auto-merges duplicate edges when added, so this validates.
    Returns: number of edges that would have been removed (always 0 for nx.Graph)
    """
    log_message("Deduplicating edges...", log_file_path)

    # NetworkX Graph doesn't allow parallel edges - duplicates are automatically merged
    # When you add an edge that already exists, it updates the attributes
    # So deduplication happens automatically during the add_edge() calls

    # This function just validates there are no duplicate edge keys
    edge_pairs = set()
    duplicate_count = 0

    for u, v in graph.edges():
        edge_key = tuple(sorted([u, v]))
        if edge_key in edge_pairs:
            duplicate_count += 1
            log_message(f"WARNING: Found duplicate edge: {u} <-> {v}", log_file_path)
        edge_pairs.add(edge_key)

    if duplicate_count > 0:
        log_message(f"WARNING: Found {duplicate_count} duplicate edges during validation", log_file_path)
    else:
        log_message("No duplicate edges found (as expected for nx.Graph)", log_file_path)

    return 0  # NetworkX Graph auto-handles duplicates


def validate_graph(graph: nx.Graph, total_nodes_expected: int, total_edges_expected: int,
                   duplicates_removed: int, log_file_path: str):
    """Validate merged graph integrity."""
    log_message("Validating merged graph...", log_file_path)

    actual_nodes = graph.number_of_nodes()
    actual_edges = graph.number_of_edges()

    log_message(f"Validation - Actual nodes: {actual_nodes}, Expected: {total_nodes_expected}", log_file_path)
    log_message(f"Validation - Actual edges: {actual_edges}, Expected (after dedup): {total_edges_expected - duplicates_removed}", log_file_path)

    # Check for duplicate edges using iteration
    edge_pairs = set()
    duplicate_count = 0
    for u, v in graph.edges():
        edge_key = tuple(sorted([u, v]))
        if edge_key in edge_pairs:
            duplicate_count += 1
        edge_pairs.add(edge_key)

    if duplicate_count > 0:
        log_message(f"WARNING: Found {duplicate_count} duplicate edges during validation", log_file_path)
    else:
        log_message("Validation passed: No duplicate edges found", log_file_path)


def main():
    """Main merge process."""
    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Graph Merge Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

    log_message("Starting graph merge process", LOG_FILE)
    log_message(f"Input files: {len(INPUT_FILES)}", LOG_FILE)

    # Initialize master graph
    master_graph = nx.Graph()
    existing_node_ids = set()
    edge_weights = {}  # Track (node1, node2) -> (weight, edge_data) for deduplication

    # Track statistics
    total_nodes_added = 0
    total_edges_added = 0
    processed_files = []

    # Calculate total size for time estimation
    total_size_gb = sum(FILE_SIZES.get(f, 0) for f in INPUT_FILES)
    start_time = time.time()
    processed_size_gb = 0

    # Process each file sequentially
    for idx, file_path in enumerate(INPUT_FILES, 1):
        file_start_time = time.time()

        log_message(f"\n{'='*80}", LOG_FILE)
        log_message(f"Processing file {idx}/{len(INPUT_FILES)}: {file_path}", LOG_FILE)
        log_message(f"{'='*80}", LOG_FILE)

        # Load and merge file
        try:
            nodes_added, edges_added = load_and_process_file(
                file_path, master_graph, existing_node_ids, LOG_FILE, edge_weights
            )

            if nodes_added > 0:
                total_nodes_added += nodes_added
                total_edges_added += edges_added
                processed_files.append(file_path)

                # Progress update (edges not added to graph yet)
                log_message(f"After processing {file_path}: {master_graph.number_of_nodes()} nodes, {len(edge_weights)} unique edges tracked", LOG_FILE)

                # Time estimation
                file_elapsed = time.time() - file_start_time
                file_size = FILE_SIZES.get(file_path, 0)
                processed_size_gb += file_size

                if processed_size_gb > 0:
                    time_per_gb = (time.time() - start_time) / processed_size_gb
                    remaining_size = total_size_gb - processed_size_gb
                    estimated_remaining_sec = time_per_gb * remaining_size

                    log_message(f"File processed in {file_elapsed:.1f}s", LOG_FILE)
                    log_message(f"Estimated time remaining: {estimated_remaining_sec/60:.1f} minutes", LOG_FILE)

        except Exception as e:
            log_message(f"ERROR: Failed to process {file_path}: {e}", LOG_FILE)
            log_message(f"Skipping {file_path} and continuing with remaining files", LOG_FILE)
            continue

    # Add deduplicated edges to graph
    log_message(f"\n{'='*80}", LOG_FILE)
    log_message("Adding deduplicated edges to graph...", LOG_FILE)
    log_message(f"Total edge candidates: {total_edges_added}, Unique edges: {len(edge_weights)}", LOG_FILE)

    edges_actually_added = 0
    for edge_key, (weight, edge_data) in edge_weights.items():
        node1, node2 = edge_key
        master_graph.add_edge(node1, node2, **edge_data)
        edges_actually_added += 1

    total_duplicates_removed = total_edges_added - edges_actually_added
    log_message(f"Added {edges_actually_added} unique edges to graph", LOG_FILE)
    log_message(f"Removed {total_duplicates_removed} duplicate edges", LOG_FILE)

    # Validate merged graph
    log_message(f"\n{'='*80}", LOG_FILE)
    log_message("Running validation...", LOG_FILE)
    validate_graph(master_graph, total_nodes_added, total_edges_added,
                   total_duplicates_removed, LOG_FILE)

    # Save merged graph
    log_message(f"\n{'='*80}", LOG_FILE)
    log_message(f"Saving merged graph to {OUTPUT_GRAPH}...", LOG_FILE)
    try:
        nx.write_gml(master_graph, OUTPUT_GRAPH)
        log_message(f"Successfully saved merged graph: {OUTPUT_GRAPH}", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to save merged graph: {e}", LOG_FILE)

    # Write summary file
    log_message(f"Writing summary to {SUMMARY_FILE}...", LOG_FILE)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("Graph Merge Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total nodes: {master_graph.number_of_nodes()}\n")
        f.write(f"Total edges: {master_graph.number_of_edges()}\n")
        f.write(f"Duplicate edges removed: {total_duplicates_removed}\n\n")
        f.write("Source files:\n")
        for file_path in processed_files:
            f.write(f"  - {file_path}\n")
        f.write(f"\nMerge completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processing time: {(time.time() - start_time)/60:.1f} minutes\n")

    total_time = time.time() - start_time
    log_message(f"\n{'='*80}", LOG_FILE)
    log_message(f"MERGE COMPLETE", LOG_FILE)
    log_message(f"Total processing time: {total_time/60:.1f} minutes", LOG_FILE)
    log_message(f"Final graph: {master_graph.number_of_nodes()} nodes, {master_graph.number_of_edges()} edges", LOG_FILE)
    log_message(f"Output files: {OUTPUT_GRAPH}, {SUMMARY_FILE}, {LOG_FILE}", LOG_FILE)


if __name__ == "__main__":
    main()

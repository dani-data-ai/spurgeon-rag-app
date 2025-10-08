"""
Merge undirected NetworkX checkpoint graph files into a single master graph.

This script specifically handles undirected graphs (nx.Graph) and merges
all checkpoint files into a unified knowledge graph.
"""

import networkx as nx
from pathlib import Path
from typing import List
import sys


def find_checkpoint_files(directory: Path) -> List[Path]:
    """
    Find all checkpoint .gml files in the specified directory.

    Args:
        directory: Directory to search

    Returns:
        Sorted list of checkpoint file paths
    """
    checkpoint_files = [
        f for f in directory.glob('checkpoint_*.gml')
        if f.is_file()
    ]
    return sorted(checkpoint_files)


def load_graph_safe(file_path: Path) -> nx.Graph:
    """
    Safely load a graph file and ensure it's undirected.

    Args:
        file_path: Path to graph file

    Returns:
        NetworkX Graph object

    Raises:
        ValueError: If graph is directed
        Exception: If file cannot be loaded
    """
    try:
        graph = nx.read_gml(str(file_path))

        # Verify it's undirected
        if graph.is_directed():
            raise ValueError(f"Graph {file_path.name} is directed, expected undirected")

        return graph

    except Exception as e:
        print(f"✗ Error loading {file_path.name}: {e}")
        raise


def merge_checkpoint_graphs(checkpoint_files: List[Path]) -> nx.Graph:
    """
    Merge multiple checkpoint graphs into a single master graph.

    Args:
        checkpoint_files: List of checkpoint file paths to merge

    Returns:
        Merged undirected graph
    """
    # Create empty undirected graph
    master_graph = nx.Graph()

    print("\n" + "="*60)
    print("MERGING CHECKPOINT GRAPHS")
    print("="*60 + "\n")

    for idx, file_path in enumerate(checkpoint_files, 1):
        print(f"[{idx}/{len(checkpoint_files)}] Loading: {file_path.name}")

        try:
            # Load the checkpoint graph
            graph = load_graph_safe(file_path)

            # Print statistics before merging
            nodes_before = master_graph.number_of_nodes()
            edges_before = master_graph.number_of_edges()

            print(f"  Graph stats: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

            # Merge into master graph using compose
            master_graph = nx.compose(master_graph, graph)

            # Print merge statistics
            nodes_added = master_graph.number_of_nodes() - nodes_before
            edges_added = master_graph.number_of_edges() - edges_before

            print(f"  Added: {nodes_added:,} new nodes, {edges_added:,} new edges")
            print(f"  Master total: {master_graph.number_of_nodes():,} nodes, {master_graph.number_of_edges():,} edges")
            print()

        except Exception as e:
            print(f"  ⚠ Skipping due to error: {e}\n")
            continue

    return master_graph


def analyze_graph(graph: nx.Graph) -> dict:
    """
    Analyze graph and return detailed statistics.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        Dictionary with statistics
    """
    stats = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'is_directed': graph.is_directed(),
        'graph_type': type(graph).__name__
    }

    if stats['nodes'] > 0:
        # Calculate average degree
        degrees = dict(graph.degree())
        stats['avg_degree'] = sum(degrees.values()) / stats['nodes']
        stats['max_degree'] = max(degrees.values())
        stats['min_degree'] = min(degrees.values())

        # Find isolated nodes
        isolated = [n for n, d in degrees.items() if d == 0]
        stats['isolated_nodes'] = len(isolated)

        # Check for node types if attributes exist
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats['node_types'] = node_types

        # Check for edge types if attributes exist
        edge_types = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('relation', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats['edge_types'] = edge_types

    return stats


def print_statistics(stats: dict):
    """
    Print formatted graph statistics.

    Args:
        stats: Statistics dictionary from analyze_graph
    """
    print("\n" + "="*60)
    print("FINAL MASTER GRAPH STATISTICS")
    print("="*60 + "\n")

    print(f"Graph Type: {stats['graph_type']}")
    print(f"Directed: {stats['is_directed']}")
    print(f"\nTotal Nodes: {stats['nodes']:,}")
    print(f"Total Edges: {stats['edges']:,}")

    if stats['nodes'] > 0:
        print(f"\nDegree Statistics:")
        print(f"  Average Degree: {stats['avg_degree']:.2f}")
        print(f"  Maximum Degree: {stats['max_degree']:,}")
        print(f"  Minimum Degree: {stats['min_degree']:,}")
        print(f"  Isolated Nodes: {stats['isolated_nodes']:,}")

        if stats.get('node_types'):
            print(f"\nNode Types:")
            for node_type, count in sorted(stats['node_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {node_type}: {count:,}")

        if stats.get('edge_types'):
            print(f"\nEdge Relation Types:")
            for edge_type, count in sorted(stats['edge_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {edge_type}: {count:,}")


def main():
    """Main execution function."""
    print("="*60)
    print("UNDIRECTED GRAPH MERGER")
    print("="*60)
    print("\nThis script merges checkpoint graphs into a master graph.")
    print("All graphs must be undirected (nx.Graph).\n")

    # Set directory
    work_dir = Path.cwd()
    print(f"Working directory: {work_dir}\n")

    # Find checkpoint files
    print("Searching for checkpoint files...")
    checkpoint_files = find_checkpoint_files(work_dir)

    if not checkpoint_files:
        print("✗ No checkpoint files found!")
        print("  Looking for files matching pattern: checkpoint_*.gml")
        sys.exit(1)

    print(f"✓ Found {len(checkpoint_files)} checkpoint file(s):\n")
    for f in checkpoint_files:
        print(f"  • {f.name}")

    # Merge graphs
    try:
        master_graph = merge_checkpoint_graphs(checkpoint_files)
    except Exception as e:
        print(f"\n✗ Fatal error during merging: {e}")
        sys.exit(1)

    # Verify we have a valid graph
    if master_graph.number_of_nodes() == 0:
        print("\n⚠ Warning: Master graph is empty!")
        sys.exit(1)

    # Analyze the merged graph
    stats = analyze_graph(master_graph)

    # Save the master graph
    output_path = work_dir / 'puritan_master_graph.gml'
    print(f"Saving master graph to: {output_path}")

    try:
        nx.write_gml(master_graph, str(output_path))
        print(f"✓ Successfully saved master graph")
    except Exception as e:
        print(f"✗ Error saving master graph: {e}")
        sys.exit(1)

    # Print final statistics
    print_statistics(stats)

    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"\nOutput file: {output_path.name}")
    print(f"Total nodes: {stats['nodes']:,}")
    print(f"Total edges: {stats['edges']:,}")
    print()


if __name__ == "__main__":
    main()

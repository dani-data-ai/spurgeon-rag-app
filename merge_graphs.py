"""
Merge multiple NetworkX graph files into a single master graph.

This script finds all .gml files in a specified directory, loads them,
and merges them into a single master graph file.
"""

import networkx as nx
from pathlib import Path
from typing import List

# Configuration
GRAPHS_DIRECTORY = '.'  # Current directory

def find_graph_files(directory: str) -> List[Path]:
    """Find all .gml files in the specified directory."""
    graph_dir = Path(directory)
    if not graph_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all .gml files but exclude the master graph to avoid circular merging
    gml_files = [
        f for f in graph_dir.rglob('*.gml')
        if f.name != 'puritan_master_graph.gml'
    ]
    return sorted(gml_files)

def merge_graphs(file_paths: List[Path]) -> nx.Graph:
    """Load and merge all graphs from the given file paths."""
    master_graph = nx.Graph()

    for file_path in file_paths:
        print(f"Loading: {file_path.name}")
        try:
            graph = nx.read_gml(file_path)

            # Print graph stats before merging
            print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

            # Use nx.compose to properly merge graphs (preserves both nodes and edges)
            master_graph = nx.compose(master_graph, graph)

            print(f"  Master graph now has: {master_graph.number_of_nodes()} nodes, {master_graph.number_of_edges()} edges")

        except Exception as e:
            print(f"  Warning: Failed to load {file_path.name}: {e}")
            continue

    return master_graph

def main():
    """Main execution function."""
    print("=" * 60)
    print("Graph Merging Script")
    print("=" * 60)
    print(f"\nSearching for .gml files in: {GRAPHS_DIRECTORY}\n")

    # Find all graph files
    graph_files = find_graph_files(GRAPHS_DIRECTORY)

    if not graph_files:
        print("No .gml files found!")
        return

    print(f"Found {len(graph_files)} graph file(s):\n")
    for file_path in graph_files:
        print(f"  - {file_path.relative_to(GRAPHS_DIRECTORY)}")

    print("\n" + "-" * 60)
    print("Merging graphs...")
    print("-" * 60 + "\n")

    # Merge all graphs
    master_graph = merge_graphs(graph_files)

    # Save the master graph
    output_path = Path('puritan_master_graph.gml')
    print(f"\nSaving master graph to: {output_path.absolute()}")
    nx.write_gml(master_graph, str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"\nFiles merged: {len(graph_files)}")
    print(f"Total nodes: {master_graph.number_of_nodes()}")
    print(f"Total edges: {master_graph.number_of_edges()}")
    print(f"\nMaster graph saved to: {output_path.absolute()}")
    print()

if __name__ == "__main__":
    main()

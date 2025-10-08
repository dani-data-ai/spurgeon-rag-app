"""
Diagnostic script to analyze all checkpoint files in the project directory.

This script searches for checkpoint .gml files, loads them, and reports
their statistics and graph properties to identify any inconsistencies.
"""

import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional
import traceback


def analyze_checkpoint(file_path: Path) -> Optional[Dict]:
    """
    Analyze a single checkpoint file.

    Args:
        file_path: Path to the checkpoint file

    Returns:
        Dictionary with analysis results, or None if loading failed
    """
    result = {
        "filename": file_path.name,
        "path": str(file_path),
        "success": False,
        "error": None,
        "nodes": 0,
        "edges": 0,
        "is_directed": None,
        "graph_type": None
    }

    try:
        print(f"\n{'='*60}")
        print(f"Analyzing: {file_path.name}")
        print(f"{'='*60}")

        # Load the graph
        graph = nx.read_gml(str(file_path))

        # Get graph information
        result["nodes"] = graph.number_of_nodes()
        result["edges"] = graph.number_of_edges()
        result["is_directed"] = graph.is_directed()
        result["graph_type"] = type(graph).__name__
        result["success"] = True

        # Print information
        print(f"✓ Successfully loaded")
        print(f"  Nodes: {result['nodes']:,}")
        print(f"  Edges: {result['edges']:,}")
        print(f"  Graph Type: {result['graph_type']}")
        print(f"  Directed: {result['is_directed']}")

        # Additional statistics
        if result["nodes"] > 0:
            if result["is_directed"]:
                avg_in_degree = sum(dict(graph.in_degree()).values()) / result["nodes"]
                avg_out_degree = sum(dict(graph.out_degree()).values()) / result["nodes"]
                print(f"  Avg In-Degree: {avg_in_degree:.2f}")
                print(f"  Avg Out-Degree: {avg_out_degree:.2f}")
            else:
                avg_degree = sum(dict(graph.degree()).values()) / result["nodes"]
                print(f"  Avg Degree: {avg_degree:.2f}")

            # Check for isolated nodes
            if result["is_directed"]:
                isolated = [n for n in graph.nodes() if graph.in_degree(n) == 0 and graph.out_degree(n) == 0]
            else:
                isolated = [n for n in graph.nodes() if graph.degree(n) == 0]

            if isolated:
                print(f"  Isolated Nodes: {len(isolated):,}")

        return result

    except Exception as e:
        result["error"] = str(e)
        print(f"✗ Error loading file")
        print(f"  Error: {str(e)}")
        print(f"  Traceback:")
        print(traceback.format_exc())
        return result


def main():
    """Main execution function."""
    print("="*60)
    print("CHECKPOINT FILE DIAGNOSTIC TOOL")
    print("="*60)

    # Search for checkpoint files
    search_dir = Path("C:/Users/danieo/spurgeon-rag-app")
    print(f"\nSearching in: {search_dir}")

    checkpoint_files = sorted(search_dir.glob("checkpoint_*.gml"))

    if not checkpoint_files:
        print("\n⚠ No checkpoint files found!")
        return

    print(f"\nFound {len(checkpoint_files)} checkpoint file(s):")
    for f in checkpoint_files:
        print(f"  - {f.name}")

    # Analyze each checkpoint
    results: List[Dict] = []
    for file_path in checkpoint_files:
        result = analyze_checkpoint(file_path)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal files analyzed: {len(results)}")
    print(f"Successfully loaded: {len(successful)}")
    print(f"Failed to load: {len(failed)}")

    if failed:
        print("\n⚠ Failed files:")
        for r in failed:
            print(f"  - {r['filename']}: {r['error']}")

    if successful:
        # Check for inconsistencies in graph types
        graph_types = set(r["graph_type"] for r in successful)
        directed_status = set(r["is_directed"] for r in successful)

        print("\nGraph Type Consistency:")
        if len(graph_types) == 1:
            print(f"  ✓ All files use the same graph type: {list(graph_types)[0]}")
        else:
            print(f"  ✗ Inconsistent graph types found:")
            for gt in graph_types:
                files = [r["filename"] for r in successful if r["graph_type"] == gt]
                print(f"    - {gt}: {', '.join(files)}")

        print("\nDirected Graph Consistency:")
        if len(directed_status) == 1:
            status = "Directed" if list(directed_status)[0] else "Undirected"
            print(f"  ✓ All files are {status}")
        else:
            print(f"  ✗ Inconsistent directed status found:")
            directed_files = [r["filename"] for r in successful if r["is_directed"]]
            undirected_files = [r["filename"] for r in successful if not r["is_directed"]]
            if directed_files:
                print(f"    - Directed: {', '.join(directed_files)}")
            if undirected_files:
                print(f"    - Undirected: {', '.join(undirected_files)}")

        # Statistics summary
        total_nodes = sum(r["nodes"] for r in successful)
        total_edges = sum(r["edges"] for r in successful)

        print("\nAggregate Statistics:")
        print(f"  Total Nodes (all files): {total_nodes:,}")
        print(f"  Total Edges (all files): {total_edges:,}")
        print(f"  Average Nodes per file: {total_nodes / len(successful):,.0f}")
        print(f"  Average Edges per file: {total_edges / len(successful):,.0f}")

        # Size comparison
        print("\nFile Size Comparison:")
        for r in sorted(successful, key=lambda x: x["nodes"], reverse=True):
            print(f"  {r['filename']:<40} {r['nodes']:>10,} nodes, {r['edges']:>10,} edges")

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

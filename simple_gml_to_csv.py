"""
Simple GML to CSV converter using NetworkX with comprehensive validation
Processes one file at a time to stay under 28GB RAM limit
"""

import networkx as nx
import csv
import gc
import time
from pathlib import Path

# Input files in order (smallest to largest for testing)
INPUT_FILES = [
    "checkpoint_spurgeon.gml",              # 2.0 GB
    "checkpoint_graph_reference_books.gml", # 2.8 GB
    "checkpoint_Epub_Vol_1-25.gml",         # 4.1 GB
    "checkpoint_Epub_Vol_80-125.gml",       # 6.5 GB (largest - test last)
    "checkpoint_Epub_Vol_26-79.gml",        # 5.9 GB
]

# Global node ID registry (tracks ALL node IDs across files)
GLOBAL_NODE_IDS = set()


def export_graph_to_csv(G, basename):
    """Export NetworkX graph to Neo4j-compatible CSVs with validation."""

    print(f"  Exporting {G.number_of_nodes():,} nodes...")

    # Validation tracking
    validation = {
        'nodes_written': 0,
        'node_ids_seen': set(),
        'duplicate_node_ids': 0,
        'nodes_missing_id': 0,
        'malformed_node_rows': 0,
        'edges_written': 0,
        'edges_missing_endpoints': 0,
        'edges_missing_weight': 0,
        'edges_empty_relationship': 0,
        'malformed_edge_rows': 0
    }

    # Write nodes
    nodes_file = f"{basename}_nodes.csv"
    with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'id:ID', ':LABEL', 'source_file:STRING',
            'text:STRING', 'entities:STRING', 'keywords:STRING',
            'figures:STRING', 'parent_id:STRING', 'label:STRING',
            'original_source_file:STRING'
        ]
        writer.writerow(header)

        # Nodes
        for node_id, node_data in G.nodes(data=True):
            # Validate node has ID
            if node_id is None or str(node_id).strip() == '':
                validation['nodes_missing_id'] += 1
                continue

            # Always prefix with basename for global uniqueness
            final_id = f"{basename}_{node_id}"

            # Check for duplicate IDs within this file
            if final_id in validation['node_ids_seen']:
                validation['duplicate_node_ids'] += 1
                continue
            validation['node_ids_seen'].add(final_id)

            # Check for duplicate IDs across ALL files
            if final_id in GLOBAL_NODE_IDS:
                print(f"    ERROR: Global duplicate node ID: {final_id}")
                validation['duplicate_node_ids'] += 1
                continue
            GLOBAL_NODE_IDS.add(final_id)

            # Build row with safe string conversion
            row = [
                final_id,
                'Node',
                f"{basename}.gml",
                str(node_data.get('text', ''))[:1000],  # Truncate to 1000 chars
                str(node_data.get('entities', '')),
                str(node_data.get('keywords', '')),
                str(node_data.get('figures', '')),
                str(node_data.get('parent_id', '')),
                str(node_data.get('label', '')),
                str(node_data.get('source_file', ''))  # Original PDF source
            ]

            # Validate row has correct number of columns
            if len(row) != len(header):
                validation['malformed_node_rows'] += 1
                continue

            # Check for missing required fields (id, label, source_file)
            if not row[0] or not row[1] or not row[2]:
                validation['malformed_node_rows'] += 1
                continue

            writer.writerow(row)
            validation['nodes_written'] += 1

    print(f"  Wrote {nodes_file} ({validation['nodes_written']:,} nodes)")

    # Report node validation issues
    if validation['duplicate_node_ids'] > 0:
        print(f"    WARNING:  Skipped {validation['duplicate_node_ids']} duplicate node IDs")
    if validation['nodes_missing_id'] > 0:
        print(f"    WARNING:  Skipped {validation['nodes_missing_id']} nodes with missing IDs")
    if validation['malformed_node_rows'] > 0:
        print(f"    WARNING:  Skipped {validation['malformed_node_rows']} malformed node rows")

    print(f"  Exporting {G.number_of_edges():,} edges...")

    # Write edges
    edges_file = f"{basename}_edges.csv"
    with open(edges_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        edge_header = [
            ':START_ID', ':END_ID', 'relationship_type:STRING',
            'weight:FLOAT', 'source_file:STRING', 'shared_items:STRING'
        ]
        writer.writerow(edge_header)

        # Edges
        for source, target, edge_data in G.edges(data=True):
            # Validate endpoints exist
            if source is None or target is None:
                validation['edges_missing_endpoints'] += 1
                continue

            # Apply same prefix to endpoints
            source_id = f"{basename}_{source}"
            target_id = f"{basename}_{target}"

            # Validate endpoints are in node set
            if source_id not in validation['node_ids_seen']:
                validation['edges_missing_endpoints'] += 1
                continue
            if target_id not in validation['node_ids_seen']:
                validation['edges_missing_endpoints'] += 1
                continue

            # Get relationship type (try 'type' first, then 'relationship_type')
            rel_type = edge_data.get('type', edge_data.get('relationship_type', ''))
            if not rel_type:
                validation['edges_empty_relationship'] += 1

            # Get weight
            weight = edge_data.get('weight', '')
            if weight == '' or weight is None:
                validation['edges_missing_weight'] += 1

            # Build row
            row = [
                source_id,
                target_id,
                str(rel_type),
                str(weight) if weight != '' else '',
                f"{basename}.gml",
                str(edge_data.get('shared_items', ''))
            ]

            # Validate row has correct number of columns
            if len(row) != len(edge_header):
                validation['malformed_edge_rows'] += 1
                continue

            # Check for missing required fields (start_id, end_id)
            if not row[0] or not row[1]:
                validation['malformed_edge_rows'] += 1
                continue

            writer.writerow(row)
            validation['edges_written'] += 1

    print(f"  Wrote {edges_file} ({validation['edges_written']:,} edges)")

    # Report edge validation issues
    if validation['edges_missing_endpoints'] > 0:
        print(f"    WARNING:  Skipped {validation['edges_missing_endpoints']} edges with missing/invalid endpoints")
    if validation['edges_empty_relationship'] > 0:
        print(f"    WARNING:  {validation['edges_empty_relationship']} edges have empty relationship_type")
    if validation['edges_missing_weight'] > 0:
        print(f"    WARNING:  {validation['edges_missing_weight']} edges missing weight")
    if validation['malformed_edge_rows'] > 0:
        print(f"    WARNING:  Skipped {validation['malformed_edge_rows']} malformed edge rows")

    return validation


def main():
    print("="*80)
    print("Simple GML to CSV Converter (NetworkX) with Validation")
    print("="*80)

    overall_start = time.time()
    all_validations = []

    for idx, gml_file in enumerate(INPUT_FILES, 1):
        print(f"\n[{idx}/{len(INPUT_FILES)}] Processing: {gml_file}")

        if not Path(gml_file).exists():
            print(f"  SKIP: File not found")
            continue

        file_size_gb = Path(gml_file).stat().st_size / (1024**3)
        print(f"  Size: {file_size_gb:.2f} GB")

        start_time = time.time()

        # Load graph (this is where RAM is used)
        print(f"  Loading with NetworkX...")
        try:
            G = nx.read_gml(gml_file)
            print(f"  Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        except Exception as e:
            print(f"  ERROR: ERROR loading: {e}")
            continue

        # Export to CSV
        basename = Path(gml_file).stem
        try:
            validation = export_graph_to_csv(G, basename)
            validation['file'] = gml_file
            all_validations.append(validation)
        except Exception as e:
            print(f"  ERROR: ERROR exporting: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Free memory before next file
        del G
        gc.collect()

        elapsed = time.time() - start_time
        print(f"  [OK] Completed in {elapsed/60:.1f} minutes")

    total_time = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")

    # Validation Summary
    print(f"\n[VALIDATION SUMMARY]")
    print(f"{'='*80}")

    total_nodes = sum(v['nodes_written'] for v in all_validations)
    total_edges = sum(v['edges_written'] for v in all_validations)

    print(f"\n[Successfully Written]")
    print(f"  Total Nodes: {total_nodes:,}")
    print(f"  Total Edges: {total_edges:,}")
    print(f"  Global Unique Node IDs: {len(GLOBAL_NODE_IDS):,}")

    # Check for issues
    total_issues = 0
    print(f"\n[Issues Found]")

    for v in all_validations:
        file_issues = (v['duplicate_node_ids'] + v['nodes_missing_id'] +
                      v['malformed_node_rows'] + v['edges_missing_endpoints'] +
                      v['malformed_edge_rows'])
        if file_issues > 0:
            print(f"  {v['file']}:")
            if v['duplicate_node_ids'] > 0:
                print(f"    - {v['duplicate_node_ids']} duplicate node IDs")
            if v['nodes_missing_id'] > 0:
                print(f"    - {v['nodes_missing_id']} nodes missing IDs")
            if v['malformed_node_rows'] > 0:
                print(f"    - {v['malformed_node_rows']} malformed node rows")
            if v['edges_missing_endpoints'] > 0:
                print(f"    - {v['edges_missing_endpoints']} edges with missing endpoints")
            if v['malformed_edge_rows'] > 0:
                print(f"    - {v['malformed_edge_rows']} malformed edge rows")
            total_issues += file_issues

    if total_issues == 0:
        print(f"  None! All CSVs are clean.")
    else:
        print(f"\n  Total Issues: {total_issues:,}")

    print(f"\n{'='*80}")
    print(f"Next step: Import to Neo4j with:")
    print(f"{'='*80}")
    print(f"neo4j-admin database import full \\")
    for gml_file in INPUT_FILES:
        if Path(gml_file).exists():
            basename = Path(gml_file).stem
            print(f"  --nodes={basename}_nodes.csv \\")
    for gml_file in INPUT_FILES:
        if Path(gml_file).exists():
            basename = Path(gml_file).stem
            print(f"  --relationships={basename}_edges.csv \\")
    print(f"  --database=theology_graph")


if __name__ == "__main__":
    main()

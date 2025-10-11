"""
CSV Preprocessor for neo4j-admin bulk import
Converts LOAD CSV format to neo4j-admin format:
1. Removes :LABEL column from node CSVs
2. Fixes header format (id:ID -> id)
3. Splits edges by relationship_type into separate files
4. Validates data integrity

Author: Claude Code
Date: 2025-10-11
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# Input/output directories
INPUT_DIR = Path(r"C:\Users\danieo\Downloads\neo4j-community-2025.09.0-windows\neo4j-community-2025.09.0\import")
OUTPUT_DIR = INPUT_DIR / "neo4j_admin_ready"

# Node and edge file patterns
NODE_FILES = [
    "checkpoint_spurgeon_nodes.csv",
    "checkpoint_graph_reference_books_nodes.csv",
    "checkpoint_Epub_Vol_1-25_nodes.csv",
    "checkpoint_Epub_Vol_80-125_nodes.csv",
    "checkpoint_Epub_Vol_26-79_nodes.csv",
]

EDGE_FILES = [
    "checkpoint_spurgeon_edges.csv",
    "checkpoint_graph_reference_books_edges.csv",
    "checkpoint_Epub_Vol_1-25_edges.csv",
    "checkpoint_Epub_Vol_80-125_edges.csv",
    "checkpoint_Epub_Vol_26-79_edges.csv",
]


def clean_node_header(header):
    """
    Convert LOAD CSV header to neo4j-admin format

    LOAD CSV format:       id:ID,:LABEL,source_file:STRING,text:STRING
    neo4j-admin format:    id:ID,source_file,text

    Rules:
    - Keep id:ID (required)
    - Remove :LABEL column
    - Remove type suffixes (:STRING, :FLOAT) from other columns
    """
    print(f"  [LOG] Original node header: {header[:5]}...")  # Show first 5 columns
    cleaned = []
    for col in header:
        if col == ':LABEL':
            print(f"  [LOG] Removing :LABEL column")
            continue  # Skip :LABEL column
        elif col == 'id:ID':
            cleaned.append('id:ID')  # Keep as-is (required)
        else:
            # Remove type suffix (e.g., "source_file:STRING" -> "source_file")
            cleaned.append(col.split(':')[0])
    print(f"  [LOG] Cleaned node header ({len(cleaned)} columns): {cleaned[:5]}...")
    return cleaned


def process_node_csv(input_file, output_file):
    """Process a single node CSV file"""
    print(f"\n[NODES] Processing {input_file.name}...")
    print(f"  [LOG] Input: {input_file}")
    print(f"  [LOG] Output: {output_file}")

    nodes_written = 0
    nodes_skipped = 0

    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Process header
        original_header = next(reader)
        cleaned_header = clean_node_header(original_header)
        writer.writerow(cleaned_header)
        print(f"  [LOG] Header written to output file")

        # Find :LABEL column index to skip it
        label_col_idx = None
        if ':LABEL' in original_header:
            label_col_idx = original_header.index(':LABEL')
            print(f"  [LOG] :LABEL column found at index {label_col_idx}")

        # Process rows
        print(f"  [LOG] Starting row processing...")
        for row in reader:
            # Skip :LABEL column value
            if label_col_idx is not None:
                cleaned_row = [val for i, val in enumerate(row) if i != label_col_idx]
            else:
                cleaned_row = row

            # Validate ID exists
            if not cleaned_row[0]:
                nodes_skipped += 1
                continue

            writer.writerow(cleaned_row)
            nodes_written += 1

            if nodes_written % 100000 == 0:
                print(f"  [PROGRESS] {nodes_written:,} nodes processed...")

    print(f"  [OK] {nodes_written:,} nodes written to {output_file.name}")
    if nodes_skipped > 0:
        print(f"  [WARNING] {nodes_skipped} nodes skipped (missing ID)")

    return nodes_written


def clean_edge_header(header):
    """
    Convert edge header from LOAD CSV to neo4j-admin format

    LOAD CSV:       :START_ID,:END_ID,relationship_type:STRING,weight:FLOAT
    neo4j-admin:    :START_ID,:END_ID,weight:float

    Rules:
    - Keep :START_ID, :END_ID as-is
    - Remove relationship_type column (goes in filename)
    - Convert :FLOAT to :float, :STRING to :string (lowercase)
    """
    print(f"  [LOG] Original edge header: {header}")
    cleaned = []
    for col in header:
        if col in [':START_ID', ':END_ID']:
            cleaned.append(col)
        elif 'relationship_type' in col.lower():  # GROK FIX: Case-insensitive
            print(f"  [LOG] Skipping relationship_type column: {col}")
            continue  # Skip, will be in filename
        elif ':FLOAT' in col.upper():  # GROK FIX: Case-insensitive
            cleaned.append(col.split(':')[0] + ':float')
        elif ':STRING' in col.upper():  # GROK FIX: Case-insensitive
            cleaned.append(col.split(':')[0] + ':string')
        elif ':' in col:
            # Keep other typed columns, just lowercase type
            parts = col.split(':')
            cleaned.append(f"{parts[0]}:{parts[1].lower()}")
        else:
            cleaned.append(col)
    print(f"  [LOG] Cleaned edge header: {cleaned}")
    return cleaned


def split_edges_by_type(input_file, output_dir):
    """
    Split edge CSV into separate files per relationship_type
    Returns dict of {relationship_type: edge_count}
    """
    print(f"\n[EDGES] Processing {input_file.name}...")
    print(f"  [LOG] Input: {input_file}")

    # First pass: count relationship types
    rel_type_counts = defaultdict(int)
    rel_type_col_idx = None

    with open(input_file, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        print(f"  [LOG] Edge header: {header}")

        # Find relationship_type column (case-insensitive)
        for i, col in enumerate(header):
            if 'relationship_type' in col.lower():
                rel_type_col_idx = i
                print(f"  [LOG] Found relationship_type column at index {i}: {col}")
                break

        if rel_type_col_idx is None:
            print(f"  [ERROR] No relationship_type column found in {input_file.name}")
            print(f"  [ERROR] Available columns: {header}")
            return {}

        print(f"  [LOG] First pass: counting relationship types...")
        for row_num, row in enumerate(reader, start=2):
            rel_type = row[rel_type_col_idx] if len(row) > rel_type_col_idx else ''
            if rel_type:
                rel_type_counts[rel_type] += 1

            if row_num % 1000000 == 0:
                print(f"  [PROGRESS] Scanned {row_num:,} rows...")

    print(f"  [LOG] Found {len(rel_type_counts)} relationship types:")
    for rel_type, count in sorted(rel_type_counts.items()):
        print(f"    - {rel_type}: {count:,} edges")

    # Second pass: write separate files per type
    edge_writers = {}
    edge_files = {}
    edges_written = defaultdict(int)
    edges_skipped = 0

    print(f"  [LOG] Second pass: splitting edges by type...")

    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            original_header = next(reader)
            cleaned_header = clean_edge_header(original_header)

            # Find column indices (GROK FIX: case-insensitive search)
            rel_type_col_idx = None
            start_id_idx = None
            end_id_idx = None

            for i, col in enumerate(original_header):
                if 'relationship_type' in col.lower():
                    rel_type_col_idx = i
                elif col == ':START_ID':
                    start_id_idx = i
                elif col == ':END_ID':
                    end_id_idx = i

            if rel_type_col_idx is None or start_id_idx is None or end_id_idx is None:
                print(f"  [ERROR] Missing required columns!")
                print(f"  [ERROR] rel_type_col_idx: {rel_type_col_idx}, start_id_idx: {start_id_idx}, end_id_idx: {end_id_idx}")
                return {}

            print(f"  [LOG] Column indices: rel_type={rel_type_col_idx}, start={start_id_idx}, end={end_id_idx}")

            for row in reader:
                # Validate
                if len(row) <= max(rel_type_col_idx, start_id_idx, end_id_idx):
                    edges_skipped += 1
                    continue

                rel_type = row[rel_type_col_idx]
                if not rel_type or not row[start_id_idx] or not row[end_id_idx]:
                    edges_skipped += 1
                    continue

                # Create writer for this rel_type if not exists
                if rel_type not in edge_writers:
                    basename = input_file.stem.replace('_edges', '')
                    output_file = output_dir / f"{basename}_edges_{rel_type}.csv"
                    print(f"  [LOG] Creating output file for relationship type '{rel_type}': {output_file.name}")
                    edge_files[rel_type] = open(output_file, 'w', encoding='utf-8', newline='')
                    edge_writers[rel_type] = csv.writer(edge_files[rel_type])
                    edge_writers[rel_type].writerow(cleaned_header)

                # Remove relationship_type column from row
                cleaned_row = [val for i, val in enumerate(row) if i != rel_type_col_idx]
                edge_writers[rel_type].writerow(cleaned_row)
                edges_written[rel_type] += 1

                total_written = sum(edges_written.values())
                if total_written % 1000000 == 0:
                    print(f"  [PROGRESS] {total_written:,} edges written...")

    finally:
        # Close all writers
        for f in edge_files.values():
            f.close()

    print(f"  [OK] {sum(edges_written.values()):,} edges written")
    if edges_skipped > 0:
        print(f"  WARNING: {edges_skipped} edges skipped (missing data)")

    return edges_written


def generate_import_command(node_counts, edge_counts):
    """Generate the neo4j-admin import command"""
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE - neo4j-admin Import Command")
    print("="*80)

    total_nodes = sum(node_counts.values())
    total_edges = sum(sum(counts.values()) for counts in edge_counts.values())

    print(f"\nTotal nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")
    print(f"Estimated import time: 20-40 minutes")

    print("\n" + "-"*80)
    print("COMMAND TO RUN:")
    print("-"*80)
    print()
    print("cd \"C:\\Users\\danieo\\Downloads\\neo4j-community-2025.09.0-windows\\neo4j-community-2025.09.0\\bin\"")
    print()

    # Build command
    cmd = ["neo4j-admin database import full"]

    # Add node files
    for node_file in sorted(OUTPUT_DIR.glob("*_nodes.csv")):
        rel_path = f"..\\import\\neo4j_admin_ready\\{node_file.name}"
        cmd.append(f'  --nodes=Node="{rel_path}" ^')

    # Add edge files
    edge_files_by_type = defaultdict(list)
    for edge_file in sorted(OUTPUT_DIR.glob("*_edges_*.csv")):
        # Extract rel type from filename (e.g., "spurgeon_edges_figure.csv" -> "figure")
        rel_type = edge_file.stem.split('_edges_')[-1].upper()
        rel_path = f"..\\import\\neo4j_admin_ready\\{edge_file.name}"
        edge_files_by_type[rel_type].append(rel_path)

    for rel_type in sorted(edge_files_by_type.keys()):
        for rel_path in edge_files_by_type[rel_type]:
            cmd.append(f'  --relationships={rel_type}="{rel_path}" ^')

    # Add options
    cmd.append('  --multiline-fields=true ^')
    cmd.append('  --skip-bad-relationships=false ^')
    cmd.append('  --skip-duplicate-nodes=false ^')
    cmd.append('  --verbose ^')
    cmd.append('  neo4j')

    print('\n'.join(cmd))

    # Save to file
    command_file = OUTPUT_DIR / "NEO4J_ADMIN_IMPORT_COMMAND.txt"
    with open(command_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("neo4j-admin BULK IMPORT COMMAND\n")
        f.write("="*80 + "\n\n")
        f.write("STEP 1: Navigate to bin directory\n")
        f.write("-"*80 + "\n")
        f.write("cd \"C:\\Users\\danieo\\Downloads\\neo4j-community-2025.09.0-windows\\neo4j-community-2025.09.0\\bin\"\n\n")
        f.write("STEP 2: Run import command\n")
        f.write("-"*80 + "\n")
        f.write('\n'.join(cmd) + "\n\n")
        f.write("STEP 3: Start Neo4j\n")
        f.write("-"*80 + "\n")
        f.write(".\\neo4j.bat console\n\n")
        f.write("STEP 4: Run post-import script\n")
        f.write("-"*80 + "\n")
        f.write("Open http://localhost:7474 and run:\n")
        f.write("C:\\Users\\danieo\\spurgeon-rag-app\\theology_graph_post_import_CORRECT.cypher\n")

    print(f"\nCommand saved to: {command_file}")
    print("\n" + "="*80)


def main():
    print("="*80)
    print("CSV PREPROCESSOR FOR NEO4J-ADMIN BULK IMPORT")
    print("="*80)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n[OK] Output directory created/verified")

    # Process node files
    print("\n" + "="*80)
    print("PHASE 1: PROCESSING NODE FILES")
    print("="*80)

    node_counts = {}
    for node_file_name in NODE_FILES:
        input_file = INPUT_DIR / node_file_name
        output_file = OUTPUT_DIR / node_file_name

        if not input_file.exists():
            print(f"\nWARNING: {node_file_name} not found, skipping...")
            continue

        count = process_node_csv(input_file, output_file)
        node_counts[node_file_name] = count

    # Process edge files
    print("\n" + "="*80)
    print("PHASE 2: PROCESSING EDGE FILES (SPLITTING BY TYPE)")
    print("="*80)

    edge_counts = {}
    for edge_file_name in EDGE_FILES:
        input_file = INPUT_DIR / edge_file_name

        if not input_file.exists():
            print(f"\nWARNING: {edge_file_name} not found, skipping...")
            continue

        counts = split_edges_by_type(input_file, OUTPUT_DIR)
        edge_counts[edge_file_name] = counts

    # Generate import command
    generate_import_command(node_counts, edge_counts)

    print("\n[OK] PREPROCESSING COMPLETE!")
    print("\nNext: Run the command in NEO4J_ADMIN_IMPORT_COMMAND.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Preprocessing stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

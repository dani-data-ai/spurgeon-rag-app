"""
GML→CSV Conversion Script for Neo4j Migration
Stream-converts large GML files to Neo4j-ready CSVs under 28GB RAM.
"""

import os
import sys
import io
import csv
import re
import time
import json
import sqlite3
import psutil
import gc
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
from pathlib import Path

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
os.environ['OMP_NUM_THREADS'] = '7'
os.environ['OPENBLAS_NUM_THREADS'] = '7'
os.environ['MKL_NUM_THREADS'] = '7'
os.environ['NUMEXPR_NUM_THREADS'] = '7'

# ============================================================================
# CONSTANTS
# ============================================================================
INPUT_FILES = [
    "checkpoint_spurgeon.gml",  # Start with smallest file for testing
    # "checkpoint_Epub_Vol_1-25.gml",
    # "checkpoint_Epub_Vol_26-79.gml",
    # "checkpoint_Epub_Vol_80-125.gml",
    # "checkpoint_graph_reference_books.gml"
]

REGISTRY_DB = "node_registry.sqlite"
PROCESSING_REGISTRY = "processing_registry.json"
SUMMARY_FILE = "conversion_summary.txt"

NODE_BATCH_SIZE = 10000
EDGE_BATCH_SIZE = 10000
LOOKUP_CHUNK_SIZE = 1000
MEMORY_THRESHOLD_GB = 25

KNOWN_OPTIONAL_NODE_PROPS = ['text', 'entities', 'keywords', 'figures', 'parent_id', 'unique_id', 'original_source_file']
KNOWN_OPTIONAL_EDGE_PROPS = ['relationship', 'shared_items', 'type']

# ============================================================================
# SQLITE REGISTRY HELPERS
# ============================================================================
class NodeRegistry:
    """Disk-backed node ID registry using SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._open_db()
        self._pending_final = []
        self._pending_original = []

    def _open_db(self):
        """Open database and apply PRAGMAs."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes_final (
                id_final TEXT PRIMARY KEY
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes_original (
                original_id TEXT,
                id_final TEXT,
                file TEXT,
                PRIMARY KEY(original_id, id_final)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_original
            ON nodes_original(original_id)
        """)
        self.conn.commit()

    def add_node(self, original_id: str, final_id: str, file: str):
        """Add node to pending batch."""
        self._pending_final.append((final_id,))
        self._pending_original.append((original_id, final_id, file))

        if len(self._pending_final) >= NODE_BATCH_SIZE:
            self.flush()

    def flush(self):
        """Write pending batches to database."""
        if self._pending_final:
            self.conn.executemany(
                "INSERT OR IGNORE INTO nodes_final(id_final) VALUES (?)",
                self._pending_final
            )
            self._pending_final = []

        if self._pending_original:
            self.conn.executemany(
                "INSERT OR IGNORE INTO nodes_original(original_id, id_final, file) VALUES (?,?,?)",
                self._pending_original
            )
            self._pending_original = []

        self.conn.commit()

    def lookup_original_ids(self, original_ids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Batch lookup of original IDs.
        Returns: {original_id: [(final_id, file), ...]}
        """
        result = {}

        # Process in chunks
        for i in range(0, len(original_ids), LOOKUP_CHUNK_SIZE):
            chunk = original_ids[i:i + LOOKUP_CHUNK_SIZE]
            placeholders = ','.join(['?'] * len(chunk))

            query = f"""
                SELECT original_id, id_final, file
                FROM nodes_original
                WHERE original_id IN ({placeholders})
            """

            for original_id, final_id, file in self.conn.execute(query, chunk):
                if original_id not in result:
                    result[original_id] = []
                result[original_id].append((final_id, file))

        return result

    def close(self):
        """Flush and close database."""
        self.flush()
        if self.conn:
            self.conn.close()

# ============================================================================
# STREAMING GML PARSER (State Machine)
# ============================================================================
class GMLStreamParser:
    """Stream-based GML parser using state machine with bracket tracking."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.file = None
        self.current_line = ""
        self.line_buffer = []

    def __enter__(self):
        self.file = open(self.file_path, 'r', encoding='utf-8', errors='replace')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_progress(self) -> float:
        """Return progress as percentage."""
        if self.file and self.file_size > 0:
            return 100.0 * self.file.tell() / self.file_size
        return 0.0

    def parse_blocks(self, block_type: str):
        """
        Generator that yields blocks of specified type ('node' or 'edge').
        Each block is a dict of properties.
        """
        in_block = False
        bracket_depth = 0
        block_start_line = 0
        block_lines = []

        # Use enumerate to avoid tell() in the hot path
        for line_no, line in enumerate(self.file, 1):
            if not in_block:
                # Robust start check (cheap)
                s = line.lstrip()
                if s.startswith(block_type) and '[' in s:
                    in_block = True
                    block_start_line = line_no
                    bracket_depth = s.count('[') - s.count(']')
                    block_lines = [line]
            else:
                block_lines.append(line)
                bracket_depth += line.count('[') - line.count(']')

                # Block complete
                if bracket_depth == 0:
                    try:
                        block_data = self._parse_block_content(block_lines)
                        yield block_data, block_start_line
                    except Exception as e:
                        yield {'_parse_error': str(e), '_line': block_start_line}, block_start_line

                    in_block = False
                    block_lines = []

    def _parse_block_content(self, lines: List[str]) -> Dict[str, Any]:
        """Parse block lines into dict of properties."""
        block = {}

        # Join all lines and remove outer brackets
        content = ''.join(lines)
        content = re.sub(r'^\s*(node|edge)\s*\[', '', content, flags=re.DOTALL)
        content = re.sub(r'\]\s*$', '', content, flags=re.DOTALL)

        # Parse key-value pairs
        # Handle: key value, key "value", key [list]
        i = 0
        while i < len(content):
            # Skip whitespace
            while i < len(content) and content[i].isspace():
                i += 1
            if i >= len(content):
                break

            # Extract key
            key_start = i
            while i < len(content) and not content[i].isspace():
                i += 1
            key = content[key_start:i]

            if not key:
                break

            # Skip whitespace
            while i < len(content) and content[i].isspace():
                i += 1

            # Extract value
            if i >= len(content):
                break

            if content[i] == '"':
                # Quoted string
                i += 1
                value_start = i
                while i < len(content):
                    if content[i] == '\\' and i + 1 < len(content):
                        i += 2
                        continue
                    if content[i] == '"':
                        break
                    i += 1
                value = content[value_start:i]
                # Unescape
                value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                i += 1
            elif content[i] == '[':
                # List
                list_start = i
                depth = 1
                i += 1
                in_quote = False
                while i < len(content) and depth > 0:
                    if content[i] == '\\' and i + 1 < len(content):
                        i += 2
                        continue
                    if content[i] == '"':
                        in_quote = not in_quote
                    elif not in_quote:
                        if content[i] == '[':
                            depth += 1
                        elif content[i] == ']':
                            depth -= 1
                    i += 1
                list_content = content[list_start+1:i-1]
                # Flatten to comma-separated (parse quoted strings properly)
                list_items = []
                j = 0
                while j < len(list_content):
                    if list_content[j] == '"':
                        j += 1
                        item_start = j
                        while j < len(list_content):
                            if list_content[j] == '\\' and j + 1 < len(list_content):
                                j += 2
                                continue
                            if list_content[j] == '"':
                                break
                            j += 1
                        list_items.append(list_content[item_start:j])
                        j += 1
                    else:
                        j += 1
                value = ','.join(list_items)
            else:
                # Unquoted value
                value_start = i
                while i < len(content) and not content[i].isspace():
                    i += 1
                value = content[value_start:i]

                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass

            block[key] = value

        return block

# ============================================================================
# LOGGING HELPERS
# ============================================================================
class FileLogger:
    """Per-file logging with console output."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.start_time = time.time()

    def log(self, message: str, console: bool = True):
        """Write message to log and optionally console."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {message}\n"
        self.log_file.write(line)
        self.log_file.flush()
        if console:
            print(message)

    def close(self):
        self.log_file.close()

# ============================================================================
# NODE PROCESSING
# ============================================================================
def detect_optional_node_columns(file_path: str, logger: FileLogger, max_blocks: int = 10000) -> Set[str]:
    """Pre-scan to detect optional node properties."""
    logger.log(f"Pre-scanning first {max_blocks} nodes to detect optional columns...")

    detected = set()
    count = 0

    with GMLStreamParser(file_path) as parser:
        for block, offset in parser.parse_blocks('node'):
            if '_parse_error' in block:
                continue

            for key in block.keys():
                # Map GML 'source_file' to 'original_source_file' for CSV
                csv_key = 'original_source_file' if key == 'source_file' else key
                if csv_key in KNOWN_OPTIONAL_NODE_PROPS:
                    detected.add(csv_key)

            count += 1
            if count >= max_blocks:
                break

    logger.log(f"Detected optional columns: {sorted(detected)}")
    return detected

def process_nodes_pass(file_path: str, basename: str, registry: NodeRegistry,
                      logger: FileLogger, optional_cols: Set[str]) -> Dict[str, int]:
    """
    Stream nodes, write CSV, register in database.
    Returns stats dict.
    """
    logger.log("Starting NODES pass...")

    stats = {
        'blocks_seen': 0,
        'nodes_written': 0,
        'skipped_missing_id': 0,
        'null_bytes_replaced': 0
    }

    nodes_csv_path = f"{basename}_nodes.csv"

    with open(nodes_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Build header
        header = ['id:ID', ':LABEL', 'source_file:STRING']
        for col in sorted(optional_cols):
            header.append(f"{col}:STRING")

        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator='\n')
        writer.writerow(header)

        batch = []
        last_progress_time = time.time()

        with GMLStreamParser(file_path) as parser:
            for block, offset in parser.parse_blocks('node'):
                stats['blocks_seen'] += 1

                # Handle parse errors
                if '_parse_error' in block:
                    logger.log(f"Parse error at line {offset}: {block['_parse_error']}", console=False)
                    continue

                # Resolve original_id: prefer 'id', else 'label'
                original_id = None
                if 'id' in block:
                    original_id = str(block['id'])
                elif 'label' in block:
                    original_id = str(block['label'])

                if not original_id:
                    stats['skipped_missing_id'] += 1
                    continue

                # Compute final_id
                final_id = f"{basename}_{original_id}"

                # Build row
                label_val = str(block.get('label', ''))
                row = [final_id, 'Node', f"{basename}.gml"]

                for col in sorted(optional_cols):
                    # Map GML 'source_file' to CSV 'original_source_file' to avoid collision
                    gml_key = 'source_file' if col == 'original_source_file' else col
                    val = block.get(gml_key, '')

                    if col == 'text' and isinstance(val, str) and len(val) > 1000:
                        val = val[:1000]

                    val_str = str(val)
                    # Replace null bytes
                    if '\x00' in val_str:
                        val_str = val_str.replace('\x00', '')
                        stats['null_bytes_replaced'] += 1

                    row.append(val_str)

                batch.append(row)

                # Register node
                registry.add_node(original_id, final_id, basename)

                # Write batch
                if len(batch) >= NODE_BATCH_SIZE:
                    writer.writerows(batch)
                    stats['nodes_written'] += len(batch)
                    batch = []
                    registry.flush()
                    gc.collect()

                    # Progress update (every 10k nodes, skip expensive .tell() call)
                    if time.time() - last_progress_time > 5:
                        rss_gb = psutil.Process().memory_info().rss / (1024**3)
                        logger.log(f"NODES: Written: {stats['nodes_written']:,} | RSS: {rss_gb:.2f}GB")
                        last_progress_time = time.time()

                        # Memory throttle
                        if rss_gb > MEMORY_THRESHOLD_GB:
                            logger.log(f"MEMORY THROTTLE: RSS {rss_gb:.2f}GB > {MEMORY_THRESHOLD_GB}GB")
                            csvfile.flush()
                            gc.collect()

            # Write remaining
            if batch:
                writer.writerows(batch)
                stats['nodes_written'] += len(batch)
                registry.flush()

    logger.log(f"NODES pass complete: {stats['nodes_written']} nodes written from {stats['blocks_seen']} blocks")
    if stats['skipped_missing_id'] > 0:
        logger.log(f"  Skipped {stats['skipped_missing_id']} nodes with missing id/label")

    return stats

# ============================================================================
# EDGE PROCESSING
# ============================================================================
def process_edges_pass(file_path: str, basename: str, registry: NodeRegistry,
                      logger: FileLogger, optional_cols: Set[str]) -> Dict[str, int]:
    """
    Stream edges, resolve endpoints, write CSV.
    Returns stats dict.
    """
    logger.log("Starting EDGES pass...")

    stats = {
        'blocks_seen': 0,
        'edges_written': 0,
        'skipped_missing_endpoints': 0,
        'skipped_ambiguous_endpoints': 0,
        'empty_relationship_type': 0,
        'missing_weight': 0,
        'null_bytes_replaced': 0
    }

    edges_csv_path = f"{basename}_edges.csv"
    ambiguous_samples = []
    missing_samples = []

    with open(edges_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Build header
        header = [':START_ID', ':END_ID', 'relationship_type:STRING', 'weight:FLOAT', 'source_file:STRING']
        for col in sorted(optional_cols):
            header.append(f"{col}:STRING")

        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator='\n')
        writer.writerow(header)

        edge_batch = []
        last_progress_time = time.time()

        with GMLStreamParser(file_path) as parser:
            for block, offset in parser.parse_blocks('edge'):
                stats['blocks_seen'] += 1

                # Handle parse errors
                if '_parse_error' in block:
                    logger.log(f"Parse error at line {offset}: {block['_parse_error']}", console=False)
                    continue

                edge_batch.append((block, offset))

                # Process batch
                if len(edge_batch) >= EDGE_BATCH_SIZE:
                    batch_stats = _process_edge_batch(
                        edge_batch, basename, registry, writer,
                        optional_cols, ambiguous_samples, missing_samples
                    )
                    for key in batch_stats:
                        stats[key] += batch_stats[key]

                    edge_batch = []
                    gc.collect()

                    # Progress update (every 10k edges, skip expensive .tell() call)
                    if time.time() - last_progress_time > 5:
                        rss_gb = psutil.Process().memory_info().rss / (1024**3)
                        logger.log(f"EDGES: Written: {stats['edges_written']:,} | RSS: {rss_gb:.2f}GB")
                        last_progress_time = time.time()

                        if rss_gb > MEMORY_THRESHOLD_GB:
                            logger.log(f"MEMORY THROTTLE: RSS {rss_gb:.2f}GB")
                            csvfile.flush()
                            gc.collect()

            # Process remaining
            if edge_batch:
                batch_stats = _process_edge_batch(
                    edge_batch, basename, registry, writer,
                    optional_cols, ambiguous_samples, missing_samples
                )
                for key in batch_stats:
                    stats[key] += batch_stats[key]

    logger.log(f"EDGES pass complete: {stats['edges_written']} edges written from {stats['blocks_seen']} blocks")
    if stats['skipped_missing_endpoints'] > 0:
        logger.log(f"  Skipped {stats['skipped_missing_endpoints']} edges with missing endpoints")
        if missing_samples:
            logger.log(f"  Sample missing endpoint lines: {missing_samples[:10]}")
    if stats['skipped_ambiguous_endpoints'] > 0:
        logger.log(f"  Skipped {stats['skipped_ambiguous_endpoints']} edges with ambiguous endpoints")
        if ambiguous_samples:
            logger.log(f"  Sample ambiguous endpoint lines: {ambiguous_samples[:10]}")
    if stats['empty_relationship_type'] > 0:
        logger.log(f"  {stats['empty_relationship_type']} edges with empty relationship_type")
    if stats['missing_weight'] > 0:
        logger.log(f"  {stats['missing_weight']} edges with missing/unparsable weight")

    # Quality warning
    if stats['blocks_seen'] > 0 and stats['empty_relationship_type'] / stats['blocks_seen'] > 0.5:
        logger.log("WARNING: Most edges missing relationship_type; verify GML schema.")

    return stats

def _process_edge_batch(edge_batch: List[Tuple[Dict, int]], basename: str,
                       registry: NodeRegistry, writer, optional_cols: Set[str],
                       ambiguous_samples: List[int], missing_samples: List[int]) -> Dict[str, int]:
    """Process a batch of edges with endpoint resolution."""
    stats = {
        'edges_written': 0,
        'skipped_missing_endpoints': 0,
        'skipped_ambiguous_endpoints': 0,
        'empty_relationship_type': 0,
        'missing_weight': 0,
        'null_bytes_replaced': 0
    }

    # Collect unique endpoint IDs
    endpoint_ids = set()
    for block, offset in edge_batch:
        if 'source' in block:
            endpoint_ids.add(str(block['source']))
        if 'target' in block:
            endpoint_ids.add(str(block['target']))

    # Batch lookup
    endpoint_map = registry.lookup_original_ids(list(endpoint_ids))

    # Process each edge
    rows = []
    for block, offset in edge_batch:
        # Extract endpoints
        source_original = str(block.get('source', ''))
        target_original = str(block.get('target', ''))

        if not source_original or not target_original:
            stats['skipped_missing_endpoints'] += 1
            if len(missing_samples) < 10:
                missing_samples.append(offset)
            continue

        # Resolve source
        source_matches = endpoint_map.get(source_original, [])
        if len(source_matches) == 0:
            stats['skipped_missing_endpoints'] += 1
            if len(missing_samples) < 10:
                missing_samples.append(offset)
            continue
        elif len(source_matches) > 1:
            # Check for hint
            hint = block.get('source_file_hint', '')
            if hint:
                source_matches = [(fid, f) for fid, f in source_matches if f == hint]
            if len(source_matches) != 1:
                stats['skipped_ambiguous_endpoints'] += 1
                if len(ambiguous_samples) < 10:
                    ambiguous_samples.append(offset)
                continue

        source_final = source_matches[0][0]

        # Resolve target
        target_matches = endpoint_map.get(target_original, [])
        if len(target_matches) == 0:
            stats['skipped_missing_endpoints'] += 1
            if len(missing_samples) < 10:
                missing_samples.append(offset)
            continue
        elif len(target_matches) > 1:
            hint = block.get('source_file_hint', '')
            if hint:
                target_matches = [(fid, f) for fid, f in target_matches if f == hint]
            if len(target_matches) != 1:
                stats['skipped_ambiguous_endpoints'] += 1
                if len(ambiguous_samples) < 10:
                    ambiguous_samples.append(offset)
                continue

        target_final = target_matches[0][0]

        # Extract properties (try 'relationship_type' first, then fall back to 'type')
        relationship_type = str(block.get('relationship_type', '') or block.get('type', ''))
        if not relationship_type:
            stats['empty_relationship_type'] += 1

        weight = block.get('weight', '')
        if weight == '' or weight is None:
            stats['missing_weight'] += 1
            weight = ''
        else:
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                stats['missing_weight'] += 1
                weight = ''

        # Build row
        row = [source_final, target_final, relationship_type, weight, f"{basename}.gml"]

        for col in sorted(optional_cols):
            val = str(block.get(col, ''))
            if '\x00' in val:
                val = val.replace('\x00', '')
                stats['null_bytes_replaced'] += 1
            row.append(val)

        rows.append(row)

    # Write batch
    writer.writerows(rows)
    stats['edges_written'] = len(rows)

    return stats

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def load_processing_registry() -> Dict:
    """Load existing processing registry."""
    if os.path.exists(PROCESSING_REGISTRY):
        with open(PROCESSING_REGISTRY, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'files_processed': [], 'registry': REGISTRY_DB}

def save_processing_registry(registry_data: Dict):
    """Save processing registry."""
    with open(PROCESSING_REGISTRY, 'w', encoding='utf-8') as f:
        json.dump(registry_data, f, indent=2)

def should_process_file(file_path: str, registry_data: Dict) -> bool:
    """Check if file needs processing based on size and mtime."""
    if not os.path.exists(file_path):
        return False

    size = os.path.getsize(file_path)
    mtime = os.path.getmtime(file_path)

    for entry in registry_data['files_processed']:
        if entry['path'] == file_path:
            if entry['size_bytes'] == size and entry['mtime'] == mtime:
                return False

    return True

def process_file(file_path: str, registry: NodeRegistry, registry_data: Dict) -> Dict:
    """Process a single GML file."""
    basename = Path(file_path).stem
    log_path = f"conversion_log_{basename}.txt"

    logger = FileLogger(log_path)
    logger.log(f"{'='*80}")
    logger.log(f"Processing: {file_path}")
    logger.log(f"Size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    logger.log(f"{'='*80}")

    start_time = time.time()

    # Detect optional columns
    optional_node_cols = detect_optional_node_columns(file_path, logger)
    optional_edge_cols = set(col for col in KNOWN_OPTIONAL_EDGE_PROPS)

    # Process nodes
    node_stats = process_nodes_pass(file_path, basename, registry, logger, optional_node_cols)

    # Process edges
    edge_stats = process_edges_pass(file_path, basename, registry, logger, optional_edge_cols)

    elapsed = time.time() - start_time
    logger.log(f"{'='*80}")
    logger.log(f"File processing complete in {elapsed/60:.1f} minutes")
    logger.log(f"{'='*80}")

    # Build result
    result = {
        'path': file_path,
        'basename': basename,
        'size_bytes': os.path.getsize(file_path),
        'mtime': os.path.getmtime(file_path),
        'nodes_seen': node_stats['blocks_seen'],
        'nodes_written': node_stats['nodes_written'],
        'edges_seen': edge_stats['blocks_seen'],
        'edges_written': edge_stats['edges_written'],
        'skipped_nodes_missing_id': node_stats['skipped_missing_id'],
        'skipped_edges_missing_endpoints': edge_stats['skipped_missing_endpoints'],
        'ambiguous_endpoints': edge_stats['skipped_ambiguous_endpoints'],
        'empty_relationship_type': edge_stats['empty_relationship_type'],
        'null_bytes_replaced': node_stats['null_bytes_replaced'] + edge_stats['null_bytes_replaced'],
        'timestamp_utc': datetime.utcnow().isoformat()
    }

    logger.close()
    return result

def main():
    """Main conversion process."""
    print("="*80)
    print("GML to CSV Converter for Neo4j")
    print("="*80)

    # Load registry
    registry_data = load_processing_registry()
    registry = NodeRegistry(REGISTRY_DB)

    total_start = time.time()
    results = []

    try:
        for idx, file_path in enumerate(INPUT_FILES, 1):
            print(f"\n[{idx}/{len(INPUT_FILES)}] {file_path}")

            if not os.path.exists(file_path):
                print(f"  SKIP: File not found")
                continue

            if not should_process_file(file_path, registry_data):
                print(f"  SKIP: Already processed (size/mtime match)")
                continue

            result = process_file(file_path, registry, registry_data)
            results.append(result)

            # Update registry
            registry_data['files_processed'] = [
                e for e in registry_data['files_processed']
                if e['path'] != file_path
            ]
            registry_data['files_processed'].append(result)
            save_processing_registry(registry_data)

    finally:
        registry.close()

    # Write summary
    total_elapsed = time.time() - total_start
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("GML→CSV Conversion Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total processing time: {total_elapsed/60:.1f} minutes\n")
        f.write(f"Files processed: {len(results)}\n\n")

        total_nodes = sum(r['nodes_written'] for r in results)
        total_edges = sum(r['edges_written'] for r in results)
        f.write(f"Total nodes: {total_nodes:,}\n")
        f.write(f"Total edges: {total_edges:,}\n\n")

        f.write("Per-file breakdown:\n")
        for r in results:
            f.write(f"\n{r['basename']}:\n")
            f.write(f"  Nodes: {r['nodes_written']:,}\n")
            f.write(f"  Edges: {r['edges_written']:,}\n")
            f.write(f"  Skipped nodes: {r['skipped_nodes_missing_id']}\n")
            f.write(f"  Skipped edges (missing): {r['skipped_edges_missing_endpoints']}\n")
            f.write(f"  Skipped edges (ambiguous): {r['ambiguous_endpoints']}\n")

    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  - {REGISTRY_DB}")
    print(f"  - {PROCESSING_REGISTRY}")
    print(f"  - {SUMMARY_FILE}")
    for r in results:
        print(f"  - {r['basename']}_nodes.csv")
        print(f"  - {r['basename']}_edges.csv")
        print(f"  - conversion_log_{r['basename']}.txt")

    print(f"\nNext step: Import to Neo4j with:")
    print(f"  neo4j-admin database import full \\")
    for r in results:
        print(f"    --nodes={r['basename']}_nodes.csv \\")
    for r in results:
        print(f"    --relationships={r['basename']}_edges.csv \\")
    print(f"    --database=theology_graph")

if __name__ == "__main__":
    main()

"""
Quick test of GML parser on real data
"""
import sys
sys.path.insert(0, '.')

from gml_to_csv_converter import GMLStreamParser

print("Testing GML Parser on test_sample.gml")
print("="*80)

with GMLStreamParser("test_sample.gml") as parser:
    print("\n--- First 3 NODES ---")
    node_count = 0
    for block, offset in parser.parse_blocks('node'):
        if '_parse_error' in block:
            print(f"ERROR at offset {offset}: {block['_parse_error']}")
            continue

        print(f"\nNode {node_count} (offset {offset}):")
        for key, val in block.items():
            val_str = str(val)[:60]
            print(f"  {key}: {val_str}")

        node_count += 1
        if node_count >= 3:
            break

print("\n" + "="*80)

with GMLStreamParser("test_sample.gml") as parser:
    print("\n--- First 3 EDGES ---")
    edge_count = 0
    for block, offset in parser.parse_blocks('edge'):
        if '_parse_error' in block:
            print(f"ERROR at offset {offset}: {block['_parse_error']}")
            continue

        print(f"\nEdge {edge_count} (offset {offset}):")
        for key, val in block.items():
            print(f"  {key}: {val}")

        edge_count += 1
        if edge_count >= 3:
            break

print("\n" + "="*80)
print("Parser test complete!")

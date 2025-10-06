import os
import sys
import re
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Directory to clean
organized_dir = r"C:\Users\danieo\Downloads\sp-library-organized"

# Pattern to match files ending with _number.pdf
duplicate_pattern = re.compile(r'_(\d+)\.pdf$', re.IGNORECASE)

deleted_files = []
deleted_count = 0

print("Scanning for duplicate files...")

for root, dirs, files in os.walk(organized_dir):
    for filename in files:
        if duplicate_pattern.search(filename):
            file_path = os.path.join(root, filename)
            try:
                # Get file size before deletion
                file_size = os.path.getsize(file_path)
                # Delete the file
                os.remove(file_path)
                deleted_files.append({
                    'path': os.path.relpath(file_path, organized_dir),
                    'size': file_size
                })
                deleted_count += 1
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")

# Calculate total size freed
total_size = sum(f['size'] for f in deleted_files)
total_size_mb = total_size / (1024 * 1024)

# Write deletion log
log_path = os.path.join(organized_dir, "deleted_duplicates.log")
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"Deleted Duplicates Log\n")
    f.write(f"{'='*70}\n")
    f.write(f"Total files deleted: {deleted_count}\n")
    f.write(f"Total space freed: {total_size_mb:.2f} MB\n\n")
    f.write(f"Deleted Files:\n")
    f.write(f"{'-'*70}\n")
    for item in sorted(deleted_files, key=lambda x: x['path']):
        f.write(f"{item['path']}\n")

print(f"\n{'='*70}")
print(f"CLEANUP COMPLETE")
print(f"{'='*70}")
print(f"Total duplicates deleted: {deleted_count}")
print(f"Total space freed: {total_size_mb:.2f} MB")
print(f"Log saved to: {log_path}")

import os
import sys
import re

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Directory to process
organized_dir = r"C:\Users\danieo\Downloads\sp-library-organized"

# Pattern to match files starting with "autor-"
autor_pattern = re.compile(r'^autor-', re.IGNORECASE)

renamed_files = []
renamed_count = 0

print("Scanning for files with 'autor-' prefix...")

for root, dirs, files in os.walk(organized_dir):
    for filename in files:
        if filename.lower().endswith('.pdf') and autor_pattern.match(filename):
            old_path = os.path.join(root, filename)
            # Remove the "autor-" prefix
            new_filename = autor_pattern.sub('', filename)
            new_path = os.path.join(root, new_filename)

            try:
                # Check if destination file already exists
                if os.path.exists(new_path):
                    print(f"Skipped (already exists): {filename} -> {new_filename}")
                    continue

                # Rename the file
                os.rename(old_path, new_path)
                renamed_files.append({
                    'old': filename,
                    'new': new_filename,
                    'folder': os.path.relpath(root, organized_dir)
                })
                renamed_count += 1
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Write rename log
log_path = os.path.join(organized_dir, "autor_prefix_removed.log")
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"Removed 'autor-' Prefix Log\n")
    f.write(f"{'='*70}\n")
    f.write(f"Total files renamed: {renamed_count}\n\n")
    f.write(f"Renamed Files:\n")
    f.write(f"{'-'*70}\n")
    for item in sorted(renamed_files, key=lambda x: x['new']):
        folder = item['folder'] if item['folder'] != '.' else 'root'
        f.write(f"Old: {item['old']}\n")
        f.write(f"New: {item['new']}\n")
        f.write(f"Folder: {folder}\n\n")

print(f"\n{'='*70}")
print(f"CLEANUP COMPLETE")
print(f"{'='*70}")
print(f"Total files renamed: {renamed_count}")
print(f"Log saved to: {log_path}")

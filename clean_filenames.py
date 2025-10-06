import os
import sys
import re

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Directory to process
organized_dir = r"C:\Users\danieo\Downloads\sp-library-organized"

# Prefixes to remove
PREFIXES_TO_REMOVE = ['texte-', 'this-', 'THE-', 'întrebări-', 'Text-', 'text-']

def clean_filename(filename):
    """Clean and optimize a filename"""
    if not filename.lower().endswith('.pdf'):
        return filename, []

    name = filename[:-4]  # Remove .pdf extension
    changes = []
    original_name = name

    # Step 1: Replace multiple consecutive underscores with single underscore
    if '__' in name:
        new_name = re.sub(r'_+', '_', name)
        if new_name != name:
            changes.append("Removed multiple underscores")
            name = new_name

    # Step 2: Remove common prefixes (case-insensitive)
    for prefix in PREFIXES_TO_REMOVE:
        pattern = re.compile(f'^{re.escape(prefix)}', re.IGNORECASE)
        if pattern.match(name):
            name = pattern.sub('', name)
            changes.append(f"Removed prefix '{prefix}'")
            break

    # Step 3: Handle Author-Title duplicate names
    # Pattern: Author-Title_with_Author_name_in_it
    # Examples:
    # Thomas-Sfinţenia_nou-testamentară_de_Thomas_Cook -> Sfinţenia_nou-testamentară_de_Thomas_Cook
    # JAMES-George_Whitefield_MA_Field-Preacher_By_JAMES_PATERSON -> George_Whitefield_MA_Field-Preacher_By_JAMES_PATERSON

    if '-' in name:
        parts = name.split('-', 1)
        if len(parts) == 2:
            potential_author = parts[0].strip()
            rest = parts[1].strip()

            # Extract first name from potential_author (e.g., "JAMES PATERSON" -> "JAMES")
            author_first_name = potential_author.split('_')[0].split()[0]

            # Check if the author name appears later in the title
            # Look for patterns like "de_AuthorName" or "By_AuthorName" or "AuthorName_"
            patterns_to_check = [
                rf'\b(de|by|De|By|BY|autor)\s*[_\s]*{re.escape(potential_author)}',  # de Thomas, by James (full name)
                rf'\b(de|by|De|By|BY|autor)\s*[_\s]*{re.escape(author_first_name)}',  # de Thomas, by James (first name)
                rf'^{re.escape(potential_author)}_',  # Duplicate at start after hyphen
            ]

            for pattern in patterns_to_check:
                if re.search(pattern, rest, re.IGNORECASE):
                    name = rest
                    changes.append(f"Removed duplicate author '{potential_author}'")
                    break

    # Step 4: Remove duplicate word sequences
    # Example: "umilire-CALEA_CRUCII_Mai_presus_de_umilire_CALEA_CRUCII"
    # Split by common separators and check for duplicates
    words = re.split(r'[-_\s]+', name)

    # Check for repeating sequences (2-5 words)
    for seq_len in range(5, 1, -1):  # Check longer sequences first
        for i in range(len(words) - seq_len):
            sequence = words[i:i+seq_len]
            # Look for this sequence appearing later
            sequence_str = '_'.join(sequence)
            remaining_words = words[i+seq_len:]

            for j in range(len(remaining_words) - seq_len + 1):
                check_sequence = remaining_words[j:j+seq_len]
                if [w.lower() for w in sequence] == [w.lower() for w in check_sequence]:
                    # Found duplicate - remove first occurrence
                    words = words[:i] + words[i+seq_len:]
                    changes.append(f"Removed duplicate phrase '{sequence_str}'")
                    name = '_'.join(words)
                    break
            if len(words) != len(re.split(r'[-_\s]+', name)):
                break

    # Step 5: Limit title length to 80 characters max
    if len(name) > 80:
        # Try to cut at a word boundary
        truncated = name[:80]
        last_underscore = truncated.rfind('_')
        last_hyphen = truncated.rfind('-')
        cut_point = max(last_underscore, last_hyphen)

        if cut_point > 40:  # Only cut at word boundary if it's not too short
            name = name[:cut_point]
        else:
            name = name[:80]

        changes.append(f"Truncated to 80 characters")

    # Step 6: Clean up any trailing/leading underscores or hyphens
    name = name.strip('_-')

    # Step 7: Final cleanup - remove any remaining multiple underscores
    name = re.sub(r'_+', '_', name)

    new_filename = name + '.pdf'

    return new_filename, changes

def process_directory(directory):
    """Process all PDF files in directory"""
    renamed_files = []
    skipped_files = []
    error_files = []

    print("Scanning for files to clean...")
    print("=" * 70)

    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            if not filename.lower().endswith('.pdf'):
                continue

            old_path = os.path.join(root, filename)
            new_filename, changes = clean_filename(filename)

            if not changes:
                # No changes needed
                continue

            new_path = os.path.join(root, new_filename)

            # Check if new filename already exists
            if os.path.exists(new_path) and new_path != old_path:
                skipped_files.append({
                    'old': filename,
                    'new': new_filename,
                    'reason': 'File already exists',
                    'folder': os.path.relpath(root, directory)
                })
                print(f"⚠ Skipped (exists): {filename}")
                continue

            try:
                os.rename(old_path, new_path)
                renamed_files.append({
                    'old': filename,
                    'new': new_filename,
                    'changes': changes,
                    'folder': os.path.relpath(root, directory)
                })
                print(f"✓ Renamed: {filename}")
                print(f"  → {new_filename}")
                print(f"  Changes: {', '.join(changes)}")
                print()
            except Exception as e:
                error_files.append({
                    'old': filename,
                    'error': str(e),
                    'folder': os.path.relpath(root, directory)
                })
                print(f"✗ Error: {filename} - {e}")

    return renamed_files, skipped_files, error_files

def write_log(directory, renamed_files, skipped_files, error_files):
    """Write detailed log file"""
    log_path = os.path.join(directory, "filename_cleaning.log")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Filename Cleaning Log\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Summary:\n")
        f.write(f"  Files renamed: {len(renamed_files)}\n")
        f.write(f"  Files skipped: {len(skipped_files)}\n")
        f.write(f"  Errors: {len(error_files)}\n\n")

        if renamed_files:
            f.write("\nRenamed Files:\n")
            f.write("-" * 70 + "\n")
            for item in renamed_files:
                folder = item['folder'] if item['folder'] != '.' else 'root'
                f.write(f"\nFolder: {folder}\n")
                f.write(f"Old: {item['old']}\n")
                f.write(f"New: {item['new']}\n")
                f.write(f"Changes: {', '.join(item['changes'])}\n")

        if skipped_files:
            f.write("\n\nSkipped Files:\n")
            f.write("-" * 70 + "\n")
            for item in skipped_files:
                folder = item['folder'] if item['folder'] != '.' else 'root'
                f.write(f"\nFolder: {folder}\n")
                f.write(f"File: {item['old']}\n")
                f.write(f"Reason: {item['reason']}\n")

        if error_files:
            f.write("\n\nErrors:\n")
            f.write("-" * 70 + "\n")
            for item in error_files:
                folder = item['folder'] if item['folder'] != '.' else 'root'
                f.write(f"\nFolder: {folder}\n")
                f.write(f"File: {item['old']}\n")
                f.write(f"Error: {item['error']}\n")

    return log_path

# Main execution
if __name__ == "__main__":
    print("Starting filename cleaning process...")
    print()

    renamed, skipped, errors = process_directory(organized_dir)
    log_path = write_log(organized_dir, renamed, skipped, errors)

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)
    print(f"Files renamed: {len(renamed)}")
    print(f"Files skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print(f"\nLog saved to: {log_path}")

import os
import sys
import re

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Directory to process
organized_dir = r"C:\Users\danieo\Downloads\sp-library-organized"

def final_clean_filename(filename):
    """Final cleanup for specific remaining issues"""
    if not filename.lower().endswith('.pdf'):
        return filename, []

    name = filename[:-4]  # Remove .pdf
    original = name
    changes = []

    # Issue 1: Remove author prefix if author name appears after "by" or "de"
    # Examples:
    # Thomas-...de_Thomas_Cook -> de_Thomas_Cook
    # JAMES-...By_JAMES_PATERSON -> By_JAMES_PATERSON
    if '-' in name:
        prefix, rest = name.split('-', 1)

        # Get first word of prefix (author first name)
        author_firstname = prefix.split('_')[0].strip()

        # Check if this name appears after "by" or "de" in the rest
        patterns = [
            (rf'\b(by|By|BY)_+{re.escape(author_firstname)}', f"by {author_firstname}"),
            (rf'\b(de|De)_+{re.escape(author_firstname)}', f"de {author_firstname}"),
        ]

        for pattern, desc in patterns:
            if re.search(pattern, rest, re.IGNORECASE):
                name = rest
                changes.append(f"Removed duplicate author prefix '{prefix}'")
                break

    # Issue 2: Remove "texte-", "this-", "THE-", "întrebări-" prefixes (case insensitive)
    prefixes_to_remove = [
        (r'^texte-', 'texte-'),
        (r'^this-', 'this-'),
        (r'^THE-', 'THE-'),
        (r'^întrebări-', 'întrebări-'),
        (r'^Text-', 'Text-'),
    ]

    for pattern, prefix_name in prefixes_to_remove:
        if re.match(pattern, name, re.IGNORECASE):
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
            changes.append(f"Removed prefix '{prefix_name}'")
            break

    # Issue 3: Clean up multiple underscores
    if '__' in name or '___' in name:
        name = re.sub(r'_+', '_', name)
        changes.append("Cleaned multiple underscores")

    # Issue 4: Truncate to 80 characters at word boundary
    if len(name) > 80:
        truncated = name[:80]
        # Try to cut at underscore or hyphen
        last_sep = max(truncated.rfind('_'), truncated.rfind('-'))
        if last_sep > 40:
            name = name[:last_sep]
        else:
            name = truncated
        changes.append("Truncated to 80 chars")

    # Clean trailing/leading separators
    name = name.strip('_-')

    # Final multi-underscore cleanup
    name = re.sub(r'_+', '_', name)

    new_filename = name + '.pdf'

    if new_filename == filename:
        return filename, []

    return new_filename, changes

def process_all_files(directory):
    """Process all PDF files"""
    renamed = []
    skipped = []
    errors = []

    print("Scanning for files needing final cleanup...")
    print("=" * 70)

    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            if not filename.lower().endswith('.pdf'):
                continue

            old_path = os.path.join(root, filename)
            new_filename, changes = final_clean_filename(filename)

            if not changes:
                continue

            new_path = os.path.join(root, new_filename)

            # Check if target exists
            if os.path.exists(new_path) and new_path != old_path:
                skipped.append({
                    'old': filename,
                    'new': new_filename,
                    'reason': 'File exists',
                    'folder': os.path.relpath(root, directory)
                })
                print(f"⚠ Skipped: {filename} (target exists)")
                continue

            try:
                os.rename(old_path, new_path)
                renamed.append({
                    'old': filename,
                    'new': new_filename,
                    'changes': changes,
                    'folder': os.path.relpath(root, directory)
                })
                print(f"✓ {filename}")
                print(f"  → {new_filename}")
                print(f"  ({', '.join(changes)})")
                print()
            except Exception as e:
                errors.append({
                    'old': filename,
                    'error': str(e),
                    'folder': os.path.relpath(root, directory)
                })
                print(f"✗ Error: {filename} - {e}")

    return renamed, skipped, errors

def write_log(directory, renamed, skipped, errors):
    """Write log file"""
    log_path = os.path.join(directory, "final_cleanup.log")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Final Cleanup Log\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Files renamed: {len(renamed)}\n")
        f.write(f"Files skipped: {len(skipped)}\n")
        f.write(f"Errors: {len(errors)}\n\n")

        if renamed:
            f.write("\nRenamed Files:\n")
            f.write("-" * 70 + "\n")
            for item in renamed:
                f.write(f"\nOld: {item['old']}\n")
                f.write(f"New: {item['new']}\n")
                f.write(f"Changes: {', '.join(item['changes'])}\n")

        if skipped:
            f.write("\n\nSkipped Files:\n")
            f.write("-" * 70 + "\n")
            for item in skipped:
                f.write(f"\n{item['old']}\n")
                f.write(f"Reason: {item['reason']}\n")

        if errors:
            f.write("\n\nErrors:\n")
            f.write("-" * 70 + "\n")
            for item in errors:
                f.write(f"\n{item['old']}\n")
                f.write(f"Error: {item['error']}\n")

    return log_path

if __name__ == "__main__":
    print("Starting final cleanup...")
    print()

    renamed, skipped, errors = process_all_files(organized_dir)
    log_path = write_log(organized_dir, renamed, skipped, errors)

    print()
    print("=" * 70)
    print("FINAL CLEANUP COMPLETE")
    print("=" * 70)
    print(f"Files renamed: {len(renamed)}")
    print(f"Files skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print(f"\nLog saved to: {log_path}")

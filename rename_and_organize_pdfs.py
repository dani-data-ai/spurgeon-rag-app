import fitz  # PyMuPDF for PDF handling
import re
import os
import sys
import shutil
import cv2  # For OCR image preprocessing
import numpy as np
from pytesseract import image_to_string, pytesseract
from PIL import Image
import io
from typing import Tuple, Optional
import csv

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set Tesseract path
pytesseract.tesseract_cmd = r'C:\Users\danieo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy: grayscale, threshold, and contrast enhancement.
    """
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    # Adaptive thresholding for varying scan qualities
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Denoise for old scans
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return Image.fromarray(denoised)

def extract_title_author(pdf_path: str, max_pages: int = 7) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract title and first author from the first max_pages of the PDF.
    Handles both text-based and scanned PDFs.
    Returns (title, author) or (None, None) if extraction fails.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return None, None

    title_candidates = []
    author_candidates = []
    has_text_layer = False

    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        # Check for text layer
        if any("lines" in b for b in blocks):
            has_text_layer = True
            # Sort blocks by font size descending for title/author hierarchy
            text_blocks = sorted(
                [b for b in blocks if "lines" in b],
                key=lambda b: max(span["size"] for line in b["lines"] for span in line["spans"]),
                reverse=True
            )

            for block in text_blocks:
                text = " ".join(span["text"].strip() for line in block["lines"] for span in line["spans"]).strip()
                if not text:
                    continue

                # Position check: prefer centered, upper-page text for titles
                bbox = block["bbox"]
                page_width = page.rect.width
                page_height = page.rect.height
                if (
                    abs((bbox[0] + bbox[2]) / 2 - page_width / 2) < page_width * 0.25  # Roughly centered
                    and bbox[1] < page_height * 0.5  # Upper half
                    and len(text.split()) > 1  # Avoid single words or page numbers
                ):
                    title_candidates.append(text)

                # Author regex: supports English/Romanian patterns, case-insensitive
                author_match = re.search(
                    r"(?:by|de|de către|autor|written by|scris de)\s+([A-ZĂÂÎȘȚ][\w\s\.,&]+(?:\s+[A-ZĂÂÎȘȚ][\w\s\.,&]+)*)",
                    text,
                    re.IGNORECASE | re.UNICODE
                )
                if author_match:
                    # Split multiple authors and take the first one
                    authors = re.split(r'\s*[,;&]\s*|\sand\s', author_match.group(1).strip())
                    first_author = authors[0].strip().split()[0] if authors else None  # First word of first author
                    if first_author:
                        author_candidates.append(first_author)

        # OCR fallback for scanned PDFs or if no text layer
        if not has_text_layer or not title_candidates:
            try:
                pix = page.get_pixmap(dpi=200)  # Higher DPI for better OCR
                img = Image.open(io.BytesIO(pix.tobytes()))
                preprocessed_img = preprocess_image_for_ocr(img)
                ocr_text = image_to_string(preprocessed_img, lang='eng+ron')  # Support English + Romanian

                # Process OCR text similarly
                lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
                for line in lines:
                    if len(line.split()) > 2 and line.isupper():  # Heuristic for titles: longer, uppercase
                        title_candidates.append(line)
                    author_match = re.search(
                        r"(?:by|de|de către|autor|written by|scris de)\s+([A-ZĂÂÎȘȚ][\w\s\.,&]+(?:\s+[A-ZĂÂÎȘȚ][\w\s\.,&]+)*)",
                        line,
                        re.IGNORECASE | re.UNICODE
                    )
                    if author_match:
                        authors = re.split(r'\s*[,;&]\s*|\sand\s', author_match.group(1).strip())
                        first_author = authors[0].strip().split()[0] if authors else None
                        if first_author:
                            author_candidates.append(first_author)
            except Exception:
                # Skip OCR errors silently (Tesseract not installed or other OCR issues)
                pass

    doc.close()

    # Select best candidates
    title = " ".join(title_candidates[:3]).strip() if title_candidates else None  # Merge top 3 for multi-line
    author = author_candidates[0] if author_candidates else None

    # Confidence check: Fail if no title/author or too short/ambiguous
    if not title or not author or len(title) < 5 or len(author) < 3:
        return None, None

    # Sanitize for filename
    title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    author = re.sub(r'[^\w\s-]', '', author).strip().replace(' ', '_')

    # Limit filename length to avoid Windows path issues (max 200 chars for combined name)
    max_title_length = 150
    max_author_length = 50
    if len(title) > max_title_length:
        title = title[:max_title_length]
    if len(author) > max_author_length:
        author = author[:max_author_length]

    return title, author

def rename_and_organize_pdfs(source_directory: str, destination_directory: str):
    """
    Recursively process all PDFs in the source_directory, rename them, and move to destination_directory.
    Maintains folder structure. Creates a CSV log with all operations.
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # List to store all operations for CSV logging
    operations = []

    print(f"Starting PDF processing...")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")

    file_count = 0

    for dirpath, dirnames, filenames in os.walk(source_directory):
        # Ignore Mac OS folders and skip the destination folder itself
        dirnames[:] = [d for d in dirnames if not d.startswith('__MACOSX') and not d.startswith('._')
                       and os.path.join(dirpath, d) != destination_directory]

        for filename in sorted(filenames):
            if not filename.lower().endswith(".pdf"):
                continue

            file_count += 1
            print(f"Processing file {file_count}: {filename}")

            old_path = os.path.join(dirpath, filename)

            # Calculate relative path from source to maintain folder structure
            rel_dir = os.path.relpath(dirpath, source_directory)
            if rel_dir == '.':
                dest_dir = destination_directory
            else:
                dest_dir = os.path.join(destination_directory, rel_dir)

            # Create destination subdirectory if needed
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Extract title and author
            title, author = extract_title_author(old_path)

            rename_status = "Failed"
            move_status = "Failed"
            new_name = "N/A"
            reason = ""

            if title and author:
                new_name = f"{author}-{title}.pdf"
                new_path = os.path.join(dest_dir, new_name)

                # Check if file already exists
                counter = 1
                original_new_name = new_name
                while os.path.exists(new_path):
                    new_name = f"{author}-{title}_{counter}.pdf"
                    new_path = os.path.join(dest_dir, new_name)
                    counter += 1

                if counter > 1:
                    print(f"  -> Duplicate detected, using: {new_name}")

                try:
                    # Copy file to new location with new name
                    shutil.copy2(old_path, new_path)
                    rename_status = "Done"
                    move_status = "Done"
                    print(f"  -> Success: {new_name}")
                except Exception as e:
                    rename_status = "Done"
                    move_status = "Failed"
                    reason = str(e)
                    print(f"  -> Error moving file: {e}")
            else:
                new_name = filename  # Keep original name
                new_path = os.path.join(dest_dir, new_name)
                rename_status = "Failed"
                reason = "Could not extract title/author"

                # Still try to copy the file with original name
                try:
                    shutil.copy2(old_path, new_path)
                    move_status = "Done"
                    print(f"  -> Copied with original name (extraction failed)")
                except Exception as e:
                    move_status = "Failed"
                    reason = f"Extraction failed; Copy error: {str(e)}"
                    print(f"  -> Error: {e}")

            # Store operation details
            operations.append({
                'old_name': os.path.join(rel_dir if rel_dir != '.' else '', filename),
                'new_name': new_name,
                'rename_status': rename_status,
                'move_status': move_status,
                'reason': reason
            })

    # Sort operations alphabetically by old_name
    operations.sort(key=lambda x: x['old_name'].lower())

    # Write CSV log
    log_path = os.path.join(destination_directory, "rename_log.csv")
    with open(log_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['old_name', 'new_name', 'rename_status', 'move_status', 'reason'])
        writer.writeheader()
        writer.writerows(operations)

    # Print summary
    total = len(operations)
    renamed_success = sum(1 for op in operations if op['rename_status'] == 'Done')
    moved_success = sum(1 for op in operations if op['move_status'] == 'Done')

    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {total}")
    print(f"Successfully renamed: {renamed_success} ({renamed_success/total*100:.1f}%)")
    print(f"Successfully moved: {moved_success} ({moved_success/total*100:.1f}%)")
    print(f"Failed to rename: {total - renamed_success}")
    print(f"Failed to move: {total - moved_success}")
    print(f"\nLog saved to: {log_path}")
    print(f"Organized files location: {destination_directory}")

# Main execution
if __name__ == "__main__":
    source_directory = r"C:\Users\danieo\Downloads\sp-library -Renaming"
    destination_directory = r"C:\Users\danieo\Downloads\sp-library-organized"

    rename_and_organize_pdfs(source_directory, destination_directory)

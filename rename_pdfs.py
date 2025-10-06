import fitz  # PyMuPDF for PDF handling
import re
import os
import cv2  # For OCR image preprocessing
import numpy as np
from pytesseract import image_to_string, pytesseract
from PIL import Image
import io
from typing import Tuple, Optional

# Set Tesseract path if needed (update based on your installation)
pytesseract.tesseract_cmd = r'C:\Users\danieo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Windows installation

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

    return title, author

def rename_pdfs(root_directory: str):
    """
    Recursively process all PDFs in the root_directory and subdirectories, ignoring Mac OS folders like __MACOSX.
    Log to a single MD table in the root.
    """
    md_content = "| Old Name | New Name | Status |\n|----------|----------|--------|\n"
    
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Ignore Mac OS folders (e.g., __MACOSX)
        dirnames[:] = [d for d in dirnames if not d.startswith('__MACOSX') and not d.startswith('._')]

        for filename in sorted(filenames):
            if not filename.lower().endswith(".pdf"):
                continue

            old_path = os.path.join(dirpath, filename)
            title, author = extract_title_author(old_path)

            if title and author:
                new_name = f"{author}-{title}.pdf"
                new_path = os.path.join(dirpath, new_name)
                try:
                    os.rename(old_path, new_path)
                    status = "Successfully renamed"
                except Exception as e:
                    status = f"Failed to rename: {str(e)}"
            else:
                new_name = "N/A"
                status = "Failed: Could not extract title/author reliably"

            # Use relative path for old_name in log for clarity
            rel_path = os.path.relpath(old_path, root_directory)
            md_content += f"| {rel_path} | {new_name} | {status} |\n"

    # Write log to root directory
    log_path = os.path.join(root_directory, "rename_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Renaming complete. Log saved to {log_path}")

# Example usage: Update with your directory
if __name__ == "__main__":
    root_directory = r"C:\Users\danieo\Downloads\sp-library -Renaming"  # Replace with actual path
    rename_pdfs(root_directory)
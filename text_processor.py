"""
Text processing module for RAG pipeline.
Handles text extraction, cleaning, and chunking operations.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Generator, Tuple, Optional
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text extraction and chunking operations."""

    def __init__(self):
        """Initialize the text processor."""
        self.parent_chunk_size = config.PARENT_CHUNK_SIZE
        self.parent_overlap = config.PARENT_CHUNK_OVERLAP
        self.child_chunk_size = config.CHILD_CHUNK_SIZE
        self.child_overlap = config.CHILD_CHUNK_OVERLAP

    def extract_text_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extract text from a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text or None if extraction failed
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                return self._extract_from_pdf(file_path)
            elif suffix == ".epub":
                return self._extract_from_epub(file_path)
            elif suffix == ".txt":
                return self._extract_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}", exc_info=True)
            return None

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    def _extract_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB file."""
        book = epub.read_epub(file_path)
        text_parts = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text_parts.append(soup.get_text())

        return "\n".join(text_parts)

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        return file_path.read_text(encoding=config.ENCODING, errors="ignore")

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?\-\'\"]', ' ', text)
        # Remove multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_parent_chunks(self, text: str) -> Generator[str, None, None]:
        """
        Create parent chunks from text using a generator for memory efficiency.

        Args:
            text: Text to chunk

        Yields:
            Parent chunks
        """
        if not text:
            return

        text_length = len(text)
        start = 0

        while start < text_length:
            end = start + self.parent_chunk_size

            # If not at the end, try to break at a sentence boundary
            if end < text_length:
                # Look for sentence boundaries (. ! ?)
                last_period = text.rfind('.', start, end)
                last_exclaim = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)

                sentence_end = max(last_period, last_exclaim, last_question)

                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                yield chunk

            # Move start position with overlap
            start = end - self.parent_overlap

    def create_child_chunks(self, parent_chunk: str) -> Generator[str, None, None]:
        """
        Create child chunks from a parent chunk using a generator.

        Args:
            parent_chunk: Parent chunk to subdivide

        Yields:
            Child chunks
        """
        if not parent_chunk:
            return

        chunk_length = len(parent_chunk)
        start = 0

        while start < chunk_length:
            end = start + self.child_chunk_size

            # If not at the end, try to break at a word boundary
            if end < chunk_length:
                # Look for space
                last_space = parent_chunk.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = parent_chunk[start:end].strip()
            if chunk:
                yield chunk

            # Move start position with overlap
            start = end - self.child_overlap

    def process_file_chunks(
        self, file_path: Path
    ) -> Generator[Tuple[str, List[str]], None, None]:
        """
        Process a file and yield parent chunks with their children.

        Args:
            file_path: Path to file to process

        Yields:
            Tuples of (parent_chunk, list_of_child_chunks)
        """
        # Extract and clean text
        text = self.extract_text_from_file(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            logger.warning(f"No text after cleaning from {file_path}")
            return

        # Generate parent chunks and their children
        for parent_chunk in self.create_parent_chunks(cleaned_text):
            child_chunks = list(self.create_child_chunks(parent_chunk))
            if child_chunks:
                yield (parent_chunk, child_chunks)

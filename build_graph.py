"""
Build Knowledge Graph from Theology Library
Processes all document files in a directory and builds a unified NetworkX graph.

Supported formats:
- PDF, EPUB, MOBI (ebooks)
- TXT, MD, Markdown, RST (text)
- DOC, DOCX, ODT, RTF (word processors)
- HTML, HTM (web)
"""

import os
import re
import sys
import argparse
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import spacy
import spacy.cli
import numpy as np
import networkx as nx
from pypdf import PdfReader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Universal document loader
try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False
    print("Warning: textract not installed. Install with: pip install textract")
    print("Falling back to basic loaders for PDF, EPUB, and TXT only")

# Checkpoint interval: save large graph file every N files
CHECKPOINT_INTERVAL = 5

# Doctrinal keywords for entity extraction
DOCTRINAL_KEYWORDS = ['grace', 'election', 'atonement', 'sovereignty', 'predestination',
                      'covenant', 'sanctification', 'justification', 'regeneration', 'faith']

# Historical figures
HISTORICAL_FIGURES = ['Calvin', 'Luther', 'Edwards', 'Whitefield', 'Bonar', 'Spurgeon',
                      'Owen', 'Bunyan', 'Wesley', 'Knox']


def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model, downloading if necessary."""
    try:
        nlp = spacy.load(model_name)
        print(f"Loaded spaCy model: {model_name}")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        print(f"Successfully downloaded and loaded: {model_name}")

    # Disable heavy components we don't need for sentence segmentation
    # Keep only the sentencizer for splitting sentences
    nlp.disable_pipes("ner", "tagger", "parser", "lemmatizer")
    nlp.add_pipe('sentencizer')
    print("Disabled NER, tagger, and parser for faster processing")
    print("Added sentencizer for fast sentence segmentation")

    return nlp


def load_pdf_text(pdf_path):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def load_epub_text(epub_path):
    """Extract text from EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        print("Warning: ebooklib not installed. Install with: pip install ebooklib beautifulsoup4")
        return ""

    book = epub.read_epub(epub_path)
    text = ""

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + "\n\n"

    return text


def load_txt_text(txt_path):
    """Extract text from TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(txt_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read {txt_path}: {e}")
            return ""
    except Exception as e:
        print(f"Warning: Could not read {txt_path}: {e}")
        return ""


def load_document_universal(file_path):
    """
    Universal document loader that supports many formats.
    Supports: TXT, MD, PDF, EPUB, MOBI, DOC, DOCX, ODT, RTF, HTML, and more.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Try textract first if available (supports most formats)
    if TEXTRACT_AVAILABLE:
        try:
            text = textract.process(str(file_path)).decode('utf-8')
            return text
        except Exception as e:
            print(f"Warning: textract failed for {file_path.name}: {e}")
            print("Falling back to format-specific loader...")

    # Fallback loaders for specific formats
    if suffix == '.pdf':
        return load_pdf_text(file_path)
    elif suffix == '.epub':
        return load_epub_text(file_path)
    elif suffix in ['.txt', '.md', '.markdown', '.rst']:
        return load_txt_text(file_path)
    elif suffix in ['.docx', '.doc']:
        try:
            import docx2txt
            return docx2txt.process(str(file_path))
        except ImportError:
            print(f"Warning: docx2txt not installed. Install with: pip install docx2txt")
            return ""
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
            return ""
    elif suffix == '.odt':
        try:
            from odf import text as odf_text, teletype
            from odf.opendocument import load
            textdoc = load(str(file_path))
            allparas = textdoc.getElementsByType(odf_text.P)
            return "\n".join([teletype.extractText(para) for para in allparas])
        except ImportError:
            print(f"Warning: odfpy not installed. Install with: pip install odfpy")
            return ""
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
            return ""
    elif suffix in ['.html', '.htm']:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except ImportError:
            print(f"Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return ""
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
            return ""
    else:
        print(f"Warning: Unsupported format '{suffix}' for {file_path.name}")
        return ""


def clean_text(text):
    """Apply general cleaning rules to text."""
    # 1. Normalize whitespace
    text = text.replace('\t', ' ')
    text = re.sub(r' {2,}', ' ', text)

    # 2. Fix hyphenated words at line breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # 3. Fix spacing around punctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'(["\(\[])\s+', r'\1', text)
    text = re.sub(r'\s+(["\)\]])', r'\1', text)

    # 4. Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def polish_text(text):
    """Apply OCR error corrections."""
    ocr_corrections = {
        'save': ['sav e'],
        'the': ['t he'],
        'day': ['da y', 'da  y'],
        'grace': ['gr ace'],
        'lord': ['lo rd'],
        'god': ['g od'],
        'christ': ['ch rist'],
        'spirit': ['spir it'],
        'world': ['w orld'],
        'scarcely': ['sc arcely'],
        'congregation': ['con gregation'],
        'every': ['ev ery'],
        'sorrow': ['sorr ow'],
        'under': ['un der'],
        'experience': ['experie nce'],
        'mighty': ['mig hty'],
        'amaze': ['am aze'],
        'voice': ['voi ce'],
        'into': ['in to'],
        'thickened': ['thic kened'],
        'proceeds': ['proce eds'],
        'reformation': ['refo rmation'],
        'instruments': ['in struments'],
        'eternal': ['et ernal'],
        'neighbours': ['neigh bours']
    }

    for correct_word, error_variations in ocr_corrections.items():
        for error in error_variations:
            pattern = re.compile(r'\b' + re.escape(error) + r'\b', re.IGNORECASE)
            text = pattern.sub(correct_word, text)

    return text


def create_parent_chunks(text, max_chunk_size=30000):
    """Create parent chunks using RecursiveCharacterTextSplitter.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk (default 100000 to stay under spaCy's 1M limit)
    """
    chunks = []

    # Try splitting on ALL CAPS lines
    lines = text.split('\n')
    all_caps_indices = []

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if len(line_stripped) >= 10:
            alpha_chars = [c for c in line_stripped if c.isalpha()]
            if alpha_chars:
                uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if uppercase_ratio > 0.8:
                    all_caps_indices.append(idx)

    if all_caps_indices:
        current_chunk = []
        for idx, line in enumerate(lines):
            if idx in all_caps_indices and current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

    # If < 3 chunks, try triple newlines
    if len(chunks) < 3:
        chunks = re.split(r'\n\n\n+', text)
        chunks = [c.strip() for c in chunks if c.strip()]

    # If still < 3 chunks, use RecursiveCharacterTextSplitter
    if len(chunks) < 3:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

    # SAFETY CHECK: Enforce hard limit on all chunks regardless of splitting method
    # Split any chunk that exceeds max_chunk_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            # Use text splitter to break down oversized chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", " ", ""]
            )
            sub_chunks = text_splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    # Assign unique parent_id to each chunk
    parent_chunks = []
    for idx, chunk_text in enumerate(final_chunks, start=1):
        parent_chunks.append({
            "parent_id": f"parent_{idx}",
            "text": chunk_text
        })

    return parent_chunks


def split_sentences(parent_chunk, nlp):
    """Split parent chunk into sentences using spaCy."""
    doc = nlp(parent_chunk["text"])
    sentences = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if sentence_text:
            sentences.append({
                "parent_id": parent_chunk["parent_id"],
                "sentence": sentence_text
            })

    return sentences


def extract_entities_and_keywords(sentence_text, nlp, doctrinal_keywords):
    """Extract entities and keywords from sentence."""
    doc = nlp(sentence_text)
    entities = set()

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities.add(ent.text.lower())

    sentence_lower = sentence_text.lower()
    for keyword in doctrinal_keywords:
        if keyword in sentence_lower:
            entities.add(keyword)

    return entities


def count_tokens(text):
    """Simple token counter."""
    return len(text.split())


def create_child_chunks(sentences, nlp, similarity_threshold=0.3, max_tokens=1000, apply_heuristics=False):
    """Create child chunks using smart hybrid strategy."""
    # Extract entities for all sentences
    sentence_entities = []
    for sent_data in sentences:
        entities = extract_entities_and_keywords(sent_data['sentence'], nlp, DOCTRINAL_KEYWORDS)
        sentence_entities.append(entities)

    # Create TF-IDF vectors
    sentence_texts = [sent_data['sentence'] for sent_data in sentences]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentence_texts)

    # Initialize chunking
    child_chunks = []
    current_chunk_sentences = []
    current_chunk_entities = set()
    current_chunk_vectors = []
    current_chunk_token_count = 0
    chunk_id = 1

    for idx, sent_data in enumerate(sentences):
        sentence_text = sent_data['sentence']
        sentence_entity_set = sentence_entities[idx]
        sentence_vector = tfidf_matrix[idx]
        sentence_tokens = count_tokens(sentence_text)

        if not current_chunk_sentences:
            current_chunk_sentences.append(sentence_text)
            current_chunk_entities = sentence_entity_set.copy()
            current_chunk_vectors.append(sentence_vector)
            current_chunk_token_count = sentence_tokens
            continue

        # Calculate entity overlap
        entity_overlap = len(sentence_entity_set & current_chunk_entities)

        # Calculate semantic similarity
        if current_chunk_vectors:
            avg_chunk_vector = np.mean([vec.toarray() for vec in current_chunk_vectors], axis=0)
            semantic_sim = cosine_similarity(sentence_vector.toarray(), avg_chunk_vector)[0][0]
        else:
            semantic_sim = 0.0

        # Inclusion rule
        should_include = (entity_overlap > 0) or (semantic_sim > similarity_threshold)
        would_exceed_limit = (current_chunk_token_count + sentence_tokens) > max_tokens

        if should_include and not would_exceed_limit:
            current_chunk_sentences.append(sentence_text)
            current_chunk_entities.update(sentence_entity_set)
            current_chunk_vectors.append(sentence_vector)
            current_chunk_token_count += sentence_tokens
        else:
            # Finalize current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_data = {
                'child_id': f"child_{chunk_id}",
                'parent_id': sent_data['parent_id'],
                'text': chunk_text,
                'entities': list(current_chunk_entities),
                'token_count': current_chunk_token_count
            }

            if apply_heuristics:
                heuristics = apply_domain_heuristics(chunk_text, DOCTRINAL_KEYWORDS, HISTORICAL_FIGURES)
                chunk_data.update(heuristics)

            child_chunks.append(chunk_data)
            chunk_id += 1

            # Start new chunk
            current_chunk_sentences = [sentence_text]
            current_chunk_entities = sentence_entity_set.copy()
            current_chunk_vectors = [sentence_vector]
            current_chunk_token_count = sentence_tokens

    # Add last chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunk_data = {
            'child_id': f"child_{chunk_id}",
            'parent_id': sentences[0]['parent_id'],
            'text': chunk_text,
            'entities': list(current_chunk_entities),
            'token_count': current_chunk_token_count
        }

        if apply_heuristics:
            heuristics = apply_domain_heuristics(chunk_text, DOCTRINAL_KEYWORDS, HISTORICAL_FIGURES)
            chunk_data.update(heuristics)

        child_chunks.append(chunk_data)

    return child_chunks


def calculate_entity_density(sentences, nlp, doctrinal_keywords):
    """Calculate average entity density."""
    total_entities = 0
    total_sentences = len(sentences)

    for sent_data in sentences:
        entities = extract_entities_and_keywords(sent_data['sentence'], nlp, doctrinal_keywords)
        total_entities += len(entities)

    average_density = total_entities / total_sentences if total_sentences > 0 else 0
    return average_density, total_entities, total_sentences


def adaptive_threshold(entity_density):
    """Determine adaptive threshold based on entity density."""
    if entity_density > 1.0:
        return 0.35, "high entity density (theological/dense section)"
    else:
        return 0.25, "low entity density (narrative section)"


def detect_bible_verses(text):
    """Detect Bible verse references."""
    pattern = r'\b(?:\d\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s\d+:\d+(?:-\d+)?(?:,\s?\d+)*\b'
    verses = re.findall(pattern, text)
    return verses


def detect_doctrinal_keywords(text, doctrinal_keywords):
    """Detect doctrinal keywords."""
    text_lower = text.lower()
    found_keywords = []
    for keyword in doctrinal_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    return found_keywords


def detect_historical_figures(text, historical_figures):
    """Detect historical figures."""
    found_figures = []
    for figure in historical_figures:
        if re.search(r'\b' + re.escape(figure) + r'\b', text, re.IGNORECASE):
            found_figures.append(figure)
    return found_figures


def apply_domain_heuristics(chunk_text, doctrinal_keywords, historical_figures):
    """Apply domain-specific heuristics."""
    verses = detect_bible_verses(chunk_text)
    keywords = detect_doctrinal_keywords(chunk_text, doctrinal_keywords)
    figures = detect_historical_figures(chunk_text, historical_figures)

    return {
        'verses': verses,
        'keywords': keywords,
        'figures': figures
    }


def process_file_worker(file_info):
    """Process a single document file through the pipeline (worker function for parallel processing)."""
    file_path, file_index = file_info

    # Load spaCy model in worker process
    nlp = load_spacy_model("en_core_web_sm")

    print(f"\nProcessing file {file_index}: {file_path.name}")
    print("-" * 80)

    # Load text using universal loader
    raw_text = load_document_universal(file_path)

    if not raw_text or len(raw_text) < 100:
        print(f"  Warning: Extracted text too short or empty. Skipping.")
        return file_path.name, []

    print(f"  Extracted {len(raw_text)} characters")

    # Clean and polish text
    text = clean_text(raw_text)
    text = polish_text(text)
    print(f"  After cleaning: {len(text)} characters")

    # Create parent chunks
    parent_chunks = create_parent_chunks(text)
    print(f"  Created {len(parent_chunks)} parent chunks")

    # Process each parent chunk
    file_chunks = []

    print(f"  Processing {len(parent_chunks)} parent chunks...")

    for idx, parent_chunk in enumerate(parent_chunks, start=1):
        # Print progress every 10 chunks
        if idx % 10 == 0:
            print(f"    Progress: {idx}/{len(parent_chunks)} parent chunks processed")

        # Sentence segmentation
        sentences = split_sentences(parent_chunk, nlp)

        # Calculate adaptive threshold
        avg_density, _, _ = calculate_entity_density(sentences, nlp, DOCTRINAL_KEYWORDS)
        adaptive_thresh, _ = adaptive_threshold(avg_density)

        # Create child chunks with metadata
        child_chunks = create_child_chunks(
            sentences, nlp,
            similarity_threshold=adaptive_thresh,
            max_tokens=1000,
            apply_heuristics=True
        )

        # Add file source to metadata and create unique IDs
        for chunk in child_chunks:
            final_chunk = {
                'child_id': chunk['child_id'],
                'parent_id': chunk['parent_id'],
                'text': chunk['text'],
                'entities': chunk.get('entities', []),
                'verses': chunk.get('verses', []),
                'keywords': chunk.get('keywords', []),
                'figures': chunk.get('figures', []),
                'source_file': file_path.name,
                'unique_id': f"file{file_index}_{chunk['parent_id']}_{chunk['child_id']}"
            }
            file_chunks.append(final_chunk)

    print(f"  Total child chunks: {len(file_chunks)}")
    return file_path.name, file_chunks


def build_knowledge_graph(all_chunks):
    """Build NetworkX graph from all chunks."""
    print("\nBuilding knowledge graph...")
    print("=" * 80)

    G = nx.Graph()

    # Add nodes for each chunk
    for chunk in all_chunks:
        G.add_node(
            chunk['unique_id'],
            text=chunk['text'][:200],  # Store truncated text
            parent_id=chunk['parent_id'],
            source_file=chunk['source_file'],
            entities=', '.join(chunk['entities']),
            keywords=', '.join(chunk['keywords']),
            figures=', '.join(chunk['figures'])
        )

    print(f"Added {len(G.nodes())} nodes to graph")

    # Add edges based on shared entities, keywords, and figures
    chunks_by_parent = {}
    for chunk in all_chunks:
        parent = chunk['parent_id']
        if parent not in chunks_by_parent:
            chunks_by_parent[parent] = []
        chunks_by_parent[parent].append(chunk)

    edge_count = 0

    # Parent-child edges (hierarchical)
    for parent_id, chunks in chunks_by_parent.items():
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                G.add_edge(chunk1['unique_id'], chunk2['unique_id'],
                          relationship='sibling', weight=1.0)
                edge_count += 1

    # Entity-based edges (semantic)
    for i, chunk1 in enumerate(all_chunks):
        for chunk2 in all_chunks[i+1:]:
            # Calculate overlap
            entities1 = set(chunk1['entities'])
            entities2 = set(chunk2['entities'])
            keywords1 = set(chunk1['keywords'])
            keywords2 = set(chunk2['keywords'])
            figures1 = set(chunk1['figures'])
            figures2 = set(chunk2['figures'])

            entity_overlap = len(entities1 & entities2)
            keyword_overlap = len(keywords1 & keywords2)
            figure_overlap = len(figures1 & figures2)

            if entity_overlap > 0 or keyword_overlap > 0 or figure_overlap > 0:
                weight = entity_overlap + keyword_overlap + figure_overlap
                G.add_edge(chunk1['unique_id'], chunk2['unique_id'],
                          relationship='semantic', weight=weight)
                edge_count += 1

    print(f"Added {edge_count} edges to graph")

    return G


def sanitize_folder_name(folder_path):
    """Create a sanitized filename suffix from folder path."""
    # Get the last part of the path
    folder_name = Path(folder_path).name
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^\w\-]', '_', folder_name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def setup_logger(log_file):
    """Setup logger for the build process."""
    logger = logging.getLogger('GraphBuilder')
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build Knowledge Graph from Theology Library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_graph.py --source "C:/Users/danieo/Downloads/TheologyLibrary/Epub Vol. 1 - 25"
  python build_graph.py --source "C:/Users/danieo/Downloads/TheologyLibrary" --output my_graph.gml
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing PDF and/or EPUB files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output graph filename (optional, auto-generated if not provided)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    print("=" * 80)
    print("THEOLOGY LIBRARY KNOWLEDGE GRAPH BUILDER")
    print("=" * 80)

    # Check source directory
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {args.source}")
        return

    print(f"\nSource directory: {args.source}")

    # Generate output filenames
    folder_suffix = sanitize_folder_name(args.source)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_file = args.output
        log_file = f"log_{Path(args.output).stem}_{timestamp}.md"
        graph_stem = Path(args.output).stem
    else:
        output_file = f"graph_{folder_suffix}.gml"
        log_file = f"log_{folder_suffix}_{timestamp}.md"
        graph_stem = folder_suffix

    # Define checkpoint file paths
    checkpoint_graph_file = f"checkpoint_{graph_stem}.gml"
    checkpoint_files_list = f"processed_files_{graph_stem}.txt"

    print(f"Output graph file: {output_file}")
    print(f"Output log file: {log_file}")

    # Setup logger
    logger = setup_logger(log_file)
    logger.info("=" * 80)
    logger.info("THEOLOGY LIBRARY KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 80)
    logger.info(f"Source directory: {args.source}")
    logger.info(f"Output graph file: {output_file}")
    logger.info(f"Output log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")

    # Check for existing checkpoint files and load resume state
    processed_files = set()
    all_library_chunks = []
    G = nx.Graph()

    if os.path.exists(checkpoint_files_list):
        print(f"\n{'='*80}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*80}")
        logger.info(f"\n{'='*80}")
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info(f"{'='*80}")

        # Load processed files list
        with open(checkpoint_files_list, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f if line.strip())

        msg = f"Resuming from checkpoint. Found {len(processed_files)} files already processed."
        print(msg)
        logger.info(msg)

        # Load existing graph state
        if os.path.exists(checkpoint_graph_file):
            G = nx.read_gml(checkpoint_graph_file)
            msg = f"Loaded checkpoint graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
            print(msg)
            logger.info(msg)
    else:
        print("\nStarting fresh build (no checkpoint found)")
        logger.info("\nStarting fresh build (no checkpoint found)")

    # Supported document formats (excluding audio/video/binary)
    SUPPORTED_EXTENSIONS = (
        '.pdf', '.epub', '.txt', '.md', '.markdown', '.rst',
        '.doc', '.docx', '.odt', '.rtf', '.html', '.htm', '.mobi'
    )

    # Recursively find all supported document files in all subdirectories
    print("\nSearching for document files in all subdirectories...")
    logger.info("\nSearching for document files in all subdirectories...")
    all_files = []

    for root, dirs, files in os.walk(args.source):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                file_path = Path(root) / filename
                all_files.append(file_path)

    # Count by type
    from collections import defaultdict
    file_counts = defaultdict(int)
    for f in all_files:
        file_counts[f.suffix.lower()] += 1

    msg = "Found files by type:"
    print(msg)
    logger.info(msg)
    for ext, count in sorted(file_counts.items()):
        msg = f"  {ext}: {count} files"
        print(msg)
        logger.info(msg)

    msg = f"Total files to process: {len(all_files)}"
    print(msg)
    logger.info(msg)

    if len(all_files) == 0:
        msg = "No files to process. Exiting."
        print(msg)
        logger.warning(msg)
        return

    # Process all files in parallel
    print("\n" + "=" * 80)
    print("PROCESSING FILES IN PARALLEL (6 workers)")
    print("=" * 80)
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING FILES IN PARALLEL (6 workers)")
    logger.info("=" * 80)

    successful_files = len(processed_files)  # Start with already processed count
    failed_files = 0
    files_completed_this_run = 0  # Counter for files completed in current run

    # Filter out already processed files
    files_to_process = [(file_path, file_index) for file_index, file_path in enumerate(all_files, start=1)
                        if file_path.name not in processed_files]

    skipped_count = len(all_files) - len(files_to_process)
    if skipped_count > 0:
        msg = f"Skipping {skipped_count} already processed files"
        print(msg)
        logger.info(msg)

    if len(files_to_process) == 0:
        print("All files already processed!")
        logger.info("All files already processed!")
    else:
        print(f"Processing {len(files_to_process)} files with 6 parallel workers...")
        logger.info(f"Processing {len(files_to_process)} files with 6 parallel workers...")
        print(f"Graph checkpoint will be saved every {CHECKPOINT_INTERVAL} files")
        logger.info(f"Graph checkpoint will be saved every {CHECKPOINT_INTERVAL} files")

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=6) as executor:
            # Submit all jobs
            future_to_file = {executor.submit(process_file_worker, file_info): file_info
                            for file_info in files_to_process}

            # Process results as they complete
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                file_path, file_index = file_info

                try:
                    filename, chunks = future.result()

                    if chunks:
                        all_library_chunks.extend(chunks)
                        successful_files += 1
                        files_completed_this_run += 1

                        # Add chunks to graph
                        for chunk in chunks:
                            G.add_node(
                                chunk['unique_id'],
                                text=chunk['text'][:200],  # Store truncated text
                                parent_id=chunk['parent_id'],
                                source_file=chunk['source_file'],
                                entities=', '.join(chunk['entities']),
                                keywords=', '.join(chunk['keywords']),
                                figures=', '.join(chunk['figures'])
                            )

                        # TIER 1: Always save the small processed files list (fast & reliable resume mechanism)
                        processed_files.add(filename)
                        with open(checkpoint_files_list, 'w', encoding='utf-8') as f:
                            for fname in sorted(processed_files):
                                f.write(fname + '\n')

                        msg = f"  File {filename} processed successfully ({len(processed_files)} total)"
                        print(msg)
                        logger.info(msg)

                        # TIER 2: Periodically save the large graph file to minimize I/O conflicts
                        if files_completed_this_run % CHECKPOINT_INTERVAL == 0:
                            nx.write_gml(G, checkpoint_graph_file)
                            msg = f"  Graph checkpoint saved (every {CHECKPOINT_INTERVAL} files)"
                            print(msg)
                            logger.info(msg)
                    else:
                        failed_files += 1
                        msg = f"  File {filename} produced no chunks"
                        print(msg)
                        logger.warning(msg)

                except Exception as e:
                    msg = f"  Error processing file {file_path.name}: {e}"
                    print(msg)
                    logger.error(msg)
                    failed_files += 1
                    continue

        # FINAL SAVE: After all parallel jobs complete, save one final checkpoint
        if files_completed_this_run > 0:
            print("\nSaving final checkpoint after parallel processing...")
            logger.info("\nSaving final checkpoint after parallel processing...")
            nx.write_gml(G, checkpoint_graph_file)
            msg = f"Final checkpoint saved with {G.number_of_nodes()} nodes"
            print(msg)
            logger.info(msg)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)

    msg = f"\nTotal chunks from all files: {len(all_library_chunks)}"
    print(msg)
    logger.info(msg)

    msg = f"Successfully processed: {successful_files} files"
    print(msg)
    logger.info(msg)

    msg = f"Failed to process: {failed_files} files"
    print(msg)
    logger.info(msg)

    if G.number_of_nodes() == 0:
        msg = "No chunks created. Cannot build graph. Exiting."
        print(msg)
        logger.warning(msg)
        return

    # NOTE: Edge building is skipped here for performance with large graphs
    # Use add_edges_memory_optimized.py to add edges efficiently:
    #   python add_edges_memory_optimized.py --source-dir . --workers 7
    print("\nSkipping edge building (use add_edges_memory_optimized.py for large graphs)")
    logger.info("\nSkipping edge building - nodes only graph")
    logger.info(f"Graph built with {G.number_of_nodes()} nodes (edges to be added separately)")

    # Save final graph
    print(f"\nSaving graph to {output_file}...")
    logger.info(f"\nSaving graph to {output_file}...")
    nx.write_gml(G, output_file)
    msg = f"Graph saved successfully!"
    print(msg)
    logger.info(msg)

    # Clean up checkpoint files
    if os.path.exists(checkpoint_graph_file):
        os.remove(checkpoint_graph_file)
        print(f"Removed checkpoint file: {checkpoint_graph_file}")
        logger.info(f"Removed checkpoint file: {checkpoint_graph_file}")

    if os.path.exists(checkpoint_files_list):
        os.remove(checkpoint_files_list)
        print(f"Removed checkpoint file: {checkpoint_files_list}")
        logger.info(f"Removed checkpoint file: {checkpoint_files_list}")

    # Print summary
    print("\n" + "=" * 80)
    print("GRAPH SUMMARY")
    print("=" * 80)
    logger.info("\n" + "=" * 80)
    logger.info("GRAPH SUMMARY")
    logger.info("=" * 80)

    msg = f"Total nodes: {G.number_of_nodes()}"
    print(msg)
    logger.info(msg)

    msg = f"Total edges: {G.number_of_edges()}"
    print(msg)
    logger.info(msg)

    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    msg = f"Average degree: {avg_degree:.2f}"
    print(msg)
    logger.info(msg)

    msg = f"Number of connected components: {nx.number_connected_components(G)}"
    print(msg)
    logger.info(msg)

    # Source file breakdown
    source_files = Counter(chunk['source_file'] for chunk in all_library_chunks)
    print(f"\nChunks per file:")
    logger.info(f"\nChunks per file:")
    for file, count in source_files.most_common():
        msg = f"  {file}: {count} chunks"
        print(msg)
        logger.info(msg)

    print(f"\n{'='*80}")
    print(f"Processing complete! Log saved to: {log_file}")
    print(f"{'='*80}")
    logger.info(f"\nProcessing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    import multiprocessing
    multiprocessing.freeze_support()
    main()

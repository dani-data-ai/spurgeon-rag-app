"""
Test Script for Step 1: Create Parent Chunks
This script implements the structural preprocessing step from the MASTER_PLAN.
"""

import re
import spacy
import spacy.cli
import numpy as np
import chromadb
from pypdf import PdfReader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_spacy_model(model_name="en_core_web_sm"):
    """
    Load spaCy model, downloading it if necessary.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        spacy.Language: Loaded spaCy model
    """
    try:
        nlp = spacy.load(model_name)
        print(f"Loaded spaCy model: {model_name}")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        print(f"Successfully downloaded and loaded: {model_name}")
    return nlp


def load_pdf_text(pdf_path):
    """
    Load and extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Extracted text from all pages
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    """
    Clean extracted PDF text using general, scalable rules.

    This function applies only generic cleaning rules that work on any text:
    1. Normalize whitespace (tabs and multiple spaces -> single space)
    2. Fix hyphenated words at line breaks
    3. Fix spacing around punctuation
    4. Normalize newlines

    Args:
        text: Raw text extracted from PDF

    Returns:
        str: Cleaned text
    """
    # 1. NORMALIZE WHITESPACE
    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # 2. FIX HYPHENATED WORDS AT LINE BREAKS
    # Remove hyphen followed by newline (e.g., "hyphen-\nated" -> "hyphenated")
    text = re.sub(r'-\s*\n\s*', '', text)

    # 3. FIX SPACING AROUND PUNCTUATION
    # Remove spaces before punctuation marks
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)

    # Remove spaces after opening quotes/brackets/parentheses
    text = re.sub(r'(["\(\[])\s+', r'\1', text)

    # Remove spaces before closing quotes/brackets/parentheses
    text = re.sub(r'\s+(["\)\]])', r'\1', text)

    # 4. NORMALIZE NEWLINES
    # Replace 3+ newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def polish_text(text):
    """
    Apply final polish to fix known OCR errors and common broken words.

    This function contains domain-specific corrections for common OCR artifacts.
    The dictionary can be extended as new OCR errors are discovered.

    Args:
        text: Text that has already been cleaned with general rules

    Returns:
        str: Polished text with OCR errors corrected
    """
    # Dictionary of correct word -> list of common OCR errors
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

    # Apply corrections using word boundaries for accuracy
    for correct_word, error_variations in ocr_corrections.items():
        for error in error_variations:
            # Case-insensitive replacement with word boundaries
            # This preserves capitalization of the first letter
            pattern = re.compile(r'\b' + re.escape(error) + r'\b', re.IGNORECASE)
            text = pattern.sub(correct_word, text)

    return text


def create_parent_chunks(text):
    """
    Create parent chunks using a robust multi-step splitting strategy.

    Strategy:
    1. First Attempt: Split on ALL CAPS lines (section headers)
    2. Second Attempt: If < 3 chunks, split on triple newlines (\n\n\n)
    3. Final Fallback: If < 3 chunks, use RecursiveCharacterTextSplitter

    Args:
        text: Full document text

    Returns:
        list: List of dictionaries with parent_id and text
    """
    chunks = []
    method_used = ""

    # FIRST ATTEMPT: Split on ALL CAPS lines (potential headers)
    print("Attempting Method 1: Splitting on ALL CAPS lines...")
    # Find lines that are all caps (at least 10 characters, mostly uppercase)
    lines = text.split('\n')
    all_caps_indices = []

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        # Check if line is ALL CAPS (at least 10 chars, >80% uppercase letters)
        if len(line_stripped) >= 10:
            alpha_chars = [c for c in line_stripped if c.isalpha()]
            if alpha_chars:
                uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if uppercase_ratio > 0.8:
                    all_caps_indices.append(idx)

    # Split text at these indices
    if all_caps_indices:
        current_chunk = []
        for idx, line in enumerate(lines):
            if idx in all_caps_indices and current_chunk:
                # Save previous chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # Add last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        method_used = "ALL CAPS line splitting"
        print(f"  -> Created {len(chunks)} chunks using ALL CAPS lines")

    # SECOND ATTEMPT: Split on triple newlines if we have fewer than 3 chunks
    if len(chunks) < 3:
        print("Attempting Method 2: Splitting on triple newlines (\\n\\n\\n)...")
        chunks = re.split(r'\n\n\n+', text)
        chunks = [c.strip() for c in chunks if c.strip()]
        method_used = "Triple newline splitting"
        print(f"  -> Created {len(chunks)} chunks using triple newlines")

    # FINAL FALLBACK: Use RecursiveCharacterTextSplitter if still < 3 chunks
    if len(chunks) < 3:
        print("Attempting Method 3: Using RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        method_used = "RecursiveCharacterTextSplitter (chunk_size=4000, overlap=200)"
        print(f"  -> Created {len(chunks)} chunks using RecursiveCharacterTextSplitter")

    # Assign unique parent_id to each chunk
    parent_chunks = []
    for idx, chunk_text in enumerate(chunks, start=1):
        parent_chunks.append({
            "parent_id": f"parent_{idx}",
            "text": chunk_text
        })

    print(f"\nFinal method used: {method_used}")
    return parent_chunks


def split_sentences(parent_chunk, nlp):
    """
    Split a parent chunk into sentences using spaCy.

    Args:
        parent_chunk: Dictionary with parent_id and text
        nlp: Loaded spaCy model

    Returns:
        list: List of sentences with their parent_id
    """
    doc = nlp(parent_chunk["text"])
    sentences = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if sentence_text:  # Only include non-empty sentences
            sentences.append({
                "parent_id": parent_chunk["parent_id"],
                "sentence": sentence_text
            })

    return sentences


def extract_entities_and_keywords(sentence_text, nlp, doctrinal_keywords):
    """
    Extract PERSON and ORG entities, plus doctrinal keywords from a sentence.

    Args:
        sentence_text: Text of the sentence
        nlp: Loaded spaCy model
        doctrinal_keywords: List of doctrinal keywords to search for

    Returns:
        set: Set of entities and keywords found
    """
    doc = nlp(sentence_text)
    entities = set()

    # Extract PERSON and ORG entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities.add(ent.text.lower())

    # Check for doctrinal keywords
    sentence_lower = sentence_text.lower()
    for keyword in doctrinal_keywords:
        if keyword in sentence_lower:
            entities.add(keyword)

    return entities


def count_tokens(text):
    """Simple token counter based on whitespace splitting."""
    return len(text.split())


def create_child_chunks(sentences, nlp, similarity_threshold=0.3, max_tokens=1000, apply_heuristics=False):
    """
    Create child chunks using smart hybrid strategy with entity overlap and semantic similarity.

    Args:
        sentences: List of sentence dictionaries from split_sentences
        nlp: Loaded spaCy model
        similarity_threshold: Minimum cosine similarity to include sentence
        max_tokens: Maximum tokens per chunk
        apply_heuristics: Whether to apply domain-specific heuristics

    Returns:
        list: List of child chunk dictionaries
    """
    # Doctrinal keywords for Puritan theology
    doctrinal_keywords = ['grace', 'election', 'atonement', 'sovereignty', 'predestination',
                          'covenant', 'sanctification', 'justification', 'regeneration', 'faith']

    # Historical figures for Puritan/Reformation theology
    historical_figures = ['Calvin', 'Luther', 'Edwards', 'Whitefield', 'Bonar', 'Spurgeon',
                          'Owen', 'Bunyan', 'Wesley', 'Knox']

    # Extract entities for all sentences
    print("Extracting entities and keywords from sentences...")
    sentence_entities = []
    for sent_data in sentences:
        entities = extract_entities_and_keywords(sent_data['sentence'], nlp, doctrinal_keywords)
        sentence_entities.append(entities)

    # Create TF-IDF vectors for semantic similarity
    print("Computing TF-IDF vectors for semantic similarity...")
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

    print(f"\nProcessing {len(sentences)} sentences with hybrid chunking logic...")
    print("-" * 80)

    for idx, sent_data in enumerate(sentences):
        sentence_text = sent_data['sentence']
        sentence_entity_set = sentence_entities[idx]
        sentence_vector = tfidf_matrix[idx]
        sentence_tokens = count_tokens(sentence_text)

        # Check if this is the first sentence
        if not current_chunk_sentences:
            # Start new chunk with this sentence
            current_chunk_sentences.append(sentence_text)
            current_chunk_entities = sentence_entity_set.copy()
            current_chunk_vectors.append(sentence_vector)
            current_chunk_token_count = sentence_tokens
            continue

        # Calculate entity overlap
        entity_overlap = len(sentence_entity_set & current_chunk_entities)

        # Calculate semantic similarity (cosine similarity with average of current chunk)
        if current_chunk_vectors:
            avg_chunk_vector = np.mean([vec.toarray() for vec in current_chunk_vectors], axis=0)
            semantic_sim = cosine_similarity(sentence_vector.toarray(), avg_chunk_vector)[0][0]
        else:
            semantic_sim = 0.0

        # Inclusion rule: entity overlap > 0 OR semantic similarity > threshold
        should_include = (entity_overlap > 0) or (semantic_sim > similarity_threshold)

        # Check if adding this sentence would exceed token limit
        would_exceed_limit = (current_chunk_token_count + sentence_tokens) > max_tokens

        # Decision logic
        if should_include and not would_exceed_limit:
            # Add to current chunk
            current_chunk_sentences.append(sentence_text)
            current_chunk_entities.update(sentence_entity_set)
            current_chunk_vectors.append(sentence_vector)
            current_chunk_token_count += sentence_tokens
        else:
            # Finalize current chunk and start new one
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_data = {
                'child_id': f"child_{chunk_id}",
                'parent_id': sent_data['parent_id'],
                'text': chunk_text,
                'entities': list(current_chunk_entities),
                'token_count': current_chunk_token_count
            }

            # Apply domain-specific heuristics if requested
            if apply_heuristics:
                heuristics = apply_domain_heuristics(chunk_text, doctrinal_keywords, historical_figures)
                chunk_data.update(heuristics)

            child_chunks.append(chunk_data)
            chunk_id += 1

            # Start new chunk with current sentence
            current_chunk_sentences = [sentence_text]
            current_chunk_entities = sentence_entity_set.copy()
            current_chunk_vectors = [sentence_vector]
            current_chunk_token_count = sentence_tokens

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunk_data = {
            'child_id': f"child_{chunk_id}",
            'parent_id': sentences[0]['parent_id'],
            'text': chunk_text,
            'entities': list(current_chunk_entities),
            'token_count': current_chunk_token_count
        }

        # Apply domain-specific heuristics if requested
        if apply_heuristics:
            heuristics = apply_domain_heuristics(chunk_text, doctrinal_keywords, historical_figures)
            chunk_data.update(heuristics)

        child_chunks.append(chunk_data)

    return child_chunks


def calculate_entity_density(sentences, nlp, doctrinal_keywords):
    """
    Calculate the average entity density (entities per sentence) for a set of sentences.

    Args:
        sentences: List of sentence dictionaries
        nlp: Loaded spaCy model
        doctrinal_keywords: List of doctrinal keywords

    Returns:
        tuple: (average_density, total_entities, total_sentences)
    """
    total_entities = 0
    total_sentences = len(sentences)

    for sent_data in sentences:
        entities = extract_entities_and_keywords(sent_data['sentence'], nlp, doctrinal_keywords)
        total_entities += len(entities)

    average_density = total_entities / total_sentences if total_sentences > 0 else 0

    return average_density, total_entities, total_sentences


def adaptive_threshold(entity_density):
    """
    Determine the semantic similarity threshold based on entity density.

    Args:
        entity_density: Average number of entities per sentence

    Returns:
        tuple: (threshold, reason)
    """
    # If density is above average (using a baseline of 1.0 as threshold)
    # we consider it "high density" and use a higher threshold
    if entity_density > 1.0:
        return 0.35, "high entity density (theological/dense section)"
    else:
        return 0.25, "low entity density (narrative section)"


def detect_bible_verses(text):
    """
    Detect Bible verse references in text using regex.

    Args:
        text: Text to search for Bible verses

    Returns:
        list: List of Bible verse references found
    """
    # Pattern matches: Book Name Chapter:Verse (e.g., "John 3:16", "1 Corinthians 13:4")
    # Matches both single-word books (John) and multi-word books (1 Corinthians)
    pattern = r'\b(?:\d\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s\d+:\d+(?:-\d+)?(?:,\s?\d+)*\b'
    verses = re.findall(pattern, text)
    return verses


def detect_doctrinal_keywords(text, doctrinal_keywords):
    """
    Detect which doctrinal keywords are present in the text.

    Args:
        text: Text to search
        doctrinal_keywords: List of keywords to search for

    Returns:
        list: List of keywords found in the text
    """
    text_lower = text.lower()
    found_keywords = []
    for keyword in doctrinal_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    return found_keywords


def detect_historical_figures(text, historical_figures):
    """
    Detect which historical figures are mentioned in the text.

    Args:
        text: Text to search
        historical_figures: List of figure names to search for

    Returns:
        list: List of figures found in the text
    """
    found_figures = []
    for figure in historical_figures:
        # Case-insensitive search
        if re.search(r'\b' + re.escape(figure) + r'\b', text, re.IGNORECASE):
            found_figures.append(figure)
    return found_figures


def apply_domain_heuristics(chunk_text, doctrinal_keywords, historical_figures):
    """
    Apply domain-specific heuristics to extract metadata from chunk text.

    Args:
        chunk_text: The text of the child chunk
        doctrinal_keywords: List of doctrinal keywords
        historical_figures: List of historical figure names

    Returns:
        dict: Dictionary with verses, keywords, and figures
    """
    verses = detect_bible_verses(chunk_text)
    keywords = detect_doctrinal_keywords(chunk_text, doctrinal_keywords)
    figures = detect_historical_figures(chunk_text, historical_figures)

    return {
        'verses': verses,
        'keywords': keywords,
        'figures': figures
    }


def main():
    """Main execution function."""
    # Define the PDF path
    pdf_path = Path(r"C:\Users\danieo\Downloads\sp-library\True Revivals and the Men God Uses.pdf")

    # Check if file exists
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading PDF from: {pdf_path}")
    print("=" * 80)

    # Load PDF text
    raw_text = load_pdf_text(pdf_path)
    print(f"Extracted {len(raw_text)} characters from PDF (raw)")

    # Two-step cleaning process
    print("\nStep 1: Applying general cleaning rules...")
    text = clean_text(raw_text)
    print(f"  After general cleaning: {len(text)} characters")

    print("Step 2: Applying OCR error corrections (final polish)...")
    text = polish_text(text)
    print(f"  After polishing: {len(text)} characters")
    print("=" * 80)

    # Create parent chunks
    parent_chunks = create_parent_chunks(text)
    print(f"\nCreated {len(parent_chunks)} parent chunks")
    print("=" * 80)

    # Print first 100 characters of each parent chunk
    print("\nParent Chunks Preview:")
    print("-" * 80)
    for chunk in parent_chunks:
        print(f"\n{chunk['parent_id']}:")
        print(chunk['text'][:100])
        print("-" * 80)

    # DEBUG: Find and display the parent chunk containing the problematic text
    print("\n" + "=" * 80)
    print("DEBUG: Checking for text extraction issues")
    print("=" * 80)

    search_text = "And is the Lord's hand shortened"
    found_chunk = None
    for chunk in parent_chunks:
        if search_text in chunk['text']:
            found_chunk = chunk
            break

    if found_chunk:
        print(f"\nFound text in {found_chunk['parent_id']}")
        print("-" * 80)
        print("RAW PARENT CHUNK TEXT (exactly as extracted from PDF):")
        print("-" * 80)
        print(found_chunk['text'])
        print("-" * 80)
        print(f"Length: {len(found_chunk['text'])} characters")
    else:
        print(f"\nText '{search_text}' not found in any parent chunk")

    print("=" * 80)

    # STEP 2-6: Process ALL parent chunks through the full pipeline
    print("\n" + "=" * 80)
    print("PROCESSING ALL PARENT CHUNKS THROUGH FULL PIPELINE")
    print("=" * 80)

    # Load spaCy model
    nlp = load_spacy_model("en_core_web_sm")

    # Doctrinal keywords (same as in create_child_chunks)
    doctrinal_keywords = ['grace', 'election', 'atonement', 'sovereignty', 'predestination',
                          'covenant', 'sanctification', 'justification', 'regeneration', 'faith']

    # Collect all child chunks from all parents
    all_document_chunks = []

    print(f"\nProcessing {len(parent_chunks)} parent chunks...")
    print("=" * 80)

    for idx, parent_chunk in enumerate(parent_chunks, start=1):
        print(f"\n[{idx}/{len(parent_chunks)}] Processing {parent_chunk['parent_id']}...")

        # STEP 2: Sentence Segmentation
        sentences = split_sentences(parent_chunk, nlp)
        print(f"  - Segmented into {len(sentences)} sentences")

        # STEP 4: Calculate entity density and adaptive threshold
        avg_density, total_entities, total_sentences = calculate_entity_density(
            sentences, nlp, doctrinal_keywords
        )
        adaptive_thresh, reason = adaptive_threshold(avg_density)
        print(f"  - Entity density: {avg_density:.2f} -> threshold: {adaptive_thresh} ({reason})")

        # STEP 3 & 5: Create child chunks with adaptive threshold and metadata
        child_chunks = create_child_chunks(
            sentences, nlp,
            similarity_threshold=adaptive_thresh,
            max_tokens=1000,
            apply_heuristics=True
        )
        print(f"  - Created {len(child_chunks)} child chunks with metadata")

        # STEP 6: Structure for database
        for chunk in child_chunks:
            final_chunk = {
                'child_id': chunk['child_id'],
                'parent_id': chunk['parent_id'],
                'text': chunk['text'],
                'entities': chunk.get('entities', []),
                'verses': chunk.get('verses', []),
                'keywords': chunk.get('keywords', []),
                'figures': chunk.get('figures', [])
            }
            all_document_chunks.append(final_chunk)

    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING COMPLETE")
    print("=" * 80)

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL DOCUMENT SUMMARY")
    print("=" * 80)

    import json

    print(f"\nTotal parent chunks processed: {len(parent_chunks)}")
    print(f"Total child chunks created:    {len(all_document_chunks)}")
    print("=" * 80)

    # Summary statistics
    print("\nDocument-Level Statistics:")
    print("-" * 80)
    total_chunks = len(all_document_chunks)
    chunks_with_entities = sum(1 for c in all_document_chunks if c['entities'])
    chunks_with_verses = sum(1 for c in all_document_chunks if c['verses'])
    chunks_with_keywords = sum(1 for c in all_document_chunks if c['keywords'])
    chunks_with_figures = sum(1 for c in all_document_chunks if c['figures'])

    print(f"Total chunks:              {total_chunks}")
    print(f"Chunks with entities:      {chunks_with_entities} ({chunks_with_entities/total_chunks*100:.1f}%)")
    print(f"Chunks with Bible verses:  {chunks_with_verses} ({chunks_with_verses/total_chunks*100:.1f}%)")
    print(f"Chunks with keywords:      {chunks_with_keywords} ({chunks_with_keywords/total_chunks*100:.1f}%)")
    print(f"Chunks with figures:       {chunks_with_figures} ({chunks_with_figures/total_chunks*100:.1f}%)")

    # Display the last chunk in full detail
    print("\n" + "=" * 80)
    print("Full structure of LAST chunk:")
    print("=" * 80)
    print(json.dumps(all_document_chunks[-1], indent=2, ensure_ascii=False))
    print("=" * 80)

    # Per-parent chunk breakdown
    print("\nBreakdown by parent chunk:")
    print("-" * 80)
    from collections import Counter
    parent_chunk_counts = Counter(c['parent_id'] for c in all_document_chunks)
    for parent_id in sorted(parent_chunk_counts.keys()):
        print(f"{parent_id}: {parent_chunk_counts[parent_id]} child chunks")

    # Verification: Check if text cleaning worked
    print("\n" + "=" * 80)
    print("TEXT CLEANING VERIFICATION")
    print("=" * 80)

    # Find the chunk with "Lord's hand shortened"
    verification_text = "Lord's hand shortened"
    verification_chunk = None
    for chunk in all_document_chunks:
        if verification_text in chunk['text']:
            verification_chunk = chunk
            break

    if verification_chunk:
        print(f"\nFound chunk with '{verification_text}':")
        print("-" * 80)
        print(json.dumps(verification_chunk, indent=2, ensure_ascii=False))
        print("-" * 80)
        # Check specifically for the word "save"
        if 'save' in verification_chunk['text']:
            print("\n[SUCCESS] Text cleaning worked! 'save' is now a complete word.")
        elif 'sav e' in verification_chunk['text']:
            print("\n[WARNING] Text still contains 'sav e' (not properly cleaned)")
        else:
            print("\n[NOTE] Neither 'save' nor 'sav e' found in expected location")
    else:
        print(f"\nChunk containing '{verification_text}' not found")

    # CHROMADB INGESTION
    print("\n" + "=" * 80)
    print("CHROMADB INGESTION")
    print("=" * 80)

    # Setup ChromaDB
    print("\nSetting up ChromaDB...")
    db_path = "./puritan_vector_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB persistent client created at: {db_path}")

    # Setup HuggingFace embeddings
    print("\nInitializing HuggingFace embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")

    # Create or get collection
    print("\nCreating/getting ChromaDB collection...")
    collection_name = "puritan_writings"

    # Delete existing collection if it exists (for clean testing)
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Puritan theological writings with hierarchical chunking"}
    )
    print(f"Collection created: {collection_name}")

    # Prepare data for ingestion
    print("\nPreparing data for ChromaDB ingestion...")
    ids = []
    documents = []
    metadatas = []

    for chunk in all_document_chunks:
        # Create unique ID by combining parent_id and child_id
        unique_id = f"{chunk['parent_id']}_{chunk['child_id']}"
        ids.append(unique_id)
        documents.append(chunk['text'])

        # Create metadata dictionary (ChromaDB doesn't support lists in metadata)
        # So we'll convert lists to comma-separated strings
        metadata = {
            'parent_id': chunk['parent_id'],
            'entities': ', '.join(chunk['entities']) if chunk['entities'] else '',
            'verses': ', '.join(chunk['verses']) if chunk['verses'] else '',
            'keywords': ', '.join(chunk['keywords']) if chunk['keywords'] else '',
            'figures': ', '.join(chunk['figures']) if chunk['figures'] else ''
        }
        metadatas.append(metadata)

    print(f"Prepared {len(ids)} chunks for ingestion")

    # Generate embeddings and ingest
    print("\nGenerating embeddings and ingesting into ChromaDB...")
    print("(This may take a moment...)")

    # Generate embeddings for all documents
    embeddings = embedding_model.embed_documents(documents)

    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\n[SUCCESS] Successfully ingested {len(ids)} chunks into ChromaDB!")
    print("=" * 80)

    # Verify with test query
    print("\nTEST QUERY: Searching for 'grace'...")
    print("-" * 80)

    query_text = "grace"
    query_embedding = embedding_model.embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    print(f"\nTop 2 results for query '{query_text}':\n")
    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), start=1):
        print(f"Result {i}:")
        print(f"  ID: {doc_id}")
        print(f"  Parent: {metadata['parent_id']}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Text (first 100 chars): {doc[:100]}...")
        if metadata.get('keywords'):
            print(f"  Keywords: {metadata['keywords']}")
        print()


if __name__ == "__main__":
    main()

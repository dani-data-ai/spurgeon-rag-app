Puritan RAG App - Master Plan v1.0
Part 1: The High-Level Flowchart (The "What")
(Flowchart image or text-based diagram we agreed on here. This is the north star )
All steps in visual flowchart
[Document Text] 
       │
       ▼
[Step 1: create_parent_chunks]
   - Split by headings/chapters & double newlines
   - Assign parent_id
       │
       ▼
[Step 2: split_sentences]
   - Use spaCy to segment sentences
   - Attach sentences to parent_id
       │
       ▼
[Step 3: create_child_chunks (Smart Hybrid)]
   ┌───────────────────────────────────────┐
   │ Iterate sentences:                     │
   │ - Compute entity overlap (PERSON, ORG, doctrinal terms) │
   │ - Compute semantic similarity (TF-IDF/GloVe)             │
   │ - Inclusion rules: overlap>0 OR similarity>adaptive threshold │
   │ - Max tokens per chunk = 1000                             │
   │ - Performance: batch-process or prioritize dense sections to reduce compute (~20-30% slower for large docs) │
   └───────────────────────────────────────┘
       │
       ▼
[Step 4: adaptive_thresholds]
   - Pre-scan entity density
   - Adjust similarity threshold dynamically per parent chunk
       │
       ▼
[Step 5: apply_domain_heuristics]
   - Detect Bible verses (regex)
   - Tag doctrinal keywords & historical figures
   - Lexicon is extendable for archaic variants
   - Store in child chunk metadata
       │
       ▼
[Step 6: link_parent_child]
   - Maintain parent-child relationships in a lookup dictionary
   - Each child chunk keeps child_id, text, entities, doctrinal_tags, parent_id reference
       │
       ▼
[Optional Step 7: multi_pass_refinement]
   - Cluster child chunks (KMeans/entity vectors)
   - Merge/split to improve topic coherence
       │
       ▼
[Optional Step 8: visualize_chunks]
   - NetworkX or Plotly: nodes=child chunks, edges=semantic similarity
   - Color nodes by parent or doctrinal topic
   - For large graphs (>100k chunks), sample 10% for visualization
       │
       ▼
[Step 9: prepare_for_chroma]
   - Output structured JSON per child chunk:
     {
       parent_id, child_id, text,
       entities, doctrinal_tags,
       token_count, semantic_vector
     }
   - Ready for Chroma ingestion

Part 2: Detailed Implementation Steps (The "How")
(This is the detailed, step-by-step guide for Claude. Each step is a self-contained unit of work.)

Libraries to Use:

spaCy (sentence segmentation, entity recognition)

scikit-learn (TfidfVectorizer for semantic overlap)

numpy (vector calculations)

NetworkX (optional visualization)

Plotly (optional interactive lightweight visualization for large graphs)

re (regex for domain heuristics)

Step 1 — Structural Preprocessing (Parent Chunks)

Action Name: create_parent_chunks
Instructions:

Input: full document text.

Split the document using double newlines (\n\n) and headings/chapters via regex (Chapter \w+:, Section \w+:).

Each piece becomes a parent chunk with a unique parent_id.

Output: list of dictionaries: { "parent_id": <id>, "text": <parent_text> }.

Step 2 — Sentence Segmentation

Action Name: split_sentences
Instructions:

Use spaCy (en_core_web_sm) to segment parent chunk text into sentences.

Store sentences in order, attached to their parent_id.

Step 3 — Smart Hybrid Child Chunking

Action Name: create_child_chunks
Instructions:

Initialize empty current_chunk.

Iterate over sentences of a parent chunk:

Extract entities (PERSON, ORG, doctrinal terms like elect, atonement, grace).

Compute semantic similarity using TfidfVectorizer.

Inclusion Rule: add sentence to current_chunk if entity overlap > 0 OR similarity > adaptive threshold.

If neither condition is met, finalize current_chunk as a child chunk and start a new one.

Max tokens per chunk = 1000.

Output child chunks as: { "child_id": <id>, "parent_id": <parent_id>, "text": <child_text>, "entities": [list] }.
Performance Note: For large documents, batch-process sentences or prioritize dense doctrinal sections to reduce compute (~20-30% slower for 2.5GB).

Step 4 — Adaptive Thresholds

Action Name: adaptive_thresholds
Instructions:

Pre-scan parent chunk to compute average entity density per sentence.

Dynamically adjust semantic similarity threshold:

High-density sections → 0.35

Low-density sections → 0.25

Apply threshold when deciding sentence inclusion in child chunk.

Step 5 — Domain-Specific Heuristics

Action Name: apply_domain_heuristics
Instructions:

Detect Bible verses via regex ([A-Z][a-z]+ \d+:\d+) as split points.

Tag doctrinal keywords: "elect", "atonement", "grace", "predestination".

Detect historical figures: "Calvin", "Edwards", "Whitefield".

Lexicon should be extendable to include archaic variants like "elecct".

Add metadata in each child chunk: {"verses": [], "keywords": [], "figures": []}.

Step 6 — Hierarchical Structure

Action Name: link_parent_child
Instructions:

Maintain parent-child relationships in a lookup dictionary, not embedded in every chunk.

Each child chunk stores child_id, text, entities, doctrinal_tags, and parent_id reference for retrieval.

Step 7 — Optional Multi-Pass Refinement

Action Name: multi_pass_refinement
Instructions:

Pass 1: initial child chunk creation.

Pass 2: cluster child chunks using entity vectors or TF-IDF (KMeans) to detect topic drift.

Merge or split chunks if they better fit another cluster.

Step 8 — Optional Visualization

Action Name: visualize_chunks
Instructions:

Use NetworkX or Plotly.

Nodes = child chunks; edges = semantic similarity.

Optional: color by parent ID or doctrinal topic.

For large graphs (>100k chunks), sample 10% for visualization to maintain performance.

Step 9 — Output for Chroma

Action Name: prepare_for_chroma
Instructions:

1.Output each child chunk with:

{
  "parent_id": "<parent_id>",
  "child_id": "<child_id>",
  "text": "<chunk_text>",
  "entities": [...],
  "doctrinal_tags": [...],
  "token_count": <count>,
  "semantic_vector": [embedding]
}


2.Ready for Chroma ingestion with hierarchical metadata intact.



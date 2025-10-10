# Spurgeon RAG App - Project Summary

## Project Type
**Retrieval-Augmented Generation (RAG) Web Application** for querying Puritan literature and Charles Spurgeon's works using AI

## Dataset Evolution
**Initial Scope**: Charles Spurgeon's sermons only
**Expanded Scope**: Puritan authors + Spurgeon's complete works

### Current Dataset
- **Puritan Literature**: 125 EPUB volumes (Epub_Vol_1-25, Epub_Vol_26-79, Epub_Vol_80-125)
- **Spurgeon Works**: Sermons and writings collection
- **Total Nodes**: 2.6M+ text chunks across 4 knowledge graph checkpoints
- **Graph Files**:
  - `checkpoint_Epub_Vol_1-25.gml` (4.1GB, 34.5M edges)
  - `checkpoint_Epub_Vol_26-79.gml` (5.9GB, ~48M edges)
  - `checkpoint_Epub_Vol_80-125.gml` (6.5GB, 54.3M edges)
  - `checkpoint_spurgeon.gml` (2.0GB, 16M edges)
- **Total Graph Corpus**: 18.5GB with 152M+ edges representing entity/keyword/figure co-occurrence relationships

## Tech Stack

### Core Application
- **Frontend**: Streamlit (Python web framework)
- **LLM**: Llama 3.2 8B via LM Studio (local inference) + OpenAI/OpenRouter API support
- **Vector Database**: ChromaDB with SQLite backend
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Reranking**: Cross-Encoder for result refinement
- **Framework**: LangChain for RAG orchestration
- **Document Processing**: PyPDF, PyMuPDF, ebooklib (EPUB)

### Graph Processing Pipeline
- **NLP**: spaCy (`en_core_web_sm`) for entity extraction and sentence segmentation
- **Graph Library**: NetworkX for knowledge graph construction
- **Text Processing**: BeautifulSoup4, lxml for parsing
- **Parallelization**: Python `multiprocessing` (7 workers)
- **Persistence**: GML format for graphs, SQLite for intermediate storage

## Evolution & Technical Challenges

### Phase 1: Initial RAG Implementation
- **Challenge**: Basic vector search with ChromaDB
- **Solution**: Implemented semantic chunking with parent-child relationships
- **Tech Used**: LangChain + ChromaDB + all-MiniLM-L6-v2

### Phase 2: Knowledge Graph Integration
- **Challenge**: Enhance retrieval with entity/keyword relationships
- **Problem**: Initial graph creation (~2.5GB documents) was missing edges
- **Tech Used**: NetworkX for undirected graph with entity/keyword/figure nodes

### Phase 3: Edge Generation Crisis
- **Challenge**: Graph files created without edges (nodes only)
- **Problem**: RAM exhaustion (>32GB) during edge pair generation
- **First Solution**: Single-threaded edge builder with SQLite temp storage
- **Result**: Worked but took 20+ hours per file

### Phase 4: Parallel Edge Optimization ✅
- **Challenge**: 20-hour processing time unacceptable
- **Solution**: 7-worker parallel edge generation with memory management
- **Tech Changes**:
  - `concurrent.futures.ProcessPoolExecutor` for CPU parallelism
  - Byte-based RAM monitoring with `psutil` (hard 19-28GB limit)
  - SQLite WAL mode with dedicated writer process (no lock conflicts)
  - Mid-item checkpointing at (i,j) pair level for resume capability
- **Result**: 4-6× speedup (~3-6 hours per file), <28GB RAM usage

### Phase 5: Checkpoint System
- **Challenge**: Handle interruptions during long-running processes
- **Solution**:
  - File-level state tracking (`.edge_builder_state.json`)
  - Per-file progress checkpoints (`.progress.json`)
  - Telemetry logging (`.telemetry.csv`)
  - Temp DB preservation on failure
- **Tech Used**: JSON state files + SQLite temp databases

### Phase 6: Mobile Optimization
- **Challenge**: Responsive UI for mobile access
- **Solution**: Mobile-first CSS with Streamlit custom components

## Current Architecture

### RAG Query Pipeline
```
User Query
    ↓
[Streamlit UI]
    ↓
[Embedding Model] → Vector Search → [ChromaDB Vector DB]
    ↓                                      ↓
[Cross-Encoder Reranking] ← Top K Results
    ↓
[LLM (LM Studio/OpenAI)] ← Reranked Context
    ↓
Answer + Source Citations
```

### Knowledge Graph Structure
```
Document Processing (Puritans + Spurgeon)
    ↓
[spaCy NLP] → Extract entities, keywords, figures
    ↓
[NetworkX Graph Builder]
    ├─ Nodes: Text chunks with metadata (2.6M+)
    ├─ Hierarchical Edges: Parent-child relationships
    └─ Semantic Edges: Entity/keyword/figure co-occurrence
    ↓
[GML Graph Files] (18.5GB total, 152M+ edges)
    ├─ Puritan Vol 1-25 (4.1GB, 34.5M edges)
    ├─ Puritan Vol 26-79 (5.9GB, ~48M edges)
    ├─ Puritan Vol 80-125 (6.5GB, 54.3M edges)
    └─ Spurgeon (2.0GB, 16M edges)
```

## Key Features
- Interactive chat interface with source citations
- Real-time cost tracking for API calls
- Connection monitoring for LM Studio
- Configurable RAG parameters (temperature, tokens, chunks)
- Mobile-responsive design
- Usage statistics dashboard

## Performance Stats
- **Vector DB Size**: 56.8 MB (ChromaDB SQLite)
- **Knowledge Graphs**: 18.5GB total (4 checkpoint files with 152M+ edges)
- **Graph Processing**: 7 CPU cores, 4-6× speedup vs single-threaded
- **Memory Usage**: Configurable 19-28GB hard limit with byte-level monitoring
- **Edge Generation Examples**:
  - Vol 80-125: 82M edge records in 6.5 hours
  - Spurgeon: 23.5M edge records in ~9.7 minutes
- **Apply Phase**: 50-90 minutes to add 16M-54M edges per graph file

## Repository Structure
```
spurgeon-rag-app/
├── app.py                          # Main Streamlit web app
├── setup_rag.py                    # Vector DB setup
├── build_graph.py                  # Parallel graph builder
├── add_edges_memory_optimized.py   # Optimized edge generation
├── requirements.txt                # Dependencies
├── vector_db/                      # ChromaDB storage
├── *.md                            # Documentation
└── checkpoint_*.gml                # Graph checkpoints
```

## Development Timeline Highlights
1. **Initial RAG**: Spurgeon sermons only, ChromaDB + LM Studio integration
2. **Dataset Expansion**: Added 125 EPUB volumes of Puritan literature
3. **Graph Pipeline**: spaCy + NetworkX for entity extraction across expanded corpus
4. **Edge Crisis**: Discovered missing edges in 2.6M node graphs after initial build
5. **Memory Optimization**: Implemented SQLite-backed edge generation to handle RAM constraints
6. **Parallel Processing**: 7-worker architecture with checkpointing for large-scale processing
7. **Production Ready**: Mobile UI + API provider support, 18.5GB graph corpus with 152M+ edges complete

## Known Limitations
- NetworkX loads entire graph into memory (unavoidable for apply phase)
- Edge generation is O(n²) for dense entity clusters
- Windows multiprocessing requires `spawn` method
- GML file format expansion: 743MB on disk → 20GB in memory

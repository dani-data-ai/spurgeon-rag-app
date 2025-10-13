# Spurgeon RAG App - Development Roadmap

## Project Evolution Timeline

### Phase 0: Initial Concept (October 6, 2025)
**Goal:** Create a RAG application for querying Spurgeon's sermons

**Scope:**
- Single source: Charles Spurgeon sermons only
- Local RAG implementation with LM Studio

**Tech Stack Established:**
- ChromaDB for vector storage
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- LM Studio + Llama 3.2 8B for local LLM inference
- Streamlit for web UI
- LangChain for RAG orchestration

**Deliverables:**
- Basic RAG query interface
- PDF processing pipeline
- Vector database setup script

---

### Phase 1: Dataset Expansion (October 7, 2025)
**Goal:** Expand beyond Spurgeon to include Puritan literature

**Changes:**
- **Expanded Dataset:**
  - Spurgeon works (sermons and writings)
  - 125 EPUB volumes of Puritan literature
    - Vol 1-25: Owen, Bunyan complete works
    - Vol 26-79: Puritan sermons
    - Vol 80-125: Additional Puritan texts
  - Reference books: Biographies, theological analysis, historical context

**Issues Faced:**
- Need for richer semantic connections beyond vector search
- Context required for multi-author theological relationships

**Solution Direction:**
- Decision to implement knowledge graph structure
- Entity extraction for authors, themes, theological concepts

---

### Phase 2: Knowledge Graph Implementation (October 8, 2025)
**Goal:** Build knowledge graphs with entity relationships

**Tech Stack Addition:**
- spaCy (en_core_web_sm) for NLP and entity extraction
- NetworkX for graph construction
- Parent-child chunking with semantic relationships

**Architecture:**
```
Documents → spaCy NLP → Extract entities/keywords/figures
         → NetworkX Graph Builder
         → Nodes: Text chunks (2.6M+)
         → Edges: Entity/keyword co-occurrence relationships
```

**Issues Faced - The Edge Crisis:**
- **Critical Bug:** Graphs created with 5.1M nodes but **0 edges**
- Silent failure - no error raised
- Only discovered after hours of processing
- Root cause: Edge generation logic failing without errors

**Impact:** Complete loss of semantic relationship data

---

### Phase 3: Edge Generation Fix & Memory Crisis (October 8-9, 2025)
**Goal:** Fix edge generation and handle memory constraints

**Problem 1: Missing Edges**
- Refactored edge generation logic
- Added validation checks
- Implemented entity/keyword/figure co-occurrence edge creation

**Problem 2: RAM Exhaustion**
- Graph building consumed >32GB RAM
- O(n²) complexity for dense entity clusters
- System crashes during edge pair generation

**Solution Implemented:**
- SQLite-backed edge generation (disk-based temp storage)
- Batch processing to control memory
- Single-threaded reliable implementation

**Result:**
- ✅ Edges generated correctly
- ❌ 20+ hours per large file (unacceptable)

---

### Phase 4: Parallel Processing Optimization (October 9, 2025)
**Goal:** Reduce 20-hour processing time

**Solution:**
- 7-worker parallel edge generation with `ProcessPoolExecutor`
- Dedicated SQLite writer process (WAL mode, no lock conflicts)
- Byte-based RAM monitoring with psutil (19-28GB hard limit)
- Mid-item checkpointing at (i,j) pair level for resume capability

**Technical Achievements:**
- Stable hash partitioning (no duplicate pairs)
- Queue-based communication with backpressure
- Atomic progress file writes
- Telemetry logging (CSV metrics per batch)

**Performance Results:**
- **4-6× speedup** (~3-6 hours per file vs 20+ hours)
- **Memory controlled** at ≤20GB RAM
- **Resumable** from exact checkpoint
- **Stable** across interruptions

**Graph Statistics:**
- checkpoint_Epub_Vol_1-25.gml: 4.1GB, 34.5M edges
- checkpoint_Epub_Vol_26-79.gml: 5.9GB, ~48M edges
- checkpoint_Epub_Vol_80-125.gml: 6.5GB, 54.3M edges
- checkpoint_spurgeon.gml: 2.0GB, 16M edges
- **Total:** 18.5GB with 152M+ edges

---

### Phase 5: Faiss Vector Store Experiment (October 9, 2025)
**Goal:** Explore Faiss as ChromaDB alternative

**Motivation:**
- Evaluate performance at scale
- GPU acceleration potential
- Better control over indexing

**Implementation:**
- Phase 1: Basic Faiss Flat index (exact search)
- IndexFlatL2 for maximum stability
- Chroma-compatible API
- Resource monitoring (RAM/VRAM limits)
- Batch processing
- Export script for Chroma migration

**Testing Results:**
- ✅ 100% stable (no crashes, no segfaults)
- ✅ Fast queries (<1ms on small datasets)
- ✅ Complete test coverage (31 tests, 100% pass)
- ✅ Production-ready for datasets up to 500k vectors

**Decision:**
- **Faiss implementation abandoned and deleted from project**
- Reason: Project direction shifted to graph-based architecture
- Vector store approach superseded by multi-graph knowledge base design

---

### Phase 6: Neo4j Migration Attempt (October 11, 2025)
**Goal:** Migrate from NetworkX GML files to Neo4j graph database

**Motivation:**
- NetworkX loads entire graph into memory (unavoidable)
- GML files: 743MB on disk → 20GB in memory
- Need for scalable graph database

**Attempt:**
- Converted GML files to Neo4j-compatible CSV (nodes and edges)
- Prepared Neo4j admin import scripts
- Post-import optimization scripts

**Issues Faced:**
- **File size too large for import**
- 18.5GB total graph corpus with 152M+ edges
- Neo4j admin import limitations encountered
- Resource constraints

**Decision:**
- **Neo4j migration halted**
- Prepared infrastructure exists but not implemented
- Files: `prepare_neo4j_import.py`, CSV conversion scripts

---

### Phase 7: Current State - Multi-Graph Architecture (October 12, 2025)
**Goal:** Define cross-graph relationship strategy

**Current Architecture:**
- **4 Separate Knowledge Graphs:**
  1. Epub_Vol_1-25: Owen/Bunyan complete works (source material)
  2. Epub_Vol_26-79: Puritan sermons (source material)
  3. Spurgeon: Spurgeon's works and sermons
  4. Reference Books: Biographical and theological analysis (human-curated connections)

**Architectural Challenge:**
- Each graph is self-referential (internal relationships only)
- **Multi-hop reasoning problem:** Queries spanning multiple graphs fail
  - Example: "How did election doctrine evolve from Calvin to Owen to Spurgeon?"
  - Requires: Reference Books → Spurgeon graph + graphs containing Owen

**Identified Need:**
- Cross-graph relationships for multi-hop queries
- Reference book graph needs links to author-specific graphs
- Entity disambiguation across graphs (same author in multiple graphs)

---

## Future Direction: Vector Database Decision

### Current Status
- ChromaDB implementation working but original scope exceeded
- Faiss implementation complete but deleted (direction changed)
- Neo4j infrastructure prepared but not implemented (file size issues)

### Options Being Considered
1. **Qdrant (embedded mode)**
   - Local deployment
   - Built-in filtering and metadata
   - Scalable architecture

2. **Weaviate**
   - Schema-based structure
   - GraphQL query interface
   - Hybrid search capabilities

### Decision Criteria
- Handle 18.5GB graph corpus with 152M+ edges
- Support cross-graph relationship queries
- Manageable memory footprint
- Efficient multi-hop reasoning
- Local deployment option

---

## Technical Debt & Known Limitations

### NetworkX Limitations
- Entire graph must fit in memory
- 20GB memory usage for large graphs
- GML format expansion: disk → memory overhead

### Multi-Graph Challenges
- No cross-graph relationships implemented
- Entity disambiguation needed
- Multi-hop query support missing

### Production Readiness
- No authentication/authorization
- Hardcoded configurations
- No rate limiting
- Local deployment only

---

## Summary: Problems → Solutions → Evolution

| Phase | Problem | Solution Attempted | Outcome | Status |
|-------|---------|-------------------|---------|--------|
| **Phase 1** | Limited to Spurgeon only | Expand to 125 EPUB volumes | ✅ Success | Complete |
| **Phase 2** | Vector search insufficient | Build knowledge graphs | ✅ Success | Complete |
| **Phase 3** | Missing edges (0 edges created) | Refactor edge generation logic | ✅ Fixed | Complete |
| **Phase 4** | RAM exhaustion (>32GB) | SQLite-backed edge generation | ✅ Fixed | Complete |
| **Phase 5** | 20-hour processing time | 7-worker parallel processing | ✅ 4-6× speedup | Complete |
| **Phase 6** | ChromaDB scaling concerns | Faiss vector store | ✅ Implemented, then ❌ Deleted | Abandoned |
| **Phase 7** | NetworkX memory overhead | Neo4j migration | ⚠️ File size too large | Halted |
| **Phase 8** | Cross-graph relationships | TBD: Qdrant or Weaviate | 🔄 Planning | Current |

---

## Key Metrics

### Dataset
- **Total Documents:** 125+ EPUB volumes + Spurgeon works + reference books
- **Total Chunks:** 2.6M+ text nodes
- **Total Edges:** 152M+ relationships
- **Graph Files:** 4 checkpoints, 18.5GB total

### Performance
- **Graph Processing:** 7 CPU cores, 4-6× speedup
- **Memory Usage:** ≤20GB (controlled)
- **Edge Generation:** 3-6 hours per large file
- **Vector DB:** 56.8MB (ChromaDB)

### Code Quality
- 26 total commits
- Comprehensive documentation (25+ MD files)
- Testing framework (31 tests, 100% pass for Faiss phase)
- Checkpointing and resume capability

---

## Next Steps (Proposed)

1. **Immediate:**
   - Finalize vector database choice (Qdrant vs Weaviate)
   - Implement cross-graph relationship model

2. **Short-term:**
   - Entity disambiguation across graphs
   - Multi-hop query support
   - Production hardening (auth, rate limiting, config)

3. **Long-term:**
   - Distributed graph processing
   - Real-time graph updates
   - Advanced semantic clustering
   - Interactive graph visualization

---

**Document Version:** 1.0
**Last Updated:** October 12, 2025
**Status:** Active Development - Phase 8 (Vector DB Selection)

# Spurgeon RAG App - Architecture & Overview

## 🎯 What It Does

A **Retrieval-Augmented Generation (RAG)** web application that allows users to ask questions about Charles Spurgeon's sermons and receive AI-powered answers based on actual sermon content.

### User Flow:
1. User asks: *"What does Spurgeon say about faith?"*
2. System searches sermon database for relevant passages about faith
3. Retrieves top 4 most relevant sermon excerpts
4. Sends excerpts + question to local LLM (Llama 3.2 8B)
5. LLM generates answer based only on retrieved context
6. User sees answer + can view source sermon excerpts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Streamlit Web App)                       │
│  - Mobile-first responsive design                           │
│  - Chat interface with source citations                     │
│  - Real-time configuration                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE FLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. QUERY INPUT                                             │
│     └─ User question via chat input                         │
│                                                             │
│  2. EMBEDDING & RETRIEVAL                                   │
│     ├─ Convert question to vector (all-MiniLM-L6-v2)        │
│     ├─ Semantic search in ChromaDB                          │
│     └─ Retrieve top-k relevant chunks (default: 4)          │
│                                                             │
│  3. CONTEXT ASSEMBLY                                        │
│     └─ Combine retrieved chunks into prompt context         │
│                                                             │
│  4. LLM GENERATION                                          │
│     ├─ Send context + question to LM Studio API            │
│     ├─ Llama 3.2 8B generates answer                        │
│     └─ Answer constrained to provided context               │
│                                                             │
│  5. RESPONSE RENDERING                                      │
│     ├─ Display answer in chat                               │
│     └─ Show source citations with expandable excerpts       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
spurgeon-rag-app/
│
├── app.py                          # Main Streamlit web application
│   ├── Page configuration & mobile-first CSS
│   ├── Session state management
│   ├── Sidebar settings (LM Studio, RAG params)
│   ├── Vector DB loading (@st.cache_resource)
│   ├── Query function (retrieval + generation)
│   └── Chat UI with source display
│
├── setup_rag.py                    # One-time vector DB setup script
│   ├── PDF loading (from C:\Users\danieo\Downloads\sp-library)
│   ├── Text chunking (1000 chars, 200 overlap)
│   ├── Embedding generation (all-MiniLM-L6-v2)
│   └── ChromaDB persistence
│
├── query_spurgeon.py               # CLI alternative (legacy)
│   └── Command-line RAG interface
│
├── requirements.txt                # Python dependencies
│   ├── streamlit (web framework)
│   ├── langchain (RAG framework)
│   ├── chromadb (vector database)
│   ├── sentence-transformers (embeddings)
│   └── requests (HTTP client)
│
├── vector_db/                      # Persisted vector database
│   ├── chroma.sqlite3              # SQLite metadata store (56.8 MB)
│   └── 5836d13c-.../               # Vector embeddings
│
├── README.md                       # User setup guide
├── MOBILE_OPTIMIZATION.md          # Mobile-first improvements
└── ARCHITECTURE.md                 # This file
```

---

## 🔧 Core Components

### 1. **Embedding Model** (all-MiniLM-L6-v2)
- **Purpose**: Convert text to 384-dimensional vectors
- **Where Used**:
  - Setup: Embed all sermon chunks → vector DB
  - Runtime: Embed user questions → semantic search
- **Performance**: CPU-optimized, fast inference
- **Library**: HuggingFace Sentence Transformers

### 2. **Vector Database** (ChromaDB)
- **Purpose**: Store & search sermon embeddings
- **Size**: ~56.8 MB (sermon collection)
- **Search Method**: Cosine similarity
- **Persistence**: SQLite backend
- **Location**: `./vector_db/`

### 3. **LLM Inference** (LM Studio + Llama 3.2 8B)
- **Deployment**: Local server (http://10.160.212.212:1234)
- **Model**: llama3.2-8b-ins-grpo
- **API**: OpenAI-compatible endpoints
- **Parameters**:
  - Temperature: 0.7 (configurable)
  - Max tokens: 500 (configurable)
  - Timeout: 60s

### 4. **Web Framework** (Streamlit)
- **UI Type**: Single-page chat application
- **State Management**: st.session_state
- **Caching**: @st.cache_resource for embeddings
- **Mobile**: Responsive CSS, collapsed sidebar

### 5. **RAG Framework** (LangChain)
- **Document Loaders**: PyPDFLoader
- **Text Splitters**: RecursiveCharacterTextSplitter
- **Vector Stores**: Chroma integration
- **Embeddings**: HuggingFaceEmbeddings wrapper

---

## 🔄 Data Flow

### Setup Phase (One-time)
```
PDFs → Load → Split into chunks → Embed → Store in ChromaDB
       (PyPDF)  (1000/200 chars)   (384-d)   (56.8 MB)
```

### Query Phase (Runtime)
```
Question → Embed → Similarity Search → Top-k chunks → Prompt → LLM → Answer
          (384-d)   (ChromaDB)          (k=4)         Assembly  (Llama)
```

---

## 🌐 Network Architecture

```
┌──────────────┐     HTTP      ┌─────────────────┐
│   Browser    │◄──────────────►│  Streamlit App  │
│  (User)      │   localhost    │  (port 8501)    │
└──────────────┘     :8501      └─────────────────┘
                                          │
                                          │ Reads
                                          ▼
                                 ┌─────────────────┐
                                 │   ChromaDB      │
                                 │  (vector_db/)   │
                                 └─────────────────┘
                                          │
                                          │ HTTP POST
                                          ▼
                                 ┌─────────────────┐
                                 │   LM Studio     │
                                 │  10.160.212.212 │
                                 │    :1234        │
                                 └─────────────────┘
```

---

## 🎨 Mobile-First Design

### Responsive Breakpoints:
- **Mobile**: < 768px (1rem padding, collapsed sidebar)
- **Tablet**: 768px - 1023px (2rem padding, 900px max-width)
- **Desktop**: ≥ 1024px (3rem padding, 1100px max-width)

### Touch Optimization:
- 44px minimum touch targets (Apple/Android guidelines)
- 16px font size (prevents iOS zoom)
- Full-width buttons on mobile
- Collapsible sidebar by default

---

## 📊 Performance Characteristics

### Latency Breakdown:
1. **Embedding**: ~50-200ms (CPU, question → vector)
2. **Retrieval**: ~10-50ms (ChromaDB similarity search)
3. **LLM Inference**: ~2-10s (depends on token count, network)
4. **Total**: ~2.5-11s per query

### Resource Usage:
- **Memory**: ~1-2 GB (Streamlit + embeddings model)
- **Disk**: 56.8 MB (vector DB)
- **Network**: ~1-10 KB/query (LM Studio API)

### Bottlenecks:
1. **LLM inference time** (largest contributor)
2. **Cold start**: Embedding model load (~3-5s first query)
3. **Network latency**: To LM Studio server

---

## 🔐 Security Considerations

### Current State (NOT Production Ready):
❌ Hardcoded IP addresses
❌ No authentication/authorization
❌ No input validation/sanitization
❌ No rate limiting
❌ No HTTPS/TLS encryption
❌ Exposed internal network endpoints

### Required for Production:
✅ Environment variables for config
✅ API key authentication
✅ Input validation & sanitization
✅ Rate limiting (per-user/IP)
✅ HTTPS with valid certificates
✅ Network segmentation
✅ Logging & monitoring

---

## 🚀 Scalability Path

### Current: Single-User Development
- Local embedding model
- Single ChromaDB instance
- Direct LM Studio connection

### Next: Multi-User Deployment
- Shared ChromaDB server
- Load-balanced LM Studio instances
- Caching layer (Redis)
- Async request handling

### Future: Production Scale
- Managed vector DB (Pinecone, Weaviate)
- Cloud LLM API (Anthropic, OpenAI)
- CDN for static assets
- Kubernetes orchestration

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **Backend** | Python 3.8+ | Application logic |
| **Embeddings** | all-MiniLM-L6-v2 | Text → vector conversion |
| **Vector DB** | ChromaDB | Similarity search |
| **LLM** | Llama 3.2 8B | Answer generation |
| **LLM Server** | LM Studio | Local inference |
| **Framework** | LangChain | RAG orchestration |
| **PDF Processing** | PyPDF | Document loading |

---

## 📈 Key Metrics

- **Vector DB Size**: 56.8 MB
- **Embedding Dimensions**: 384
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Default Context Chunks**: 4
- **LLM Temperature**: 0.7
- **Max Tokens**: 500
- **Request Timeout**: 60 seconds

---

## 💡 How RAG Works (Simple Explanation)

**Traditional ChatGPT:**
- User asks question → LLM answers from training data
- Problem: Can hallucinate, no source citations

**This RAG System:**
1. User asks question
2. Find relevant sermon excerpts (semantic search)
3. Give LLM only those excerpts + question
4. LLM answers based only on provided context
5. User sees answer + exact sources

**Benefits:**
✅ Grounded in actual sermon text
✅ Source citations for verification
✅ No hallucinations about Spurgeon
✅ Works with custom/private documents
✅ Lower cost than fine-tuning

---

## 🎯 Use Cases

1. **Theological Research**: Find Spurgeon's views on specific topics
2. **Sermon Preparation**: Quote accurate sources in sermons
3. **Bible Study**: Explore Spurgeon's commentary on verses
4. **Historical Analysis**: Understand 19th-century Reformed theology
5. **Educational**: Learn from Spurgeon's preaching style

---

## 🔮 Future Enhancements

- [ ] Multi-document support (add more authors)
- [ ] Conversation memory (follow-up questions)
- [ ] Advanced filters (date, sermon series, Bible passage)
- [ ] Export answers as PDF/Markdown
- [ ] Audio sermon integration
- [ ] Multilingual support
- [ ] Fine-tuned embeddings for theology
- [ ] Hybrid search (keyword + semantic)

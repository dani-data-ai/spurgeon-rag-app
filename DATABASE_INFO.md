# Database & Dataset Information

## 📊 Database System

The Spurgeon RAG app uses **ChromaDB** as its vector database system.

### Technology Stack:

**Vector Database:** ChromaDB (Chroma)
- Local, persistent vector database
- Stores document embeddings for semantic search
- Location: `./vector_db/` directory

**Embedding Model:** HuggingFace Sentence Transformers
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Converts text to vector embeddings (384 dimensions)
- Runs locally on CPU (can be switched to GPU)

**Reranker:** Cross-Encoder
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Improves search relevance by reranking retrieved documents
- Optional feature (can be toggled on/off)

## 📚 Dataset Source

**PDF Source Folder:** `C:\Users\danieo\Downloads\sp-library`

The setup process:
1. **Recursively scans** the source folder for all PDF files (including subdirectories)
2. **Loads PDFs** using PyPDFLoader (extracts text from each page)
3. **Chunks documents** into 1000-character segments with 200-character overlap
4. **Creates embeddings** for each chunk using the sentence transformer model
5. **Stores in ChromaDB** for fast semantic similarity search

## 🔄 Data Processing Pipeline

```
PDF Files (sp-library/)
    ↓
PyPDFLoader (extracts text)
    ↓
RecursiveCharacterTextSplitter (chunks: 1000 chars, overlap: 200)
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
    ↓
ChromaDB Vector Database (./vector_db/)
    ↓
Query → Semantic Search → Reranking → Results
```

## 🔍 How Queries Work

1. **User submits question**
2. **Query is embedded** using the same all-MiniLM-L6-v2 model
3. **Semantic search** in ChromaDB finds similar chunks (top 20 candidates)
4. **Reranking** (optional) improves relevance using cross-encoder
5. **Top N chunks** sent to LLM (GPT-4, etc.) with the question
6. **LLM generates answer** based on retrieved context

## 📁 Current Setup

**Vector Database Location:** `./vector_db/`
**Embedding Dimensions:** 384
**Chunk Size:** 1000 characters
**Chunk Overlap:** 200 characters
**Default Retrieval:** 20 candidates → 5 final chunks (with reranking)

## 🛠️ Setup Command

To rebuild the vector database from your PDF collection:

```bash
python setup_rag.py
```

This will:
- Scan all PDFs in `C:\Users\danieo\Downloads\sp-library`
- Process and chunk the documents
- Create embeddings
- Build the ChromaDB vector database in `./vector_db/`

## 💡 Benefits of This Approach

✅ **Local & Private** - All embeddings and search happen locally
✅ **Fast Search** - Vector similarity search is very quick
✅ **Semantic Understanding** - Finds conceptually similar content, not just keyword matches
✅ **Scalable** - ChromaDB can handle millions of documents efficiently
✅ **No API Costs** - Embedding and search are free (runs locally)
✅ **Persistent** - Database is saved and can be reused across sessions

## 📈 Current Database Stats

To check your current database size, you can:
1. Look at the `./vector_db/` folder size
2. Run the app and it will display the number of chunks loaded
3. Check the console output from `setup_rag.py` for exact chunk count

---

**Note:** The organized PDF library you created is at:
`C:\Users\danieo\Downloads\sp-library-organized`

If you want to use this organized version instead, update line 14 in `setup_rag.py`:
```python
pdf_folder = r"C:\Users\danieo\Downloads\sp-library-organized"
```

Then run `python setup_rag.py` to rebuild the database with the cleaned PDFs.

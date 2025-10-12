# Spurgeon Sermon RAG Web App

A Retrieval-Augmented Generation (RAG) web application that answers questions about Charles Spurgeon's sermons using Llama 3.2 8B via LM Studio.

## Features

- 💬 Interactive chat interface
- 📚 Source citation for answers
- ⚙️ Configurable RAG parameters (temperature, tokens, context chunks)
- 🔍 Semantic search through Spurgeon's sermons
- 📡 LM Studio connection monitoring

## Prerequisites

1. **LM Studio** running locally on `localhost:1234`
   - Download from: https://lmstudio.ai/
   - Load the Llama 3.2 8B model

2. **Python 3.8+**

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your PDFs

PDFs are stored in `C:\Users\danieo\Downloads\sp-library` with subdirectories:

```
sp-library/
├── autobiography/
│   ├── CHS_Autobiography Vol 1.pdf
│   ├── CHS_Autobiography Vol 2.pdf
│   └── ...
└── CHS_MTP Pulpit Sermons BY TEXT/
    ├── CHS 01 Genesis.pdf
    ├── CHS 02 Exodus.pdf
    └── ...
```

The setup script automatically scans all subdirectories for PDF files.

### 3. Build the Vector Database

Run the setup script to process PDFs and create embeddings:

```bash
python setup_rag.py
```

This will:
- Load all PDFs from the `pdfs` folder
- Split them into chunks
- Create embeddings using `all-MiniLM-L6-v2`
- Store them in a ChromaDB vector database

### 4. Start LM Studio

1. Open LM Studio
2. Load the Llama 3.2 8B model (or your preferred model)
3. Start the local server (should be running on `http://localhost:1234`)

### 5. Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type your question in the chat input at the bottom
2. **View Sources**: Click "View Sources" below each answer to see the sermon excerpts used
3. **Adjust Settings**: Use the sidebar to:
   - Change LM Studio URL and model name
   - Adjust temperature and max tokens
   - Configure number of context chunks
   - Test connection to LM Studio
4. **Clear History**: Click "Clear Chat History" to start fresh

## Configuration

### Sidebar Settings

- **LM Studio URL**: Default `http://localhost:1234/v1/chat/completions`
- **Model Name**: Default `llama-3.2-8b` (adjust to your model)
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **Max Tokens**: Maximum length of generated answers
- **Number of Context Chunks**: How many relevant passages to retrieve (1-10)

## Project Structure

```
.
├── app.py                 # Streamlit web application
├── setup_rag.py          # Vector database setup script
├── query_spurgeon.py     # CLI query interface (alternative)
├── requirements.txt      # Python dependencies
├── pdfs/                 # Your Spurgeon sermon PDFs
└── vector_db/            # ChromaDB vector database (created after setup)
```

## Troubleshooting

### "Cannot connect to LM Studio"
- Ensure LM Studio is running
- Check that the server is on `localhost:1234`
- Click "Test LM Studio Connection" in the sidebar

### "Vector DB not found"
- Run `python setup_rag.py` first to build the database
- Ensure the `pdfs` folder contains PDF files

### Slow responses
- Reduce "Max Tokens" in settings
- Reduce "Number of Context Chunks"
- Use a smaller model in LM Studio

## Alternative: CLI Interface

If you prefer a command-line interface, use the original script:

```bash
python query_spurgeon.py
```

## Technologies Used

- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embeddings (all-MiniLM-L6-v2)
- **LM Studio**: Local LLM inference (Llama 3.2 8B)
- **PyPDF**: PDF processing

## License

MIT License

the logic of the app is this: vol 1-25 for example has source material (complete works of john owen, complete works of john bunyan, for example) . Then vol 26-70 other puritan sermons and so on. Spurgeon has sermons and books of spurgeon. The refrence books graph has biografies, and theology of the puritans , and election in the writing of augustin, john owen and spurgeon books for example. The source material has the authors, maybe some mentions of theyre times (dates, names of political and historical figures and places ) and tehmes - elections, jobs suffering, fasting and prayer and so on, themes that repeat from one author to the other, but maybe are not explicitly mentions in these documents. The refrences books graphs has acutally this relationships done by a human author who is like an agregatior of times, places, names, themes. The books analyse influences among authors, figures that were living in the same place/period, author that have writen about same themes. Now, if i makeeach model refrence himself, how wil multihop logic work in the end app, when the user askes - how din election of the saint changed from calvin to owen to spurgeon? this would need refrence books model  to have a relationshipt to the spurgeon graph and also to the graphs were owen is mentioned and has his documents and sermons.
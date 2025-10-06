from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import requests
import json

# ====================
# STEP 1: Load PDFs (Recursively from all subdirectories)
# ====================
print("Loading PDFs...")
documents = []
pdf_folder = r"C:\Users\danieo\Downloads\sp-library"

# Recursively find all PDF files in folder and subfolders
pdf_files = []
for root, dirs, files in os.walk(pdf_folder):
    for filename in files:
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            pdf_files.append(pdf_path)

print(f"Found {len(pdf_files)} PDF files")

# Load each PDF
for pdf_path in pdf_files:
    relative_path = os.path.relpath(pdf_path, pdf_folder)
    print(f"Loading {relative_path}...")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    except Exception as e:
        print(f"  ⚠️ Error loading {relative_path}: {str(e)}")
        continue

print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} PDF files")

# ====================
# STEP 2: Split into Chunks
# ====================
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust based on your needs
    chunk_overlap=200,  # Overlap to maintain context
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ====================
# STEP 3: Create Embeddings
# ====================
print("\nCreating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you have GPU
)

# ====================
# STEP 4: Create Vector Database
# ====================
print("\nCreating vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)

print("\nVector database created successfully!")
print(f"Stored {len(chunks)} chunks in database")
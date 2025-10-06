from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests
import json

# ====================
# Load Vector Database
# ====================
print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# ====================
# Query Function
# ====================
def query_spurgeon(question):
    # 1. Find relevant chunks
    print(f"\nSearching for: {question}")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    
    # 2. Build context from chunks
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 3. Create prompt
    prompt = f"""Based on the following excerpts from Charles Spurgeon's sermons, answer the question.

Context:
{context}

Question: {question}

Answer based only on the context above:"""
    
    # 4. Send to LM Studio (Llama 3.2)
    print("\nQuerying Llama 3.2...")
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "llama-3.2-8b",  # Adjust to your model name
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    result = response.json()
    answer = result['choices'][0]['message']['content']
    
    return answer, relevant_docs

# ====================
# Interactive Query Loop
# ====================
print("\n" + "="*50)
print("Spurgeon Sermon RAG System Ready!")
print("="*50)

while True:
    question = input("\nAsk a question (or 'quit' to exit): ")
    
    if question.lower() == 'quit':
        break
    
    answer, sources = query_spurgeon(question)
    
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer)
    
    print("\n" + "="*50)
    print("SOURCES:")
    print("="*50)
    for i, doc in enumerate(sources, 1):
        print(f"\n{i}. {doc.metadata.get('source', 'Unknown')}")
        print(f"   Preview: {doc.page_content[:200]}...")
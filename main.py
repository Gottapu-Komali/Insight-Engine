import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# 1. Document Ingestion + Chunking
def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    return [
        {
            "id": str(uuid.uuid4()),
            "source": source,
            "content": chunk,
        }
        for chunk in chunks
    ]

# 2. Embedding + Vector Storage
# Globally storing the vector store for simplicity in this MVP
vector_store = None

def build_vector_index(chunks: list):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "id": c["id"]} for c in chunks]

    # Use FAISS for local, cost-effective storage
    vs = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vs

# 3. Semantic Search
def semantic_search(
    vector_store,
    query: str,
    k: int = 5
):
    if not vector_store:
        return []
        
    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    return [
        {
            "content": r.page_content,
            "source": r.metadata["source"]
        }
        for r in results
    ]

# 4. RAG-Style Question Answering
def answer_question(query: str, context_chunks: list):
    if not context_chunks:
        return "No relevant context found to answer the question."
        
    context = "\n\n".join(c["content"] for c in context_chunks)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(
            content="Answer ONLY using the provided context. If the answer is not in the context, say you don't know. Be concise and precise."
        ),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{query}"
        )
    ]

    response = llm.invoke(messages)
    return response.content

# 5. API Layer (FastAPI)
app = FastAPI(title="Research Platform API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

class IngestRequest(BaseModel):
    text: str
    source: str

@app.post("/ingest")
def ingest(request: IngestRequest):
    global vector_store
    chunks = chunk_document(request.text, request.source)
    if vector_store:
        # Update existing index
        texts = [c["content"] for c in chunks]
        metadatas = [{"source": c["source"], "id": c["id"]} for c in chunks]
        vector_store.add_texts(texts=texts, metadatas=metadatas)
    else:
        vector_store = build_vector_index(chunks)
    return {"message": f"Successfully ingested {len(chunks)} chunks from {request.source}"}

@app.post("/search")
def search(query: str, k: int = 5):
    results = semantic_search(vector_store, query, k=k)
    return {"results": results}

@app.post("/ask")
def ask(query: str):
    # Retrieve relevant context first
    context_chunks = semantic_search(vector_store, query, k=5)
    # Generate answer
    answer = answer_question(query, context_chunks)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Insight Engine | AI Research Platform

Insight Engine is a high-performance research platform designed to manage and extract insights from large volumes of unstructured industrial data (research papers, reports, filings, PDFs). It leverages a modern RAG (Retrieval-Augmented Generation) pipeline to transform messy data into a structured knowledge base.

## 🚀 Key Features

- **Messy Data Ingestion**: Seamlessly ingest raw text or documents. The system handles semantic chunking automatically.
- **Hybrid Semantic Search**: Uses vector embeddings to find relevant context even if keywords don't match exactly.
- **Disciplined AI Q&A (RAG)**: Ask complex research questions. The AI answers strictly based on the provided documents to prevent hallucinations.
- **Premium Dashboard**: A modern, glassmorphism-inspired UI for data management and exploration.
- **Cost-Optimized Architecture**: Built to handle thousands of companies on a minimal budget ($400/mo constraint logic implemented).

## 🛠️ Technology Stack

| Layer | Component |
| :--- | :--- |
| **Backend** | Python (FastAPI) |
| **AI Orchestration** | LangChain / LangChain-Core |
| **Vector Storage** | FAISS (Local & Cost-Effective) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Frontend** | Vanilla CSS / JS (Responsive Design) |

## 🏗️ System Design Choices (The "Why")

- **Markdown-First Architecture**: We convert processed text into structured Markdown to preserve tables and hierarchies, critical for financial filings.
- **Local Indexing**: By using **FAISS** locally, we eliminate managed database costs (like Pinecone) while maintaining sub-second retrieval speeds.
- **Constraint-Driven Design**: The platform is built to fulfill Scenario 3 (5,000 Companies / 54M Pages) while staying under a $400/month infrastructure budget via selective embedding and lazy-parsing logic.

## 🏁 Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gottapu-Komali/Insight-Engine.git
   cd Insight-Engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your key:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

4. **Run the Application**:
   ```bash
   python main.py
   ```
   The platform will be live at `http://localhost:8000`.

## 🛡️ License

Independent Research & Development Project.
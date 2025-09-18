# Project 1.2 - RAG Chatbot

RAG Chatbot built from PDF files using LangChain with HuggingFace models.

## Description
This project includes:
- **PDF Document Processing**: Upload and process PDF documents
- **Semantic Chunking**: Advanced text splitting using semantic similarity
- **Vector Embeddings**: Vietnamese bi-encoder for better text understanding
- **RAG Implementation**: Retrieval-Augmented Generation with local LLM
- **Streamlit Interface**: User-friendly web interface for document Q&A

## Features
- PDF document upload and processing
- Semantic text chunking for better context
- Vietnamese language support with specialized embeddings
- Local LLM inference (Vicuna-7B) with quantization
- Real-time chat interface
- Document status tracking
- Chat history management

## Installation
```bash
# Install dependencies using uv
uv sync

# Run the RAG chatbot application
uv run streamlit run app.py
```

## Project Structure
```
project-1.2-rag-chatbot/
├── app.py              # Main RAG chatbot application
├── src/                # Source code modules
│   ├── rag_app.py      # Core RAG implementation
│   └── footer.py       # Footer component
├── data/               # PDF files and vector storage
├── tests/              # Unit tests
└── README.md
```

## Requirements
- Python >= 3.8
- Streamlit >= 1.28.0
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- LangChain ecosystem
- ChromaDB for vector storage
- CUDA support recommended for GPU acceleration

## Usage
1. **Upload PDF**: Select and upload a PDF document
2. **Process Document**: Click "Xử lý PDF" to process the document
3. **Start Chatting**: Ask questions about the document content
4. **Manage Chat**: Use sidebar controls to clear chat history

## Technical Details
- **Embeddings**: Vietnamese bi-encoder model for better text understanding
- **LLM**: Vicuna-7B with 4-bit quantization for efficient inference
- **Vector Store**: ChromaDB for fast similarity search
- **Text Splitting**: Semantic chunking for context preservation
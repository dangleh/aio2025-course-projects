# Project 1.2 - RAG Chatbot

Xây dựng RAG Chatbot từ file PDF với LangChain.

## Mô tả

Dự án này bao gồm:

- Xử lý và đọc file PDF
- Tạo vector embeddings
- Xây dựng chatbot với RAG (Retrieval-Augmented Generation)
- Giao diện web với Streamlit

## Cài đặt

```bash
# Sử dụng uv để cài đặt dependencies
uv sync

# Tạo file .env với API keys
cp .env.example .env

# Chạy ứng dụng
uv run streamlit run app.py
```

## Cấu trúc dự án

```
project-1.2-rag-chatbot/
├── app.py              # Ứng dụng chính
├── src/                # Source code
│   ├── document_processor.py
│   ├── vector_store.py
│   └── chatbot.py
├── data/               # PDF files
├── tests/              # Unit tests
├── .env.example        # Template cho environment variables
└── README.md
```

## Yêu cầu

- Python >= 3.8
- OpenAI API key
- LangChain
- FAISS

import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import getpass
import time


import getpass
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = ""

@st.cache_resource
def load_embeddings():
    embedding =  GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embedding

@st.cache_resource  
def load_llm():
    MODEL_NAME = "gemini-2.0-flash"
    return ChatGoogleGenerativeAI(model=MODEL_NAME)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs,
                                      embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()

    os.unlink(tmp_file_path)
    return retriever, len(docs), docs


def get_pdf_rag_prompt():
    return ChatPromptTemplate.from_template(
        """
        System instruction: Bạn là một trợ lý AI giúp trả lời câu hỏi dựa trên nội dung tài liệu PDF.
        Hãy sử dụng thông tin từ context sau để trả lời câu hỏi một cách chính xác và ngắn gọn.

        Context: {context}

        Question: {question}

        Answer:
        """
    )


def add_message(role, content):
    """Thêm tin nhắn vào lịch sử chat"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def clear_chat():
    """Xóa lịch sử chat"""
    st.session_state.chat_history = []

def display_chat():
    """Hiển thị lịch sử chat"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊")

# UI
def main():
    initialize_session_state()

    st.set_page_config(
        page_title="PDF RAG Chatbot", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("PDF RAG Assistant")
    st.logo("./aibybit_logo.png", size="large")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        
        # Load models
        if not st.session_state.models_loaded:
            st.warning("⏳ Đang tải models...")
            with st.spinner("Đang tải AI models..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm()
                st.session_state.models_loaded = True
            st.success("✅ Models đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ Models đã sẵn sàng!")

        st.markdown("---")
        
        # Upload PDF
        st.subheader("📄 Upload tài liệu")
        uploaded_files = st.file_uploader("Upload file PDF", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("Xử lý PDF"):
            try:
                with st.spinner("Đang xử lý PDF..."):
                    # Xử lý nhiều PDF
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        retriever, num_chunks, chunks = process_pdf(uploaded_file)
                        all_chunks.extend(chunks)

                    # Tạo vector database từ tất cả chunks
                    vector_db = Chroma.from_documents(
                        documents=all_chunks,
                        embedding=st.session_state.embeddings
                    )
                    retriever = vector_db.as_retriever()

                    st.session_state.rag_chain = retriever
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = ", ".join([f.name for f in uploaded_files])
                    st.session_state.chunks = all_chunks

                    # Hiển thị thông báo
                    message = f"""
                    ✅ Đã xử lý thành công file **{st.session_state.pdf_name}**!
                    Tài liệu được chia thành {len(st.session_state.chunks)} phần.
                    Bạn có thể bắt đầu đặt câu hỏi về nội dung tài liệu.
                    """

                    clear_chat()
                    add_message("assistant", message)
                    st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xử lý PDF: {str(e)}")
        
        # PDF status
        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa có tài liệu")
            
        st.markdown("---")
        
        # Chat controls
        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()
            
        st.markdown("---")
        
        # Instructions
        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Nhận trả lời** - AI sẽ trả lời dựa trên nội dung PDF
        """)

    # Main content
    st.markdown("*Trò chuyện với Chatbot để trao đổi về nội dung tài liệu PDF của bạn*")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat()
    
    # Chat input
    if st.session_state.models_loaded and st.session_state.rag_chain:
        if st.session_state.pdf_processed:
            # User input
            user_input = st.chat_input("Nhập câu hỏi của bạn...")
            
            if user_input:
                # Add user message
                add_message("user", user_input)
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            relevant_docs = st.session_state.rag_chain.invoke(user_input)
                            context = "\n\n".join([doc.page_content for doc in relevant_docs])

                            prompt = get_pdf_rag_prompt()
                            output_parser = StrOutputParser()
                            chain = prompt | st.session_state.llm | output_parser

                            answer = chain.invoke({"context": context, "question": user_input})

                            st.markdown(f"**Answer**: {answer}")
                            # Clean up the response
                            # if 'Answer:' in answer:
                            #     answer = answer.split('Answer:')[0].strip()
                            
                            # Add assistant message to history
                            add_message("assistant", answer)
                            
                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)

if __name__ == "__main__":
    main()
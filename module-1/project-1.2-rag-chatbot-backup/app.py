"""
RAG Chatbot Application
Xây dựng RAG Chatbot từ file PDF với LangChain
"""

import streamlit as st
import os
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.chatbot import RAGChatbot

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Chatbot")
    st.markdown("Chatbot thông minh với khả năng trả lời câu hỏi dựa trên tài liệu PDF")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Vui lòng cấu hình OPENAI_API_KEY trong file .env")
        st.code("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize components
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Tải lên tài liệu")
        
        uploaded_file = st.file_uploader(
            "Chọn file PDF",
            type="pdf",
            help="Tải lên file PDF để chatbot có thể trả lời câu hỏi về nội dung"
        )
        
        if uploaded_file is not None:
            if st.button("Xử lý tài liệu"):
                with st.spinner("Đang xử lý tài liệu..."):
                    try:
                        # Save uploaded file
                        with open(f"data/{uploaded_file.name}", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        processor = DocumentProcessor()
                        documents = processor.process_pdf(f"data/{uploaded_file.name}")
                        
                        # Create vector store
                        vector_store = VectorStore()
                        vector_store.create_from_documents(documents)
                        st.session_state.vector_store = vector_store
                        
                        # Initialize chatbot
                        chatbot = RAGChatbot(vector_store)
                        st.session_state.chatbot = chatbot
                        
                        st.success(f"✅ Đã xử lý thành công: {uploaded_file.name}")
                        st.info(f"📊 Số lượng chunks: {len(documents)}")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý tài liệu: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.chatbot is None:
        st.info("👆 Vui lòng tải lên file PDF để bắt đầu chat")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Lỗi: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()

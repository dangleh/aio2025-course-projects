"""
RAG Chatbot Module
Chatbot với khả năng Retrieval-Augmented Generation
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List
import os

class RAGChatbot:
    def __init__(self, vector_store):
        """
        Khởi tạo RAG Chatbot
        
        Args:
            vector_store: VectorStore instance để tìm kiếm documents
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Define prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp trong context.
            
            Hướng dẫn:
            - Chỉ sử dụng thông tin từ context để trả lời
            - Nếu không có thông tin liên quan trong context, hãy nói rõ rằng bạn không biết
            - Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
            - Nếu có thể, hãy cung cấp thêm ví dụ hoặc giải thích chi tiết
            
            Context:
            {context}
            """),
            ("human", "{question}")
        ])
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format documents thành context string
        
        Args:
            documents: List các Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Tài liệu {i}]\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def chat(self, question: str, k: int = 4) -> str:
        """
        Trả lời câu hỏi sử dụng RAG
        
        Args:
            question: Câu hỏi của người dùng
            k: Số lượng documents để retrieve
            
        Returns:
            Câu trả lời từ chatbot
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            if not relevant_docs:
                return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu."
            
            # Format context
            context = self._format_context(relevant_docs)
            
            # Create prompt
            prompt = self.prompt_template.format_messages(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm(prompt)
            
            return response.content
            
        except Exception as e:
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
    
    def chat_with_sources(self, question: str, k: int = 4):
        """
        Trả lời câu hỏi kèm theo thông tin về sources
        
        Args:
            question: Câu hỏi của người dùng
            k: Số lượng documents để retrieve
            
        Returns:
            Tuple (answer, sources)
        """
        try:
            # Retrieve relevant documents with scores
            relevant_docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
            
            if not relevant_docs_with_scores:
                return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.", []
            
            # Separate documents and scores
            relevant_docs = [doc for doc, score in relevant_docs_with_scores]
            scores = [score for doc, score in relevant_docs_with_scores]
            
            # Format context
            context = self._format_context(relevant_docs)
            
            # Create prompt
            prompt = self.prompt_template.format_messages(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm(prompt)
            
            # Format sources
            sources = []
            for i, (doc, score) in enumerate(zip(relevant_docs, scores)):
                sources.append({
                    "document": doc.metadata.get("source", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", i),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:200] + "..."
                })
            
            return response.content, sources
            
        except Exception as e:
            return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}", []

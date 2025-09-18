"""
Vector Store Module
Quản lý vector embeddings và similarity search
"""

import os
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List
import pickle

class VectorStore:
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Khởi tạo VectorStore
        
        Args:
            embedding_model: Tên model embedding từ OpenAI
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.index_path = "data/vector_index"
        self.metadata_path = "data/metadata.pkl"
    
    def create_from_documents(self, documents: List[Document]):
        """
        Tạo vector store từ danh sách documents
        
        Args:
            documents: List các Document objects
        """
        if not documents:
            raise Exception("Danh sách documents không được rỗng")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Save to disk
        self.save()
    
    def save(self):
        """Lưu vector store vào disk"""
        if self.vectorstore is None:
            raise Exception("Vector store chưa được khởi tạo")
        
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.vectorstore.docstore._dict, f)
    
    def load(self):
        """Load vector store từ disk"""
        if not os.path.exists(self.index_path):
            raise Exception("Vector store chưa được tạo")
        
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings
            )
        except Exception as e:
            raise Exception(f"Lỗi khi load vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Tìm kiếm documents tương tự với query
        
        Args:
            query: Câu hỏi tìm kiếm
            k: Số lượng kết quả trả về
            
        Returns:
            List các Document tương tự
        """
        if self.vectorstore is None:
            raise Exception("Vector store chưa được khởi tạo")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Tìm kiếm documents với similarity scores
        
        Args:
            query: Câu hỏi tìm kiếm
            k: Số lượng kết quả trả về
            
        Returns:
            List các tuple (Document, score)
        """
        if self.vectorstore is None:
            raise Exception("Vector store chưa được khởi tạo")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)

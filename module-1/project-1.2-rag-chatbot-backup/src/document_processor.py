"""
Document Processor Module
Xử lý và chia nhỏ tài liệu PDF thành các chunks
"""

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Khởi tạo DocumentProcessor
        
        Args:
            chunk_size: Kích thước mỗi chunk (số ký tự)
            chunk_overlap: Số ký tự overlap giữa các chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Trích xuất text từ file PDF
        
        Args:
            pdf_path: Đường dẫn đến file PDF
            
        Returns:
            Text content từ PDF
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Lỗi khi đọc PDF: {str(e)}")
        
        return text
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Xử lý file PDF và chia thành các chunks
        
        Args:
            pdf_path: Đường dẫn đến file PDF
            
        Returns:
            List các Document chunks
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise Exception("Không thể trích xuất text từ PDF")
        
        # Create document
        document = Document(
            page_content=text,
            metadata={"source": os.path.basename(pdf_path)}
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([document])
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks

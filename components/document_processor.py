import PyPDF2
import re
from typing import List, Dict
import io

class DocumentProcessor:
    """
    Handles PDF document processing and intelligent text chunking
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_file) -> Dict[str, any]:
        """
        Extract text from PDF file and preserve metadata
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Dict containing extracted text, metadata, and page information
        """
        try:
            # Read PDF using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract metadata
            metadata = {
                'filename': pdf_file.name,
                'num_pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                'subject': pdf_reader.metadata.get('/Subject', '') if pdf_reader.metadata else ''
            }
            
            # Extract text from each page
            pages_text = []
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        pages_text.append({
                            'page_number': page_num + 1,
                            'text': page_text.strip()
                        })
                        full_text += f"\n\n[Page {page_num + 1}]\n{page_text.strip()}"
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            return {
                'full_text': full_text.strip(),
                'pages': pages_text,
                'metadata': metadata
            }
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('�', '')  # Remove replacement characters
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of text chunks with metadata
        """
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'metadata': {
                        **metadata,
                        'chunk_size': len(current_chunk),
                        'chunk_index': chunk_id
                    }
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last few sentences for overlap
                    overlap_sentences = current_chunk.split('. ')[-2:]
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    'chunk_size': len(current_chunk),
                    'chunk_index': chunk_id
                }
            })
        
        return chunks
    
    def process_document(self, pdf_file) -> List[Dict]:
        """
        Complete document processing pipeline
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            List of processed text chunks with metadata
        """
        # Extract text and metadata
        doc_data = self.extract_text_from_pdf(pdf_file)
        
        # Create chunks
        chunks = self.chunk_text(doc_data['full_text'], doc_data['metadata'])
        
        return chunks
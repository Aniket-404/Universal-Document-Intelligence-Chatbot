import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import json
import re
from collections import Counter
import math
import config

# Import torch for device handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import Hugging Face client
try:
    from .huggingface_client import HuggingFaceEmbeddingModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Fallback to sentence transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence transformers not available. Using TF-IDF fallback.")

class VectorStore:
    """
    Vector store using Sentence Transformers for embeddings and FAISS for similarity search
    """
    
    def __init__(self, model_name: str = None, index_path: str = "vector_index"):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.index_path = index_path
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.dimension = None
        self.use_huggingface = HUGGINGFACE_AVAILABLE
        self.use_sentence_transformers = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_huggingface:
            self._load_huggingface_model()
        elif self.use_sentence_transformers:
            self._load_sentence_transformer_model()
        else:
            self._init_simple_search()
    
    def _load_huggingface_model(self):
        """Load the Hugging Face embedding model"""
        try:
            self.embedding_model = HuggingFaceEmbeddingModel(self.model_name)
            # Get dimension
            self.dimension = self.embedding_model.get_dimension()
            print(f"Loaded HuggingFace embedding model: {self.model_name} (dimension: {self.dimension})")
        except Exception as e:
            print(f"Error loading HuggingFace model: {str(e)}")
            self.use_huggingface = False
            if self.use_sentence_transformers:
                self._load_sentence_transformer_model()
            else:
                self._init_simple_search()
    
    def _load_sentence_transformer_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            # Load with careful device handling - let the library handle device assignment
            self.embedding_model = SentenceTransformer(
                self.model_name, 
                device=None,  # Let the library choose the best device
                trust_remote_code=True
            )
            
            # Get dimension from a sample embedding
            sample_embedding = self.embedding_model.encode(["sample"])
            self.dimension = sample_embedding.shape[1] if hasattr(sample_embedding, 'shape') else len(sample_embedding)
            print(f"Loaded sentence transformer model: {self.model_name} (dimension: {self.dimension})")
        except Exception as e:
            print(f"Error loading sentence transformer model: {str(e)}")
            self.use_sentence_transformers = False
            self._init_simple_search()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing for TF-IDF"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and remove empty strings
        words = [word for word in text.split() if len(word) > 2]
        return words
    
    def _compute_tf(self, words: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        word_count = len(words)
        tf_dict = {}
        for word in words:
            tf_dict[word] = tf_dict.get(word, 0) + 1
        # Normalize by total word count
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / word_count
        return tf_dict
    
    def _compute_idf(self):
        """Compute inverse document frequency for all terms"""
        N = len(self.documents)
        all_words = set()
        for doc in self.documents:
            words = self._preprocess_text(doc['text'])
            all_words.update(set(words))
        
        for word in all_words:
            containing_docs = sum(1 for doc in self.documents 
                                if word in self._preprocess_text(doc['text']))
            self.idf_scores[word] = math.log(N / containing_docs) if containing_docs > 0 else 0
    
    def _compute_tfidf_similarity(self, query: str, doc_text: str) -> float:
        """Compute TF-IDF cosine similarity between query and document"""
        query_words = self._preprocess_text(query)
        doc_words = self._preprocess_text(doc_text)
        
        if not query_words or not doc_words:
            return 0.0
        
        query_tf = self._compute_tf(query_words)
        doc_tf = self._compute_tf(doc_words)
        
        # Get all unique words
        all_words = set(query_words + doc_words)
        
        # Compute TF-IDF vectors
        query_vector = []
        doc_vector = []
        
        for word in all_words:
            idf = self.idf_scores.get(word, 0)
            query_tfidf = query_tf.get(word, 0) * idf
            doc_tfidf = doc_tf.get(word, 0) * idf
            query_vector.append(query_tfidf)
            doc_vector.append(doc_tfidf)
        
        # Compute cosine similarity
        if not query_vector or not doc_vector:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
        query_norm = math.sqrt(sum(a * a for a in query_vector))
        doc_norm = math.sqrt(sum(a * a for a in doc_vector))
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)
    
    def _init_simple_search(self):
        """Initialize simple TF-IDF search"""
        self.vocabulary = {}
        self.idf_scores = {}
        print("Initialized simple TF-IDF search (advanced embeddings not available)")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        if self.use_huggingface or self.use_sentence_transformers:
            try:
                embeddings = self.embedding_model.encode(texts)
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()
                return embeddings.astype('float32')
            except Exception as e:
                print(f"Error creating embeddings, falling back to simple search: {str(e)}")
                self.use_huggingface = False
                self.use_sentence_transformers = False
                self._init_simple_search()
        
        # Return dummy embeddings for simple search
        return np.zeros((len(texts), 100), dtype='float32')
    
    def initialize_index(self):
        """Initialize FAISS index"""
        if not (self.use_huggingface or self.use_sentence_transformers):
            return
        
        if self.dimension is None:
            raise Exception("Embedding model not properly loaded")
        
        # Use IndexFlatIP for cosine similarity (Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        print(f"Initialized FAISS index with dimension {self.dimension}")
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        # Store documents with metadata
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'id': len(self.documents),
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'embedding_id': len(self.documents)
            })
        
        if self.use_huggingface or self.use_sentence_transformers:
            # Initialize index if not done
            if self.index is None:
                self.initialize_index()
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            print(f"Added {len(chunks)} document chunks to FAISS vector store")
        else:
            # For simple search, compute IDF scores
            self._compute_idf()
            print(f"Added {len(chunks)} document chunks to simple vector store")
    
    def search(self, query: str, k: int = 5, similarity_threshold: float = 0.0) -> List[Dict]:
        """Search for similar documents using semantic similarity with very low threshold"""
        if len(self.documents) == 0:
            return []

        if (self.use_huggingface or self.use_sentence_transformers) and self.index is not None:
            return self._advanced_search(query, k, similarity_threshold)
        else:
            return self._simple_search(query, k, similarity_threshold)
    
    def _advanced_search(self, query: str, k: int, similarity_threshold: float) -> List[Dict]:
        """Advanced search using FAISS and sentence transformers"""
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Filter by similarity threshold
            if score >= similarity_threshold and idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'score': float(score),
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def _simple_search(self, query: str, k: int, similarity_threshold: float) -> List[Dict]:
        """Simple search using improved TF-IDF similarity with better matching"""
        if not self.documents:
            return []
        
        # Compute similarities
        similarities = []
        for doc in self.documents:
            # Calculate multiple similarity scores for better matching
            tfidf_similarity = self._compute_tfidf_similarity(query, doc['text'])
            keyword_similarity = self._compute_keyword_similarity(query, doc['text'])
            combined_similarity = max(tfidf_similarity, keyword_similarity * 0.7)  # Boost keyword matches
            
            similarities.append({
                'document': doc,
                'score': combined_similarity,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Always return results, ignore similarity threshold for TF-IDF fallback
        results = []
        for i, result in enumerate(similarities[:k]):
            result['rank'] = i + 1
            results.append(result)
        
        return results
    
    def _compute_keyword_similarity(self, query: str, text: str) -> float:
        """Compute simple keyword-based similarity"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def save_index(self):
        """Save vector store to disk"""
        try:
            if (self.use_huggingface or self.use_sentence_transformers) and self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save documents and metadata
            with open(f"{self.index_path}_docs.pkl", "wb") as f:
                pickle.dump({
                    'documents': self.documents,
                    'dimension': self.dimension,
                    'model_name': self.model_name,
                    'use_huggingface': self.use_huggingface,
                    'use_sentence_transformers': self.use_sentence_transformers,
                    'vocabulary': getattr(self, 'vocabulary', {}),
                    'idf_scores': getattr(self, 'idf_scores', {})
                }, f)
            
            print(f"Saved vector index to {self.index_path}")
        except Exception as e:
            print(f"Error saving index: {str(e)}")
    
    def load_index(self):
        """Load vector store from disk"""
        try:
            if os.path.exists(f"{self.index_path}_docs.pkl"):
                # Load documents and metadata
                with open(f"{self.index_path}_docs.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.dimension = data.get('dimension')
                    self.vocabulary = data.get('vocabulary', {})
                    self.idf_scores = data.get('idf_scores', {})
                    stored_use_hf = data.get('use_huggingface', False)
                    stored_use_st = data.get('use_sentence_transformers', data.get('use_advanced', True))
                
                # Load FAISS index if available and we're using embeddings
                if ((self.use_huggingface or self.use_sentence_transformers) and 
                    (stored_use_hf or stored_use_st) and 
                    os.path.exists(f"{self.index_path}.faiss")):
                    self.index = faiss.read_index(f"{self.index_path}.faiss")
                
                print(f"Loaded vector index from {self.index_path}")
                return True
        except Exception as e:
            print(f"Error loading index: {str(e)}")
        
        return False
    
    def clear_index(self):
        """Clear the current index and documents"""
        self.index = None
        self.documents = []
        self.vocabulary = {}
        self.idf_scores = {}
        print("Cleared vector index")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if ((self.use_huggingface or self.use_sentence_transformers) and self.index) else len(self.documents),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'search_type': 'HuggingFace Embeddings + FAISS' if self.use_huggingface else 'Sentence Transformers + FAISS' if self.use_sentence_transformers else 'Simple TF-IDF'
        }
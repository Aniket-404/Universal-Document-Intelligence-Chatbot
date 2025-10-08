"""
Local Hugging Face model integration with automatic model downloading
"""

import os
import torch
from typing import List, Dict, Optional
import config
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class HuggingFaceClient:
    """
    Client for local Hugging Face models with automatic downloading
    """
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = model_name or config.CHAT_MODEL
        self.cache_dir = cache_dir or config.MODEL_CACHE_DIR
        self.max_length = config.MODEL_MAX_LENGTH
        self.temperature = config.TEMPERATURE
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize models (will be loaded on first use)
        self.tokenizer = None
        self.model = None
        self.model_type = None  # Will be set during loading
        self.is_loaded = False
        
        print(f"HuggingFace Client initialized")
        print(f"Model: {self.model_name}")
        print(f"Cache: {self.cache_dir}")
        print(f"Device: {self.device}")
    
    def _setup_device(self):
        """Setup computation device (CPU/GPU)"""
        if config.DEVICE == "auto":
            if config.USE_CUDA and torch.cuda.is_available():
                device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("Using CPU")
        else:
            device = config.DEVICE
        
        return device
    
    def _load_model(self):
        """Load the model and tokenizer (downloads automatically if not cached)"""
        if self.is_loaded:
            return True
        
        try:
            print(f"Loading model: {self.model_name}")
            print("This might take a few minutes on first run (downloading model)...")

            # Import here to avoid slow startup if not needed
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Determine model type and load accordingly
            is_t5_model = "t5" in self.model_name.lower() or "flan" in self.model_name.lower()
            
            if is_t5_model:
                print("Loading T5/FLAN model for text-to-text generation...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32,  # T5 works better with float32
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model_type = "seq2seq"
                print("T5/FLAN model loaded successfully!")
            else:
                print("Loading causal language model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model_type = "causal"
                
                # Add pad token for causal models
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Causal model loaded successfully!")
            
            self.model.eval()  # Set to evaluation mode
            self.is_loaded = True

            print(f"Model size: ~{self._get_model_size_mb():.1f} MB")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Model will run in offline mode - document search will still work!")
            self.is_loaded = False
            return False
    
    def _get_model_size_mb(self):
        """Estimate model size in MB"""
        if self.model is None:
            return 0
        
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / 1024 / 1024
    
    def generate_response(self, query: str, context: str = "", system_prompt: str = "") -> str:
        """Generate a response given a query and context with offline fallback"""
        # Load model on first use
        if not self.is_loaded:
            success = self._load_model()
            if not success:
                # Return offline fallback response
                return self._generate_offline_response(query, context)
        
        try:
            # Prepare the input text based on model type
            if hasattr(self, 'model_type') and self.model_type == "seq2seq":
                # T5/FLAN models work better with instruction-style prompts
                if context:
                    # For document-based questions
                    context_truncated = context[:800] if len(context) > 800 else context
                    
                    if any(word in query.lower() for word in ['summarize', 'summary', 'main points', 'key points', 'overview']):
                        input_text = f"Summarize the following text: {context_truncated}"
                    else:
                        input_text = f"Answer the question based on the context.\nContext: {context_truncated}\nQuestion: {query}\nAnswer:"
                else:
                    input_text = f"Answer this question: {query}"
                
                # Tokenize for T5
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                # Ensure input_ids are on the same device as the model
                if hasattr(self.model, 'device'):
                    model_device = next(self.model.parameters()).device
                    input_ids = input_ids.to(model_device)
                else:
                    input_ids = input_ids.to(self.device)
                
                # Generate with T5/FLAN
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=200,  # Good length for summaries
                        min_length=20,   # Ensure substantial response
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0
                    )
                
                # Decode T5 response (T5 outputs only the generated text)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # Original logic for causal models (DialoGPT, etc.)
                if context:
                    context_truncated = context[:500] if len(context) > 500 else context
                    
                    if any(word in query.lower() for word in ['summarize', 'summary', 'main points', 'key points', 'overview']):
                        input_text = f"Summarize this: {context_truncated}\nSummary:"
                    else:
                        input_text = f"Context: {context_truncated}\nQuestion: {query}\nAnswer:"
                else:
                    input_text = f"Question: {query}\nAnswer:"
                
                # Tokenize input with simpler approach
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=300)
                
                # Ensure input_ids are on the same device as the model
                if hasattr(self.model, 'device'):
                    model_device = next(self.model.parameters()).device
                    input_ids = input_ids.to(model_device)
                else:
                    input_ids = input_ids.to(self.device)
                
                # Generate response with causal model
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 100,
                        min_length=input_ids.shape[1] + 5,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.1,
                        length_penalty=1.0
                    )
                
                # Decode causal model response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new generated text for causal models
                if response.startswith(input_text):
                    response = response[len(input_text):].strip()
                else:
                    # Fallback: try to find the answer part
                    for separator in ["Answer:", "Summary:", "\nBot:", "\n"]:
                        if separator in response:
                            parts = response.split(separator)
                            if len(parts) > 1:
                                response = parts[-1].strip()
                                break
            
            print(f"Extracted response: '{response[:100]}...'")
            
            # Clean up the response
            cleaned_response = self._clean_response(response)
            
            # Debug logging
            print(f"Raw AI response length: {len(response)}")
            print(f"Cleaned AI response length: {len(cleaned_response)}")
            print(f"Cleaned response: '{cleaned_response[:100]}...'")
            
            # Be more lenient - if we have any response, use it
            if cleaned_response and len(cleaned_response.strip()) > 0:
                return cleaned_response
            elif response and len(response.strip()) > 0:
                # Use raw response if cleaning removed too much
                return response.strip()
            else:
                # Try a simple fallback generation
                print("Attempting fallback generation with simpler prompt...")
                return self._try_simple_generation(query, context)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            # Fall back to offline response
            return self._generate_offline_response(query, context)
    
    def _try_simple_generation(self, query: str, context: str = "") -> str:
        """Try a very simple generation as last resort"""
        try:
            # Ultra-simple prompt
            simple_prompt = f"{query}"
            input_ids = self.tokenizer.encode(simple_prompt, return_tensors="pt", max_length=50)
            
            # Ensure input_ids are on the same device as the model
            if hasattr(self.model, 'device'):
                model_device = next(self.model.parameters()).device
                input_ids = input_ids.to(model_device)
            else:
                input_ids = input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 30,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(simple_prompt):].strip()
            
            if response and len(response) > 2:
                return f"AI Response: {response}"
            
        except Exception as e:
            print(f"Simple generation also failed: {e}")
        
        return self._generate_offline_response(query, context)
    
    def _generate_offline_response(self, query: str, context: str = "") -> str:
        """Generate a structured response when AI model is unavailable or gives poor response"""
        # Check if this is being called because model is unavailable or just poor response
        model_available = self.is_loaded
        note_suffix = "*Note: AI model generated poor response - showing raw content*" if model_available else "*Note: AI model unavailable - showing raw content*"
        
        if context:
            if "Relevant information from your documents:" in context:
                # Extract and format document content
                lines = context.split('\n')
                document_info = []
                current_info = ""
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("From ") and "relevance:" in line:
                        if current_info:
                            document_info.append(current_info)
                        # Extract filename
                        filename = line.split("(relevance:")[0].replace("From ", "").strip()
                        current_info = f"**From {filename}:**"
                    elif line and not line.startswith("Relevant information") and len(line) > 10:
                        current_info += f"\n{line}"
                
                if current_info:
                    document_info.append(current_info)
                
                if document_info:
                    response = "Based on your uploaded documents:\n\n"
                    for info in document_info[:2]:  # Show top 2 sources
                        response += f"{info}\n\n"
                    response += f"\n{note_suffix}"
                    return response
            
            elif "Web search results:" in context:
                # Format web search results
                lines = context.split('\n')
                search_results = []
                
                for line in lines:
                    if line.strip() and not line.startswith('Web search results:'):
                        search_results.append(line.strip())
                
                if search_results:
                    response = "Based on web search results:\n\n"
                    for i, result in enumerate(search_results[:3], 1):
                        response += f"{i}. {result}\n"
                    response += f"\n{note_suffix}"
                    return response
        
        # No context or fallback case
        if model_available:
            return (f"I received your question: '{query}'\n\n"
                    f"I'm having trouble generating a good response right now. "
                    f"This might be due to the complexity of the question or model limitations.\n\n"
                    f"Try:\n"
                    f"• Rephrasing your question more simply\n"
                    f"• Being more specific about what you want to know\n"
                    f"• Uploading relevant documents for better context")
        else:
            return (f"I received your question: '{query}'\n\n"
                    f"Unfortunately, I cannot provide a detailed answer because:\n"
                    f"• The AI model failed to load (likely network connectivity issue)\n"
                    f"• This appears to be a connection problem with huggingface.co\n\n"
                    f"To resolve this:\n"
                    f"• Check your internet connection\n"
                    f"• Try again in a few minutes\n"
                    f"• Consider using a VPN if there are regional restrictions\n\n"
                    f"The app can still search your documents - try uploading PDFs and asking questions about them!")
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove common artifacts
        response = response.strip()
        
        # Stop at certain tokens that indicate end of response
        stop_tokens = ["\nUser:", "\nBot:", "Question:", "Context:", "Answer:", "<|endoftext|>"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]
        
        # Remove repetitive patterns (but be more lenient)
        lines = response.split('\n')
        if len(lines) > 1:
            unique_lines = []
            for line in lines:
                line = line.strip()
                if line and line not in unique_lines:
                    unique_lines.append(line)
            response = ' '.join(unique_lines)
        
        # Only remove if response is very short (reduced threshold)
        if len(response.strip()) < 3:
            return ""
        
        return response.strip()
    
    def is_available(self) -> bool:
        """Check if the model is available for use"""
        try:
            if not self.is_loaded:
                success = self._load_model()
                return success
            return self.is_loaded
        except Exception as e:
            print(f"Error checking model availability: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "cache_dir": self.cache_dir,
            "size_mb": self._get_model_size_mb() if self.is_loaded else 0
        }


class HuggingFaceEmbeddingModel:
    """
    Embedding model using Sentence Transformers with automatic downloading
    """
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.cache_dir = cache_dir or config.MODEL_CACHE_DIR
        self.model = None
        self.device = self._setup_device()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Embedding model: {self.model_name}")
    
    def _setup_device(self):
        """Setup computation device"""
        if config.USE_CUDA and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is not None:
            return
        
        try:
            print(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            
            # Load with explicit device=None to let the library handle device assignment
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=None,  # Let the library choose the best device
                trust_remote_code=True
            )
            
            print(f"Embedding model loaded successfully!")
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            raise e
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        if self.model is None:
            self._load_model()
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error encoding texts: {str(e)}")
            # Return dummy embeddings as fallback
            import numpy as np
            return np.random.rand(len(texts), 384).astype('float32')
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            self._load_model()
        
        # Test with sample text
        sample_embedding = self.encode(["sample text"])
        return sample_embedding.shape[1]
    
    def is_available(self) -> bool:
        """Check if embedding model is available"""
        try:
            if self.model is None:
                self._load_model()
            return self.model is not None
        except:
            return False
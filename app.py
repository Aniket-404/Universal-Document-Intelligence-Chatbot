import streamlit as st
import os
from typing import List, Dict
import time

# Import custom components
from components.document_processor import DocumentProcessor
from components.vector_store import VectorStore
from components.query_router import QueryRouter, QueryType
from components.web_search import WebSearcher
from components.huggingface_client import HuggingFaceClient

# Page configuration
st.set_page_config(
    page_title="Universal Document Intelligence Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_hf_client():
    """Get or create HuggingFace client with caching"""
    try:
        print("Initializing cached HuggingFace client...")
        client = HuggingFaceClient()
        # Force model loading
        success = client._load_model()
        print(f"Model loading success: {success}")
        print(f"Model is_loaded: {client.is_loaded}")
        return client, success
    except Exception as e:
        print(f"Failed to initialize HuggingFace client: {str(e)}")
        return None, False

class DocumentChatbot:
    """
    Main chatbot application class
    """
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.query_router = QueryRouter()
        self.web_searcher = None
        
        # Get cached HuggingFace client
        self.hf_client, self.model_loaded = get_hf_client()
        
        # Initialize web searcher if API key is available
        try:
            self.web_searcher = WebSearcher()
        except ValueError as e:
            st.warning(f"Web search disabled: {str(e)}")
        
        # Load existing index if available
        self.vector_store.load_index()
    
    def is_ai_model_available(self):
        """Check if AI model is available"""
        return self.hf_client is not None and self.hf_client.is_loaded
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files"""
        if not uploaded_files:
            return
        
        with st.spinner("Processing uploaded documents..."):
            all_chunks = []
            
            for uploaded_file in uploaded_files:
                try:
                    # Process the PDF
                    chunks = self.doc_processor.process_document(uploaded_file)
                    all_chunks.extend(chunks)
                    
                    st.success(f"Processed {uploaded_file.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            if all_chunks:
                # Add to vector store
                self.vector_store.add_documents(all_chunks)
                self.vector_store.save_index()
                
                st.success(f"Successfully processed {len(all_chunks)} document chunks!")
                
                # Update session state
                st.session_state.documents_loaded = True
                st.session_state.vector_stats = self.vector_store.get_stats()
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents using vector similarity"""
        if self.vector_store.index is None or len(self.vector_store.documents) == 0:
            print(f"No documents available - index: {self.vector_store.index is not None}, docs: {len(self.vector_store.documents) if hasattr(self.vector_store, 'documents') else 'N/A'}")
            return []
        
        results = self.vector_store.search(query, k=k)
        print(f"Document search for '{query}': found {len(results)} results")
        if results:
            scores = [r.get('score', 0) for r in results]
            print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
        return results
    
    def get_web_search_results(self, query: str) -> List[Dict]:
        """Get web search results"""
        if not self.web_searcher:
            return []
        
        try:
            return self.web_searcher.search_and_format(query, num_results=3)
        except Exception as e:
            st.error(f"Web search error: {str(e)}")
            return []
    
    def generate_response(self, query: str) -> Dict:
        """Generate response using smart routing and HuggingFace for LLM responses"""
        response = {
            'query': query,
            'sources': [],
            'answer': '',
            'routing_info': '',
            'search_strategy': 'unknown'
        }
        
        # Search documents first, but respect query routing
        doc_results = self.search_documents(query)
        
        # NEW: Use semantic-based routing instead of keyword-based
        routing_analysis = self.query_router.analyze_query_semantic(query, self.vector_store, similarity_threshold=0.15)
        
        print(f"DEBUG: Semantic routing result: {routing_analysis}")
        
        # SMART ROUTING: Use semantic similarity to determine strategy
        if routing_analysis['suggested_route'] == QueryType.WEB_SEARCH:
            # Query is not relevant to documents - use web search
            response['search_strategy'] = 'web_search'
            response['routing_info'] = f"Strategy: web_search (reason: {routing_analysis['reasoning'][0] if routing_analysis['reasoning'] else 'semantic analysis'})"
            print(f"DEBUG: Using web search for query: '{query}' (similarity: {routing_analysis.get('similarity_score', 0):.3f})")
            web_results = self.get_web_search_results(query)
            print(f"DEBUG: Web search returned {len(web_results) if web_results else 0} results")
            
            if web_results:
                # Create context from web results
                context = "Web search results:\n"
                for i, result in enumerate(web_results[:3], 1):
                    context += f"{i}. {result['title']}: {result['snippet']}\n"
                    response['sources'].append({
                        'type': 'web',
                        'title': result['title'],
                        'snippet': result['snippet'],
                        'link': result.get('link', ''),
                        'source': result.get('source', '')
                    })
                
                print(f"DEBUG: Web context created, length: {len(context)}")
                
                # Generate response using HuggingFace
                if self.is_ai_model_available():
                    system_prompt = "You are a helpful AI assistant that answers questions based on web search results. Be accurate and cite sources when appropriate."
                    ai_response = self.hf_client.generate_response(query, context, system_prompt)
                    
                    if len(ai_response.strip()) < 50 or "not sure" in ai_response.lower():
                        response['answer'] = f"**🌐 Web Search Results:**\n{context}\n\n**🤖 AI Analysis:**\n{ai_response}"
                    else:
                        response['answer'] = f"**🤖 AI Analysis:**\n{ai_response}\n\n**🌐 Web Search Results:**\n{context}"
                    response['ai_model_used'] = True
                else:
                    response['answer'] = f"**🌐 Web Search Results:**\n{context}"
                    response['ai_model_used'] = False
                
                print(f"DEBUG: Returning web search response")
                return response
            else:
                print("DEBUG: No web results, falling back to document search")
        
        # If semantic routing suggests documents, use them
        elif routing_analysis['suggested_route'] == QueryType.DOCUMENT_ONLY and doc_results and len(doc_results) > 0:
            best_score = max([r.get('score', 0) for r in doc_results])
            
            print(f"DEBUG: Using documents based on semantic routing: {len(doc_results)} results, best score: {best_score:.3f}")
            
            response['search_strategy'] = 'document_search'
            response['routing_info'] = f"Strategy: document_search (semantic similarity: {routing_analysis.get('similarity_score', 0):.3f}, found {len(doc_results)} matches)"
            
            # Create context from document results
            context = "Relevant information from your documents:\n"
            for i, result in enumerate(doc_results[:3], 1):
                doc = result['document']
                score = result['score']
                context += f"{i}. From {doc['metadata']['filename']} (relevance: {score:.2f}):\n{doc['text']}\n\n"
                
                response['sources'].append({
                    'type': 'document',
                    'filename': doc['metadata']['filename'],
                    'text': doc['text'],
                    'score': score,
                    'chunk_id': doc['metadata'].get('chunk_index', 0)
                })
                
            # Generate response using HuggingFace
            if self.is_ai_model_available():
                system_prompt = "You are a helpful AI assistant that answers questions based on provided document context. Be accurate and cite the source documents when appropriate."
                print(f"DEBUG: Generating AI response for query: '{query[:50]}...'")
                print(f"DEBUG: Context length: {len(context)}")
                ai_response = self.hf_client.generate_response(query, context, system_prompt)
                print(f"DEBUG: AI response received: '{ai_response[:100]}...'")
                print(f"DEBUG: AI response length: {len(ai_response.strip())}")
                
                # Always combine AI response with document context for better user experience
                if ai_response and len(ai_response.strip()) > 5:
                    response['answer'] = f"**🤖 AI Summary:**\n{ai_response}\n\n**📄 Source Documents:**\n{context}"
                    response['ai_model_used'] = True
                else:
                    # Fallback if AI response is empty
                    response['answer'] = f"**📄 Source Documents:**\n{context}"
                    response['ai_model_used'] = False
            else:
                print("DEBUG: AI model not available, using fallback")
                # Fallback response if HuggingFace is not available
                response['answer'] = f"**📄 Source Documents:**\n{context}"
                response['ai_model_used'] = False
            
            return response
        
        # Fallback: Use web search if no relevant documents found
        print("DEBUG: Using web search fallback")
        response['search_strategy'] = 'web_search'
        response['routing_info'] = f"Strategy: web_search (no relevant documents found or documents not relevant enough)"
        web_results = self.get_web_search_results(query)
        
        if web_results:
            # Create context from web results
            context = "Web search results:\n"
            for i, result in enumerate(web_results[:3], 1):
                context += f"{i}. {result['title']}: {result['snippet']}\n"
                response['sources'].append({
                    'type': 'web',
                    'title': result['title'],
                    'snippet': result['snippet'],
                    'link': result.get('link', ''),
                    'source': result.get('source', '')
                })
            
            # Generate response using HuggingFace
            if self.is_ai_model_available():
                system_prompt = "You are a helpful AI assistant. Answer the user's question based on the provided web search results. Be informative and cite your sources."
                ai_response = self.hf_client.generate_response(query, context, system_prompt)
                
                if len(ai_response.strip()) < 50 or "not sure" in ai_response.lower():
                    response['answer'] = f"**🌐 Web Search Results:**\n{context}\n\n**🤖 AI Analysis:**\n{ai_response}"
                else:
                    response['answer'] = f"**🤖 AI Analysis:**\n{ai_response}\n\n**🌐 Web Search Results:**\n{context}"
                response['ai_model_used'] = True
            else:
                response['answer'] = f"**🌐 Web Search Results:**\n{context}"
                response['ai_model_used'] = False
        else:
            response['answer'] = "I couldn't find relevant information in your documents or through web search. Please try rephrasing your question or upload more relevant documents."
        
        return response

def main():
    """Main application function"""
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DocumentChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    # Header
    st.title("Universal Document Intelligence Chatbot")
    st.markdown("*Upload documents and ask questions - get answers from your files or the web*")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to create a knowledge base"
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                st.session_state.chatbot.process_uploaded_files(uploaded_files)
        
        # Display statistics
        if st.session_state.documents_loaded:
            st.subheader("Knowledge Base Stats")
            stats = st.session_state.chatbot.vector_store.get_stats()
            st.metric("Documents", stats['total_documents'])
            st.metric("Vector Dimension", stats['dimension'])
            st.info(f"Model: {stats['model_name']}")
        
        # Clear documents
        if st.session_state.documents_loaded:
            if st.button("Clear All Documents", type="secondary"):
                st.session_state.chatbot.vector_store.clear_index()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.success("Documents cleared!")
                st.rerun()
        
        # Web search status
        st.subheader("AI Model Status")
        if st.session_state.chatbot.hf_client and st.session_state.chatbot.hf_client.is_available():
            st.success("✅ AI model loaded")
        else:
            st.error("❌ AI model not loaded")
            if st.button("🔄 Load AI Model", type="primary"):
                success = st.session_state.chatbot.initialize_ai_model()
                if success:
                    st.rerun()
        
        st.subheader("Web Search")
        if st.session_state.chatbot.web_searcher:
            st.success("Web search enabled")
        else:
            st.error("Web search disabled")
            st.info("Add SERPER_API_KEY to .env file to enable web search")
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['query'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            # Show routing info
            if chat.get('routing_info'):
                with st.expander("Search Strategy"):
                    st.info(chat['routing_info'])
            
            # Show sources
            if chat.get('sources'):
                with st.expander(f"Sources ({len(chat['sources'])} found)"):
                    for j, source in enumerate(chat['sources'], 1):
                        if source['type'] == 'document':
                            st.markdown(f"**{j}. Document Source:**")
                            st.markdown(f"- **File:** {source['filename']}")
                            st.markdown(f"- **Relevance:** {source['score']:.2f}")
                            st.markdown(f"- **Text:** {source['text'][:200]}...")
                        elif source['type'] == 'web':
                            st.markdown(f"**{j}. Web Source:**")
                            st.markdown(f"- **Title:** {source['title']}")
                            st.markdown(f"- **Source:** {source.get('source', 'Unknown')}")
                            if source.get('link'):
                                st.markdown(f"- **Link:** {source['link']}")
    
    # Query input
    query = st.chat_input("Ask a question about your documents or anything else...")
    
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(query)
            
            st.write(response['answer'])
            
            # Show routing info
            if response.get('routing_info'):
                with st.expander("Search Strategy"):
                    st.info(response['routing_info'])
                    st.caption(f"Strategy used: {response['search_strategy']}")
            
            # Show sources
            if response.get('sources'):
                with st.expander(f"Sources ({len(response['sources'])} found)"):
                    for j, source in enumerate(response['sources'], 1):
                        if source['type'] == 'document':
                            st.markdown(f"**{j}. Document Source:**")
                            st.markdown(f"- **File:** {source['filename']}")
                            st.markdown(f"- **Relevance:** {source['score']:.2f}")
                            st.markdown(f"- **Text:** {source['text'][:200]}...")
                        elif source['type'] == 'web':
                            st.markdown(f"**{j}. Web Source:**")
                            st.markdown(f"- **Title:** {source['title']}")
                            st.markdown(f"- **Source:** {source.get('source', 'Unknown')}")
                            if source.get('link'):
                                st.markdown(f"- **Link:** {source['link']}")
        
        # Add to chat history
        st.session_state.chat_history.append({
            'query': query,
            'answer': response['answer'],
            'routing_info': response.get('routing_info'),
            'sources': response.get('sources', []),
            'search_strategy': response.get('search_strategy')
        })
    
    # Instructions
    if not st.session_state.chat_history:
        st.markdown("""
        ### Getting Started:
        
        1. **Upload PDFs** - Use the sidebar to add your documents
        2. **Click Process** - This creates a searchable knowledge base
        3. **Start Chatting** - Ask questions in the box below
        
        ### What you can ask:
        
        **About your documents:**
        - "What does the report say about..."
        - "Summarize the main points"
        - "Find information about X"
        
        **General questions:**
        - "What's the latest news on..."
        - "How does X work?"
        - "Compare A and B"
        
        The chatbot automatically decides whether to search your documents or the web.
        """)

if __name__ == "__main__":
    main()
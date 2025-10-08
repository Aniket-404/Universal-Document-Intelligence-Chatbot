import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

class QueryType(Enum):
    DOCUMENT_ONLY = "document_only"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"

class QueryRouter:
    """
    Smart query routing logic to determine whether to use document search,
    web search, or both based on query characteristics
    """
    
    def __init__(self):
        # Keywords that trigger web search
        self.web_search_keywords = {
            'temporal': [
                'latest', 'recent', 'current', 'now', 'today', 'this year', 
                '2024', '2025', 'new', 'updated', 'modern', 'contemporary'
            ],
            'explanatory': [
                'explain', 'how does', 'how to', 'what is', 'what are',
                'why does', 'why is', 'tell me about', 'describe'
            ],
            'comparative': [
                'vs', 'versus', 'compare', 'comparison', 'difference between',
                'alternatives to', 'better than', 'similar to', 'like'
            ],
            'current_data': [
                'price', 'cost', 'stock', 'trend', 'trending', 'popular',
                'market', 'value', 'rate', 'statistics', 'data'
            ],
            'specifications': [
                'specs', 'specifications', 'features', 'details', 'technical',
                'performance', 'benchmark', 'review'
            ],
            'superlatives': [
                'slowest', 'biggest', 'smallest', 'best', 'worst',
                'most', 'least', 'highest', 'lowest', 'top', 'bottom',
                'largest', 'tallest', 'strongest', 'weakest'
            ],
            'factual_queries': [
                'world record', 'world', 'global', 'worldwide', 'international',
                'country', 'countries', 'nation', 'capital', 'population'
            ]
        }
        
        # Keywords that strongly suggest document search
        self.document_keywords = [
            'according to', 'in the document', 'from the file', 'mentioned',
            'stated', 'written', 'document says', 'file contains',
            'pdf', 'pdf about', 'this pdf', 'document about', 'file about',
            'resume', 'cv', 'uploaded', 'this document', 'this file'
        ]
        
        # General knowledge keywords that might need web search
        self.general_knowledge_keywords = [
            'definition', 'meaning', 'concept', 'theory', 'principle',
            'history', 'background', 'overview', 'introduction'
        ]
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine routing strategy
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with routing analysis
        """
        query_lower = query.lower()
        
        # Initialize analysis
        analysis = {
            'query': query,
            'web_indicators': [],
            'document_indicators': [],
            'confidence_scores': {
                'web_search': 0.0,
                'document_search': 0.0
            },
            'suggested_route': QueryType.DOCUMENT_ONLY,
            'reasoning': []
        }
        
        # Check for web search indicators
        web_score = 0
        for category, keywords in self.web_search_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    analysis['web_indicators'].append(f"{keyword} ({category})")
                    web_score += self._get_keyword_weight(category)
        
        # Check for document indicators
        doc_score = 0
        for keyword in self.document_keywords:
            if keyword in query_lower:
                analysis['document_indicators'].append(keyword)
                doc_score += 2.0  # High weight for explicit document references
        
        # Check for general knowledge that might need web search
        for keyword in self.general_knowledge_keywords:
            if keyword in query_lower:
                analysis['web_indicators'].append(f"{keyword} (general_knowledge)")
                web_score += 0.5
        
        # Question word analysis
        question_words = ['how', 'what', 'why', 'when', 'where', 'who', 'which']
        question_count = sum(1 for word in question_words if word in query_lower.split())
        if question_count > 0:
            web_score += 0.3 * question_count
        
        # Length analysis (longer queries often need more context)
        if len(query.split()) > 10:
            web_score += 0.2
        
        # Normalize scores
        max_possible_score = 10.0
        analysis['confidence_scores']['web_search'] = min(web_score / max_possible_score, 1.0)
        analysis['confidence_scores']['document_search'] = min(doc_score / max_possible_score, 1.0)
        
        # If no explicit document indicators, boost document search slightly
        if doc_score == 0:
            analysis['confidence_scores']['document_search'] = 0.3
        
        # Determine routing strategy
        web_confidence = analysis['confidence_scores']['web_search']
        doc_confidence = analysis['confidence_scores']['document_search']
        
        if doc_confidence > 0.7:  # Strong document indicators
            analysis['suggested_route'] = QueryType.DOCUMENT_ONLY
            analysis['reasoning'].append("Strong document reference indicators")
        elif web_confidence > 0.35:  # Even lower threshold for web search
            analysis['suggested_route'] = QueryType.WEB_SEARCH
            analysis['reasoning'].append("Web search indicators detected")
        elif web_confidence > 0.25 and doc_confidence > 0.3:  # Mixed signals
            analysis['suggested_route'] = QueryType.HYBRID
            analysis['reasoning'].append("Mixed indicators suggest hybrid approach")
        else:  # Default to document search when documents are available
            analysis['suggested_route'] = QueryType.DOCUMENT_ONLY
            analysis['reasoning'].append("Default to document search - prefer uploaded documents")
        
        return analysis
    
    def _get_keyword_weight(self, category: str) -> float:
        """Get weight for different keyword categories"""
        weights = {
            'temporal': 1.5,        # Strong indicator for web search
            'explanatory': 0.8,     # Medium indicator
            'comparative': 1.2,     # Strong indicator  
            'current_data': 1.5,    # Strong indicator
            'specifications': 1.0,  # Medium indicator
            'superlatives': 1.8,    # Very strong indicator for web search
            'factual_queries': 1.6  # Strong indicator for web search
        }
        return weights.get(category, 0.5)
    
    def should_use_web_search(self, query: str, document_results: List = None) -> Tuple[bool, str]:
        """
        Determine if web search should be used based on query and document results
        
        Args:
            query: User query
            document_results: Results from document search (if any)
            
        Returns:
            Tuple of (should_use_web, reasoning)
        """
        analysis = self.analyze_query(query)
        
        # Always use web search if suggested route is WEB_SEARCH
        if analysis['suggested_route'] == QueryType.WEB_SEARCH:
            return True, "Query indicates need for web search"
        
        # For hybrid queries, be more conservative - prefer documents when available
        if analysis['suggested_route'] == QueryType.HYBRID:
            if not document_results or len(document_results) == 0:
                return True, "Hybrid query with no document results"
            elif len(document_results) > 0:
                # Check quality of document results - lowered threshold to prefer documents
                best_score = max([r.get('score', 0) for r in document_results])
                if best_score < 0.05:  # Very low similarity scores only
                    return True, "Hybrid query with very low-quality document results"
        
        # For document-only queries, almost never use web search
        if analysis['suggested_route'] == QueryType.DOCUMENT_ONLY:
            # Only use web search if absolutely no document results
            if document_results is not None and len(document_results) == 0:
                return True, "No document results found, falling back to web search"
        
        return False, "Document search should be sufficient"
    
    def get_routing_explanation(self, query: str) -> str:
        """
        Get human-readable explanation of routing decision
        
        Args:
            query: User query
            
        Returns:
            Explanation string
        """
        analysis = self.analyze_query(query)
        
        explanation = f"**Query Analysis for:** {query}\n\n"
        
        if analysis['web_indicators']:
            explanation += "**Web Search Indicators Found:**\n"
            for indicator in analysis['web_indicators'][:3]:  # Show top 3
                explanation += f"- {indicator}\n"
            explanation += "\n"
        
        if analysis['document_indicators']:
            explanation += "**Document Search Indicators Found:**\n"
            for indicator in analysis['document_indicators']:
                explanation += f"- {indicator}\n"
            explanation += "\n"
        
        explanation += f"**Suggested Strategy:** {analysis['suggested_route'].value}\n\n"
        
        if analysis['reasoning']:
            explanation += "**Reasoning:** " + ", ".join(analysis['reasoning'])
        
        return explanation
    
    def analyze_query_semantic(self, query: str, vector_store=None, similarity_threshold: float = 0.15) -> Dict:
        """
        Semantic-based query routing using embedding similarity to determine
        if the query is relevant to indexed documents
        
        Args:
            query: User's input query
            vector_store: VectorStore instance with indexed documents
            similarity_threshold: Minimum similarity score to prefer documents (0.0-1.0)
            
        Returns:
            Dict with routing decision and reasoning
        """
        try:
            # If no vector store or no documents, default to web search
            if not vector_store or not hasattr(vector_store, 'search') or len(getattr(vector_store, 'documents', [])) == 0:
                return {
                    'suggested_route': QueryType.WEB_SEARCH,
                    'reasoning': ['No documents available - using web search'],
                    'similarity_score': 0.0
                }
            
            # Still check for strong temporal indicators that should always use web search
            temporal_keywords = ['latest', 'recent', 'current', 'now', 'today', 'this year', '2024', '2025', 'breaking', 'news']
            query_lower = query.lower()
            
            for keyword in temporal_keywords:
                if keyword in query_lower:
                    return {
                        'suggested_route': QueryType.WEB_SEARCH,
                        'reasoning': [f'Temporal keyword "{keyword}" detected - using web search for current information'],
                        'similarity_score': 0.0
                    }
            
            # Get semantic similarity with documents
            try:
                # Search for similar documents
                results = vector_store.search(query, k=3)
                
                if not results:
                    return {
                        'suggested_route': QueryType.WEB_SEARCH,
                        'reasoning': ['No document matches found - using web search'],
                        'similarity_score': 0.0
                    }
                
                # Get the best similarity score
                best_score = max([r.get('score', 0) for r in results])
                
                print(f"DEBUG: Semantic routing - Query: '{query[:50]}...', Best similarity: {best_score:.3f}, Threshold: {similarity_threshold}")
                
                if best_score >= similarity_threshold:
                    return {
                        'suggested_route': QueryType.DOCUMENT_ONLY,
                        'reasoning': [f'High document relevance (score: {best_score:.3f}) - using document search'],
                        'similarity_score': best_score
                    }
                else:
                    return {
                        'suggested_route': QueryType.WEB_SEARCH,
                        'reasoning': [f'Low document relevance (score: {best_score:.3f}) - using web search'],
                        'similarity_score': best_score
                    }
                    
            except Exception as search_error:
                print(f"DEBUG: Semantic search failed: {search_error}")
                return {
                    'suggested_route': QueryType.WEB_SEARCH,
                    'reasoning': ['Document search failed - using web search'],
                    'similarity_score': 0.0
                }
                
        except Exception as e:
            print(f"DEBUG: Semantic routing error: {e}")
            # Fallback to keyword-based routing
            return self.analyze_query(query)
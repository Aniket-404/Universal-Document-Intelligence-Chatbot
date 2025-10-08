import requests
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebSearcher:
    """
    Serper.dev API integration for web search functionality
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            raise ValueError("Serper API key is required. Please set SERPER_API_KEY in your .env file")
    
    def search(self, query: str, num_results: int = 5) -> Dict:
        """
        Perform web search using Serper API
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results,
            'page': 1
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Web search failed: {str(e)}")
    
    def format_search_results(self, search_response: Dict) -> List[Dict]:
        """
        Format search results into a standardized structure
        
        Args:
            search_response: Raw response from Serper API
            
        Returns:
            List of formatted search results
        """
        formatted_results = []
        
        # Process organic results
        organic_results = search_response.get('organic', [])
        
        for i, result in enumerate(organic_results):
            formatted_result = {
                'rank': i + 1,
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', ''),
                'source': result.get('displayLink', ''),
                'type': 'organic'
            }
            formatted_results.append(formatted_result)
        
        # Process answer box if available
        answer_box = search_response.get('answerBox')
        if answer_box:
            formatted_result = {
                'rank': 0,  # Answer box gets top priority
                'title': answer_box.get('title', 'Direct Answer'),
                'snippet': answer_box.get('answer', answer_box.get('snippet', '')),
                'link': answer_box.get('link', ''),
                'source': answer_box.get('displayLink', 'Google'),
                'type': 'answer_box'
            }
            formatted_results.insert(0, formatted_result)
        
        # Process knowledge graph if available
        knowledge_graph = search_response.get('knowledgeGraph')
        if knowledge_graph:
            formatted_result = {
                'rank': 0,
                'title': knowledge_graph.get('title', 'Knowledge Graph'),
                'snippet': knowledge_graph.get('description', ''),
                'link': knowledge_graph.get('descriptionLink', ''),
                'source': knowledge_graph.get('source', 'Google Knowledge Graph'),
                'type': 'knowledge_graph'
            }
            formatted_results.insert(0 if not answer_box else 1, formatted_result)
        
        return formatted_results
    
    def search_and_format(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform search and return formatted results
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of formatted search results
        """
        try:
            # Perform search
            search_response = self.search(query, num_results)
            
            # Format results
            formatted_results = self.format_search_results(search_response)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in web search: {str(e)}")
            return []
    
    def create_search_summary(self, results: List[Dict], max_length: int = 1000) -> str:
        """
        Create a summary from search results
        
        Args:
            results: List of search results
            max_length: Maximum length of summary
            
        Returns:
            Summary text with sources
        """
        if not results:
            return "No web search results found."
        
        summary_parts = []
        sources = []
        current_length = 0
        
        for result in results[:3]:  # Use top 3 results for summary
            snippet = result.get('snippet', '')
            title = result.get('title', '')
            source = result.get('source', '')
            link = result.get('link', '')
            
            if snippet and current_length + len(snippet) < max_length:
                summary_parts.append(f"**{title}**: {snippet}")
                if source and link:
                    sources.append(f"- [{source}]({link})")
                current_length += len(snippet) + len(title) + 4
        
        # Combine summary parts
        summary = "\n\n".join(summary_parts)
        
        if sources:
            summary += "\n\n**Sources:**\n" + "\n".join(sources)
        
        return summary
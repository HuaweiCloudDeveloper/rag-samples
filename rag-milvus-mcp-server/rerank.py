from abc import ABC, abstractmethod
from typing import Dict,Any,List
from http_json_client import HTTPJSONClient

class RerankClient(ABC,HTTPJSONClient):

    @abstractmethod
    def documents_rerank(self, query:str, documents: List[str], min_score:float) -> List[Dict[str, Any]]:
        pass

class OpenAIRerankClient(RerankClient):

    def __init__(self,base_url: str):
        super().__init__(base_url)
        self.endpoint = "/v1/rerank"

    def documents_rerank(self, query:str, documents: List[str]) -> List[Dict[str, Any]]:
        
        result = self.post(self.endpoint,data= {"query": query, "documents": documents})
        return result['results']
    
"""
{
    'model': 'gpt-3.5-turbo', 
    'object': 'list', 
    'usage': {'prompt_tokens': 246, 'total_tokens': 246}, 
    'results': [
                {'index': 0, 'relevance_score': -3.395951271057129}, 
                {'index': 1, 'relevance_score': -1.8673471212387085}, 
                {'index': 2, 'relevance_score': 8.277612686157227}, 
                {'index': 3, 'relevance_score': -11.033196449279785}
    ]
}
"""
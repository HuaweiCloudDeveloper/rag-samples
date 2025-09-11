from abc import ABC, abstractmethod
from typing import Union,List
from http_json_client import HTTPJSONClient

class EmbeddingClient(ABC,HTTPJSONClient):

    @abstractmethod
    def text_embedding(self, text: Union[str,List[str]]) -> Union[List,List[List]]:
        pass

class OpenAIEmbeddingClient(EmbeddingClient):

    def __init__(self,base_url: str):
        super().__init__(base_url)
        self.endpoint = "/v1/embeddings"

    def text_embedding(self, text: Union[str,List[str]]) -> Union[List,List[List]]:
        result = self.post(self.endpoint,data= {"input":text})
        data = result['data']
        if isinstance(text,List):
            return [dd['embedding'] for dd in data]
        return data[0]['embedding']
from typing import Union
from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]

class Usage(BaseModel):
    total_tokens: int

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[Union[float, int]]

class EmbeddingResponse(BaseModel):
    model: str
    object: str = "list"
    data: list[EmbeddingData]
    usage: Usage
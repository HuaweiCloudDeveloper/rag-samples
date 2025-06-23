from fastapi import APIRouter, Request
from typing import List,Union
from pydantic import BaseModel
from model_service import EmbeddingService
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_router = APIRouter(tags=['Embedding'])

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class Usage(BaseModel):
    total_tokens: int | str

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[Union[float, int]]
    index: int

class EmbeddingResponse(BaseModel):
    model: str
    object: str = "list"
    data: List[EmbeddingData]
    usage: Usage

@embedding_router.post("/embeddings", response_model=EmbeddingResponse, summary="文本向量化")
def text_embedding(request: Request,embed_request:EmbeddingRequest) -> EmbeddingResponse:    
    texts = embed_request.input
    logger.info("请求句子=>%s",texts)
    if isinstance(embed_request.input,str):
        texts = [embed_request.input]

    start_time = time.perf_counter()
    engine:EmbeddingService = request.app.state.embed_engine
    results = engine.execute(texts)
    embeddings = results[0]
    total_tokens = results[1]

    usage = Usage(prompt_tokens=len(embed_request.input),total_tokens=total_tokens)

    embedding_data = [
        EmbeddingData(
            object = "embedding",
            embedding = emb,
            index = count,
        )
        for count, emb in enumerate(embeddings)
    ]

    logger.info("处理耗时 %.4f ms",(time.perf_counter() - start_time) * 1000)
    return EmbeddingResponse(
        model=embed_request.model,
        object="list",
        data=embedding_data,
        usage=usage
    )

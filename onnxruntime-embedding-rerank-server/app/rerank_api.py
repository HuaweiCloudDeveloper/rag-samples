from fastapi import APIRouter, Request
from typing import List
from pydantic import BaseModel,Field
from uuid import uuid4
from log_config import logger
from model_service import RerankService
import time

rerank_router = APIRouter(tags=['Rerank'])

class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    return_documents: bool
    documents: List[str]

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str

class RerankResponse(BaseModel):
    id: str
    results: List[RerankResult]

class Usage(BaseModel):
    total_tokens: int | str

@rerank_router.post("/rerank", response_model=RerankResponse, summary="文本重排序")
def text_rerank(request: Request,rerank_request:RerankRequest) -> RerankResponse:  
    logger.info("请求内容=>\n%s",rerank_request) 
    query = rerank_request.query 
    documents = rerank_request.documents 
    top_n = rerank_request.top_n 
    return_documents = rerank_request.return_documents 

    start_time = time.perf_counter()
    engine:RerankService = request.app.state.rerank_engine
    results = engine.execute(query,documents,top_n,return_documents)
    
    # usage = Usage(prompt_tokens=len(documents),total_tokens=total_tokens)
    logger.info("请求结果:\n%s\n耗时 %.4f ms",results,(time.perf_counter() - start_time) * 1000)
    rerankResult = [
        RerankResult(
            index = result['id'],
            relevance_score = result['score'],
            document = result['document'],
        )
        for result in results
    ]
    return RerankResponse(
        id = f'{uuid4()}',
        results = rerankResult
    )

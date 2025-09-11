import os,uvicorn
from pydantic import BaseModel
from fastapi import FastAPI,Request
from fastapi.concurrency import asynccontextmanager
from rerank_provider import RerankProvider

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.rerank_engine = RerankProvider()
    yield
    app.state.rerank_engine.destroy()

app = FastAPI(
        title="Rerank Inference Server",
        lifespan=lifespan,
        root_path="/v1",
        version="1.0",
        license_info={"name":"Apache License 2.0","identifier":"Apache"}
    )

class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    return_documents: bool
    documents: list[str]

class RerankData(BaseModel):
    index: int
    relevance_score: float
    document: str

class RerankResponse(BaseModel):
    model: str
    results: list[RerankData]
 
@app.post("/rerank", response_model=RerankResponse, summary="文本重排序")
def text_rerank(request: Request,rerank_request:RerankRequest) -> RerankResponse:  
    query = rerank_request.query 
    documents = rerank_request.documents 
    top_n = rerank_request.top_n 
    return_documents = rerank_request.return_documents 

    engine:RerankProvider = request.app.state.rerank_engine
    results = engine.rerank(query,documents,top_n,return_documents)
    
    rerank_datas = [
        RerankData(
            index = result['id'],
            relevance_score = result['score'],
            document = result['document'],
        )
        for result in results
    ]
    return RerankResponse(
        model = rerank_request.model,
        results = rerank_datas
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
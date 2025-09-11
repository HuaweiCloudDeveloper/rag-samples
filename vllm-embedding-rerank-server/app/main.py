import uvicorn
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI,Request
from fastapi.concurrency import asynccontextmanager
from embedding_provider import EmbeddingProvider

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.embed_engine = EmbeddingProvider()
    yield
    app.state.embed_engine.destroy()

app = FastAPI(
        title="Embedding Inference Server",
        lifespan=lifespan,
        root_path="/v1",
        version="1.0",
        license_info={"name":"Apache License 2.0","identifier":"Apache"}
    )

class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]

class Usage(BaseModel):
    total_tokens: int

class EmbeddingData(BaseModel):
    embedding: list[Union[float, int]]

class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]
    usage: Usage

@app.post("/embeddings", response_model=EmbeddingResponse, summary="文本向量化")
def text_embedding(request: Request,embed_request:EmbeddingRequest) -> EmbeddingResponse:    
    texts = embed_request.input

    engine:EmbeddingProvider = request.app.state.embed_engine
    results = engine.embedding(documents=texts)
    embeddings = results[0]
    total_tokens = results[1]

    usage = Usage(total_tokens=total_tokens)

    embedding_datas = [
        EmbeddingData(
            embedding = emb,
            index = count,
        )
        for count, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        model=embed_request.model,
        data=embedding_datas,
        usage=usage
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8097)
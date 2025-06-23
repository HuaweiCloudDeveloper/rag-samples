from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from model_service import EmbeddingService,RerankService
from embedding_api import embedding_router
from rerank_api import rerank_router
from download_model import download_embedding_models,download_rerank_models
import logging
import os
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH",download_embedding_models())
RERANK_MODEL_PATH = os.getenv("LOCAL_RERANK_MODEL_PATH",download_rerank_models())

logger.info(f"EMBEDDING_MODEL_PATH:{EMBEDDING_MODEL_PATH}")
logger.info(f"RERANK_MODEL_PATH:{RERANK_MODEL_PATH}")

@asynccontextmanager
async def lifespan(app:FastAPI):
    """生命周期"""
    app.state.embed_engine = EmbeddingService(EMBEDDING_MODEL_PATH, 8)
    app.state.rerank_engine = RerankService(RERANK_MODEL_PATH, 8)
    logger.info("应用初始化成功")
    yield
    logger.info("应用正常停止")

# 创建应用
app = FastAPI(
        title="Embedding and Reranker Inference Server",
        lifespan=lifespan,
        root_path="/v1",
        version="1.0",
        license_info={"name":"Apache License 2.0","identifier":"Apache"}
    )

app.include_router(embedding_router)
app.include_router(rerank_router)

if __name__ == "__main__":
    # 启动应用
    uvicorn.run(app, host="0.0.0.0", port=9997)
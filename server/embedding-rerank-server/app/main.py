from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from model_service import EmbeddingService,RerankService
from embedding_api import embedding_router
from rerank_api import rerank_router
from modelscope import snapshot_download,model_file_download
from log_config import logger,LOGGING_CONFIG
import os
import uvicorn

# 当前文件的绝对路径
filepath = os.path.abspath(__file__)

def download_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(filepath)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # embedding model
    embedding_dir = os.path.join(models_dir,"bce_embedding_base")
    if not os.path.exists(embedding_dir):
        snapshot_download(model_id='netease-youdao/bce-embedding-base_v1',local_dir = embedding_dir)
        model_file_download(model_id='maidalun/bce-embedding-base_v1',file_path='tokenizer_config.json',local_dir=embedding_dir)
        model_file_download(model_id='maidalun/bce-embedding-base_v1',file_path='tokenizer.json',local_dir=embedding_dir)
    
    # reranker 
    reranker_dir = os.path.join(models_dir,"bce_reranker_base")
    if not os.path.exists(reranker_dir):
        snapshot_download(model_id='netease-youdao/bce-reranker-base_v1',local_dir = reranker_dir)
        model_file_download(model_id='maidalun/bce-reranker-base_v1',file_path='tokenizer_config.json',local_dir=reranker_dir)
        model_file_download(model_id='maidalun/bce-reranker-base_v1',file_path='tokenizer.json',local_dir=reranker_dir)
    return embedding_dir,reranker_dir

def create_app(embedding_dir,reranker_dir) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app:FastAPI):
        """生命周期"""
        app.state.embed_engine = EmbeddingService(embedding_dir, 8)
        app.state.rerank_engine = RerankService(reranker_dir, 8)
        logger.info("应用初始化成功")
        yield
        logger.info("应用正常停止")

    # 创建应用
    app = FastAPI(
            title="Embedding Inference Server",
            lifespan=lifespan,
            root_path="/v1",
            version="1.0",
            license_info={"name":"Apache License 2.0","identifier":"Apache"}
        )

    app.include_router(embedding_router)
    app.include_router(rerank_router)
    return app

if __name__ == "__main__":
    # 下载模型
    embedding_dir,reranker_dir = download_models()
    # 创建应用
    app = create_app(embedding_dir,reranker_dir)
    # 启动应用
    uvicorn.run(app, host="0.0.0.0", port=9997,log_config=LOGGING_CONFIG)
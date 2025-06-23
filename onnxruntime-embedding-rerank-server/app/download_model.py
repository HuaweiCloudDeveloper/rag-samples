import os
from modelscope import snapshot_download

# 当前文件的绝对路径
filepath = os.path.abspath(__file__)
models_dir = os.path.join(os.path.dirname(os.path.dirname(filepath)), "models")
os.makedirs(models_dir, exist_ok=True)

def download_embedding_models():
    embedding_dir = os.path.join(models_dir,"bce-embedding-base_v1")
    if not os.path.exists(embedding_dir):
        snapshot_download(model_id='netease-youdao/bce-embedding-base_v1',local_dir = embedding_dir)
    return embedding_dir

def download_rerank_models():
    reranker_dir = os.path.join(models_dir,"bce-reranker-base_v1")
    if not os.path.exists(reranker_dir):
        snapshot_download(model_id='netease-youdao/bce-reranker-base_v1',local_dir = reranker_dir)
    return reranker_dir
# llama.cpp 部署 Embedding和Rerank模型

## 下载模型

`Embedding`: bge-m3

`Rerank`: bge-reranker-v2-m3

```shell
cd /home
mkdir models
uv venv
uv tool install modescope

modelscope download --model gpustack/bge-m3-GGUF bge-m3-Q8_0.gguf --local_dir ./bge-m3-GGUF

modescope modelscope download --model gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-Q8_0.gguf --local_dir ./bge-reranker-v2-m3-GGUF

```

## 部署模型

```shell
docker compose up -d
```

Embedding和Rerank模型服务支持标准OpenAI接口规范 
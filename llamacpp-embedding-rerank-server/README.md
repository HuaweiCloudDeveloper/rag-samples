# llama.cpp 部署 Embedding和Rerank模型

## ARM架构下编译llama.cpp

1. 下载llama.cpp源码
```shell
cd /home
git clone https://github.com/ggml-org/llama.cpp.git
```

2. 拉取``ubuntu:22.04``镜像

```shell
docker pull ubuntu:22.04
docker run -itd  -v /home/llama.cpp:/home/llama.cpp ubuntu:22.04 --name ubuntu_llamacpp

docker exec -it ubuntu_llamacpp /bin/bash
```

3. 容器内编译

```shell
apt-get update
apt-get install -y git cmake libcurl4-openssl-dev build-essential
cmake -B build
cmake --build build --config Release -j 8
```

4. 宿主机上构建llama.cpp镜像

经过在上面容器内编译的步骤，``/home/llama.cpp``下新增``build``文件夹，里面就是基于``ubuntu:22.04``镜像环境编译的llama.cpp的库文件。下面就可根据这些文件再基于``ubuntu:22.04``构建``llama.cpp-arm64:server``镜像。

```shell
cd /home/llama.cpp
mkdir -p docker/lib
cp -r ../build/bin/*.so ./docker/lib/
cp ../build/bin/llama-server ./docker/llama-server
vi Dockerfile
```

``Dockerfile``内容

```shell
FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY lib/ /app
COPY llama-server /app

ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /app

ENTRYPOINT ["/app/llama-server"]
```

构建镜像

```shell
docker build -t llama.cpp-arm64:server .
```

## 下载模型

`Embedding`: bge-m3

`Rerank`: bge-reranker-v2-m3

```shell
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip install modelscope
cd /home
mkdir models
cd models

modelscope download --model gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-Q8_0.gguf --local_dir ./bge-reranker-v2-m3-GGUF
modelscope download --model gpustack/bge-m3-GGUF bge-m3-Q8_0.gguf --local_dir ./bge-m3-GGUF
```

## 编制docker-compose文件

```shell
services:
  llamacpp-embedding-server:
    image: llama.cpp-arm64:server
    container_name: llamacpp-embedding-server
    command: --embedding --pooling mean --verbose-prompt
    restart: always
    ports:
      - 8081:8080
    volumes:
      - /home/models/bge-m3-GGUF:/models
    environment:
      LLAMA_ARG_MODEL: /models/bge-m3-Q8_0.gguf
      LLAMA_ARG_CTX_SIZE: 8192
      LLAMA_ARG_N_PARALLEL: 8
      LLAMA_ARG_PORT: 8080
      LLAMA_ARG_UBATCH: 8192
      LLAMA_ARG_N_GPU_LAYERS_DRAFT: 0

  llamacpp-rerank-server:
    image: llama.cpp-arm64:server
    container_name: llamacpp-rerank-server
    command: --reranking --pooling rank
    restart: always
    ports:
      - 8082:8080
    volumes:
      - /home/models/bge-reranker-v2-m3-GGUF:/models
    environment:
      LLAMA_ARG_MODEL: /models/bge-reranker-v2-m3-Q8_0.gguf
      LLAMA_ARG_CTX_SIZE: 8192
      LLAMA_ARG_N_PARALLEL: 8
      LLAMA_ARG_PORT: 8080
      LLAMA_ARG_BATCH: 8192
      LLAMA_ARG_UBATCH: 8192
      LLAMA_ARG_FLASH_ATTN: enable
      LLAMA_ARG_N_GPU_LAYERS_DRAFT: 0
```

## 部署模型

```shell
docker compose up -d
```

Embedding和Rerank模型服务支持标准OpenAI接口规范 
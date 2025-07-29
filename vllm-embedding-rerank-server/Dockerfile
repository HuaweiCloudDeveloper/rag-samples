FROM quay.io/ascend/vllm-ascend:v0.9.2rc1

RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple \
    && pip install fastapi uvicorn

WORKDIR /workspace
RUN mkdir -p /root/.cache/modelscope/hub/models/Qwen

COPY ./app .
COPY /root/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-8B /root/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-8B

CMD ["python","main.py"]

# docker build -t vllm-qwen3-rerank-server .
FROM python:3.10-slim

LABEL maintainer="003AAC"
LABEL description="联想式知识图谱增强问答系统 (GraphRAG)"
LABEL version="0.3.0"

WORKDIR /app

# 运行时系统库（numpy/openblas依赖libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖（在线模式，不含llama-cpp-python）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-docker.txt

# 拷贝项目代码
COPY . .

# 预下载Embedding模型到镜像
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python3 -c "from embedding_matcher import EmbeddingMatcher; \
    concepts=['test']; descs=['test']; \
    m = EmbeddingMatcher(concepts, descs); \
    m.ensure_loaded(); \
    print('Embedding模型预下载完成')" || echo "WARNING: Embedding预下载失败，将在运行时重试"

# 运行时环境变量
ENV RUN_MODE=auto \
    ONLINE_BASE_URL=https://api.deepseek.com \
    ONLINE_MODEL=deepseek-chat \
    EMBEDDING_MODE=auto \
    EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
    EMBEDDING_DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_ENDPOINT=https://hf-mirror.com

# HF缓存目录
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN mkdir -p ${TRANSFORMERS_CACHE}

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

# 非root用户运行
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python3", "web_app.py"]

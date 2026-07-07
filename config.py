"""
联想式知识图谱增强问答系统 - 配置文件
支持离线(ChatGLM3) / 在线(DeepSeek) 双模式
支持关键词 / Embedding语义匹配
支持环境变量覆盖（Docker友好）
"""
import os


def _env(key, default, cast=str):
    """从环境变量读取配置，支持类型转换"""
    val = os.environ.get(key)
    if val is None:
        return default
    if cast is bool:
        return val.lower() in ("1", "true", "yes")
    return cast(val)


# ===== 模型配置 =====
# 离线模式：本地ChatGLM3
OFFLINE_MODEL_PATH = _env("OFFLINE_MODEL_PATH", "")  # GGUF模型路径，Docker中挂载到 /app/models/
OFFLINE_N_CTX = _env("OFFLINE_N_CTX", 2048, int)
OFFLINE_N_GPU_LAYERS = _env("OFFLINE_N_GPU_LAYERS", 0, int)

# 在线模式：DeepSeek API
ONLINE_PROVIDER = _env("ONLINE_PROVIDER", "deepseek")
ONLINE_API_KEY = _env("ONLINE_API_KEY", "")            # 留空则走离线模式；也可通过Web界面输入
ONLINE_BASE_URL = _env("ONLINE_BASE_URL", "https://api.deepseek.com")
ONLINE_MODEL = _env("ONLINE_MODEL", "deepseek-chat")

# 当前运行模式：auto(有key走在线，否则离线) / offline / online
RUN_MODE = _env("RUN_MODE", "auto")

# ===== Embedding配置 =====
# keyword: 关键词匹配（无需额外依赖）
# embedding: sentence-transformers语义匹配（需安装依赖）
EMBEDDING_MODE = _env("EMBEDDING_MODE", "auto")  # auto / keyword / embedding

# Embedding模型（仅EMBEDDING_MODE=embedding/auto时生效）
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE = _env("EMBEDDING_DEVICE", "cpu")  # cpu / cuda
EMBEDDING_SIMILARITY_THRESHOLD = _env("EMBEDDING_SIMILARITY_THRESHOLD", 0.35, float)

# ===== 图谱配置 =====
GRAPH_WALK_DEPTH = _env("GRAPH_WALK_DEPTH", 2, int)
GRAPH_TOP_K_NEIGHBORS = _env("GRAPH_TOP_K_NEIGHBORS", 6, int)
SEMANTIC_FILTER_THRESHOLD = _env("SEMANTIC_FILTER_THRESHOLD", 0.3, float)
CROSS_DOMAIN_WEIGHT_BOOST = _env("CROSS_DOMAIN_WEIGHT_BOOST", 1.5, float)

# ===== 话题推荐配置 =====
TOPIC_MAX_CROSS_DOMAIN = _env("TOPIC_MAX_CROSS_DOMAIN", 3, int)
TOPIC_MAX_SAME_DOMAIN = _env("TOPIC_MAX_SAME_DOMAIN", 1, int)

# ===== 服务配置 =====
HOST = _env("HOST", "0.0.0.0")
PORT = _env("PORT", 8000, int)

# ===== 图谱发现配置 =====
GRAPH_DISCOVER_ENABLED = _env("GRAPH_DISCOVER_ENABLED", True, bool)  # 是否自动发现新关系
GRAPH_DISCOVER_THRESHOLD = _env("GRAPH_DISCOVER_THRESHOLD", 0.7, float)  # 置信度阈值（固定0.7）

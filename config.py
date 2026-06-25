"""
联想式知识图谱增强问答系统 - 配置文件
支持离线(ChatGLM3) / 在线(DeepSeek) 双模式
支持关键词 / Embedding语义匹配
"""

# ===== 模型配置 =====
# 离线模式：本地ChatGLM3
OFFLINE_MODEL_PATH = ""  # 本地GGUF模型路径，如: "F:/models/chatglm3-6b.Q4_K_M.gguf"
OFFLINE_N_CTX = 2048
OFFLINE_N_GPU_LAYERS = 0

# 在线模式：DeepSeek API
ONLINE_PROVIDER = "deepseek"
ONLINE_API_KEY = ""            # 留空则走离线模式；也可通过Web界面输入
ONLINE_BASE_URL = "https://api.deepseek.com"
ONLINE_MODEL = "deepseek-chat"

# 当前运行模式：auto(有key走在线，否则离线) / offline / online
RUN_MODE = "auto"

# ===== Embedding配置 =====
# keyword: 关键词匹配（无需额外依赖）
# embedding: sentence-transformers语义匹配（需安装依赖）
EMBEDDING_MODE = "auto"  # auto / keyword / embedding

# Embedding模型（仅EMBEDDING_MODE=embedding/auto时生效）
# BAAI/bge-small-zh-v1.5: 中文专用，~100MB，推荐
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: 多语言，~500MB
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # cpu / cuda
EMBEDDING_SIMILARITY_THRESHOLD = 0.35  # 语义相似度阈值

# ===== 图谱配置 =====
GRAPH_WALK_DEPTH = 2
GRAPH_TOP_K_NEIGHBORS = 6
SEMANTIC_FILTER_THRESHOLD = 0.3
CROSS_DOMAIN_WEIGHT_BOOST = 1.5  # 跨域概念加权倍数

# ===== 话题推荐配置 =====
TOPIC_MAX_CROSS_DOMAIN = 3  # 跨域推荐上限
TOPIC_MAX_SAME_DOMAIN = 1   # 同域推荐上限

# ===== 服务配置 =====
HOST = "0.0.0.0"
PORT = 8000

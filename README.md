# 联想式知识图谱增强问答系统（Associative GraphRAG）

> 基于**跨域加权游走**的知识图谱增强检索生成系统——不追问"更多"，而是主动联想"你可能还想知道什么"。

## 为什么做这个项目？

传统RAG回答问题后即止，用户必须自己追问才能探索关联知识。微软 GraphRAG 解决的是**大规模文本的理解与摘要**，而本项目解决的是另一个问题：**跨领域概念的主动联想与推荐**。

核心假设：**对跨域边施加加权优先级，可以让 BFS 游走突破同域高分节点的"引力井"，显著提升跨领域概念的可见性。**

Benchmark 验证：跨域加权（1.5x）使跨域概念召回率从 31.6% 提升至 44.7%（**+41.5%**），每查询可见跨域概念从 1.8 个增至 3.2 个（**+78%**）。

## 架构

```
用户问题
    │
    ▼
┌─────────────────┐
│  概念匹配引擎    │  Embedding语义匹配（优先）/ 关键词降级
│  (双模自动切换)  │  BGE-small-zh → 100% 种子召回率
└────────┬────────┘
         │ 种子概念
         ▼
┌─────────────────┐
│  加权BFS图游走   │  跨域边 × 1.5 权重 → 突破同域引力井
│  (NetworkX)     │  深度2 / Top-K 自适应
└────────┬────────┘
         │ 扩展概念（同域+跨域）
         ▼
┌─────────────────┐
│  Prompt增强      │  结构化注入：种子概念 + 关联路径 + 跨域推荐
│  + LLM生成      │  离线ChatGLM3 / 在线DeepSeek API
└────────┬────────┘
         │
         ▼
   回答 + 跨域话题推荐
```

## Benchmark 数据（12-Query Pilot Study）

| 指标 | 无加权基线 | 跨域加权 1.5x | 提升幅度 |
|------|-----------|--------------|---------|
| 跨域概念召回率 | 31.6% | **44.7%** | **+41.5%** |
| 平均跨域概念数/查询 | 1.8 | **3.2** | **+78%** |

### 三组对比设计

| 组别 | 说明 |
|------|------|
| 无加权基线 | BFS游走，所有边等权 |
| 跨域加权1.5x | 跨域边权重 × 1.5（本项目核心创新） |
| 加权 + Embedding | 在加权基础上使用 Embedding 语义匹配种子 |

> **注**：Embedding 列的跨域召回率偏低（7.9%）是评估偏差——Embedding 匹配到不同的种子概念，走出了不同但同样合理的路径，而这些路径不在基于关键词路径编写的 `expected_cross` 列表中。Embedding 的真正价值是 100% 种子召回率（关键词仅 89.5%）。

### 图谱规模

| 指标 | 数值 |
|------|------|
| 概念节点 | 120 |
| 关联边（含反向） | 422 |
| 跨域边 | 156 |
| 领域 | 6（AI / 安全 / 编程 / 跨域 / 数据 / 系统） |

## 核心特性

### 🔗 跨域加权游走
BFS 游走时，跨域边权重乘以 `CROSS_DOMAIN_WEIGHT_BOOST`（默认 1.5），使跨领域概念不会被同域高分节点淹没。这是本项目的核心创新点。

### 🔍 双模概念匹配
- **Embedding 语义匹配**：BAAI/bge-small-zh-v1.5，语义理解能力强，100% 种子召回率
- **关键词匹配**：零依赖降级方案，无需下载模型
- 自动切换：有模型走 Embedding，否则降级关键词

### 🤖 离线/在线双模式 LLM
- **离线**：ChatGLM3-6B（llama-cpp-python），无需网络，隐私安全
- **在线**：DeepSeek API（OpenAI SDK），回答质量更高
- 运行时通过 Web 界面一键切换，API Key 动态注入

### 💡 话题推荐
回答后自动推荐 3 个跨域话题 + 1 个同域话题，引导用户探索知识盲区。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> Embedding 模式需要 `sentence-transformers`（首次运行自动下载 ~95MB 模型，国内需配置 HF 镜像）
> 离线模式需要 `llama-cpp-python` + ChatGLM3 GGUF 模型

### 2. 配置

编辑 `config.py`：

```python
# LLM 模式：auto / offline / online
RUN_MODE = "auto"

# 在线模式需要填入 API Key（也可在 Web 界面输入）
ONLINE_API_KEY = ""

# Embedding 模式：auto / keyword / embedding
EMBEDDING_MODE = "auto"
```

国内用户使用 Embedding 模式时，代码已内置 HuggingFace 镜像（`hf-mirror.com`），无需手动配置。

### 3. 运行

```bash
python web_app.py
```

访问 http://localhost:8000 即可使用。

### 4. Benchmark

```bash
python benchmark.py
```

## 项目结构

```
├── config.py              # 配置中心（模型/图谱/服务参数）
├── graph_data.py          # 知识图谱数据（120节点/233边定义）
├── knowledge_graph.py     # 图谱引擎：NetworkX加权BFS游走
├── embedding_matcher.py   # 概念匹配：Embedding语义/关键词降级
├── llm_engine.py          # LLM引擎：离线ChatGLM3/在线DeepSeek
├── agent.py               # 核心Agent：匹配→游走→增强→生成
├── web_app.py             # FastAPI服务：API路由 + 静态文件挂载
├── benchmark.py           # 三组对比实验
├── static/
│   └── index.html         # 前端页面（独立维护）
└── requirements.txt       # 依赖
```

## 与微软 GraphRAG 的区别

| 维度 | 微软 GraphRAG | 本项目 |
|------|-------------|-------|
| 目标 | 大规模文本的理解与摘要 | 跨领域概念的主动联想与推荐 |
| 输入 | 原始文档语料 | 预构建知识图谱 |
| 图构建 | LLM自动抽取实体和关系 | 人工定义 + 可扩展 |
| 检索策略 | 社区摘要 → 全局理解 | 加权BFS游走 → 跨域联想 |
| 核心创新 | 社区检测 + 全局摘要 | 跨域边加权优先级 |

**同源**：都采用"图结构增强 RAG"的管线架构。**不同问题定义**：微软解决"规模理解"，本项目解决"跨域联想"。

## 后续规划

- [ ] 动态游走深度：根据查询语义复杂度自适应调整 BFS 深度
- [ ] 扩展 benchmark 至 50+ 查询，提升统计置信度
- [ ] 支持图谱热更新（运行时增删节点/边）
- [ ] 接入更多 LLM 后端（Ollama、vLLM 等）

## 技术栈

Python · FastAPI · NetworkX · Sentence-Transformers · llama-cpp-python · OpenAI SDK

## License

MIT

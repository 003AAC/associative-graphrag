"""
FastAPI Web服务 - 提供API + 图谱可视化 + 设置面板
前端页面由 static/index.html 独立维护
"""
import logging
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from knowledge_graph import KnowledgeGraph
from llm_engine import LLMEngine
from agent import KnowledgeAgent
from config import HOST, PORT

logger = logging.getLogger(__name__)

# ===== 初始化 =====
kg = KnowledgeGraph()
llm = LLMEngine()
agent = KnowledgeAgent(kg, llm)

app = FastAPI(title="联想式知识图谱Agent", version="0.3.0")

# 静态文件挂载
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ===== 请求/响应模型 =====
class QuestionRequest(BaseModel):
    question: str
    walk_depth: Optional[int] = None

class ModeSwitchRequest(BaseModel):
    mode: str  # offline / online

class ApiKeyRequest(BaseModel):
    api_key: str


class EvocativeExploreRequest(BaseModel):
    original_question: str
    evocative: dict


# ===== 页面路由 =====
@app.get("/")
async def index():
    """返回前端页面"""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="前端页面未找到，请确认 static/index.html 存在")


# ===== API路由 =====
@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    result = agent.answer(req.question)
    return result


@app.get("/api/graph/nodes")
async def get_nodes():
    return {"nodes": kg.get_all_nodes()}


@app.get("/api/graph/edges")
async def get_edges():
    return {"edges": kg.get_all_edges()}


@app.get("/api/graph/stats")
async def get_stats():
    return kg.stats()


@app.get("/api/graph/walk")
async def walk_from_concept(concept: str, depth: int = 2, top_k: int = 6):
    if concept not in kg.graph:
        raise HTTPException(status_code=404, detail=f"概念 '{concept}' 不在图谱中")
    results = kg.weighted_walk([concept], depth=depth, top_k=top_k)
    return {"seed": concept, "related": results}


@app.post("/api/mode")
async def switch_mode(req: ModeSwitchRequest):
    if req.mode not in ("offline", "online"):
        raise HTTPException(status_code=400, detail="模式必须是 offline 或 online")
    try:
        llm.switch_mode(req.mode)
        return {"mode": llm.current_mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mode")
async def get_mode():
    return {"mode": llm.current_mode, "has_api_key": llm.has_api_key}


@app.post("/api/apikey")
async def set_api_key(req: ApiKeyRequest):
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API Key不能为空")
    llm.set_api_key(req.api_key.strip())
    return {"mode": llm.current_mode, "message": f"API Key已设置，切换到在线模式"}


@app.get("/api/status")
async def get_status():
    return {
        "llm_mode": llm.current_mode,
        "has_api_key": llm.has_api_key,
        "match_mode": agent.matcher.mode,
        "embedding_available": agent.matcher.available,
        "graph_stats": kg.stats(),
        "discovered_total": agent.discoverer.total_discovered if agent.discoverer else 0
    }


@app.get("/api/discoveries")
async def get_discoveries(n: int = 20):
    """获取最近发现的图谱关系"""
    if not agent.discoverer:
        return {"discoveries": [], "total": 0}
    return {
        "discoveries": agent.discoverer.get_recent(n),
        "total": agent.discoverer.total_discovered
    }


@app.post("/api/evocative/explore")
async def explore_evocative(req: EvocativeExploreRequest):
    """用户点击启发式反问后，深入展开隐藏跨域关联"""
    try:
        explanation = agent.evocative.explore_evocative(req.original_question, req.evocative)
        return {"explanation": explanation, "evocative": req.evocative}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = kg.stats()
    logger.info(f"🚀 启动服务: http://{HOST}:{PORT}")
    logger.info(f"🧠 LLM模式: {llm.current_mode}")
    logger.info(f"🔍 匹配模式: {agent.matcher.mode}")
    logger.info(f"📊 图谱: {stats['nodes']}节点 / {stats['edges']}边 / {stats['cross_domain_edges']}跨域边")
    uvicorn.run(app, host=HOST, port=PORT)

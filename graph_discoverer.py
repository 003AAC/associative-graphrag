"""
图谱关系自动发现器 - AI自主扩展图谱
每次问答后自动触发：LLM从问答中识别概念和关系 → 新概念/新边写入图谱 + 持久化JSON
即使当前图谱完全没有相关概念，也能从零扩展
"""
import json
import os
import logging
from typing import List, Dict
from knowledge_graph import KnowledgeGraph
from llm_engine import LLMEngine
from config import GRAPH_DISCOVER_THRESHOLD

logger = logging.getLogger(__name__)

DISCOVERED_FILE = os.path.join(os.path.dirname(__file__), "discovered_relations.json")


class GraphDiscoverer:
    """从问答结果中自动发现新的概念和关系，自主扩展图谱"""

    def __init__(self, kg: KnowledgeGraph, llm: LLMEngine):
        self.kg = kg
        self.llm = llm
        self._discovered = self._load_discovered()
        self._apply_discovered()
        logger.info(f"🔎 图谱发现器就绪，已加载 {len(self._discovered)} 条历史发现")

    def _load_discovered(self) -> List[Dict]:
        if os.path.exists(DISCOVERED_FILE):
            try:
                with open(DISCOVERED_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ 加载discovered_relations.json失败: {e}")
        return []

    def _save_discovered(self):
        try:
            with open(DISCOVERED_FILE, "w", encoding="utf-8") as f:
                json.dump(self._discovered, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ 保存discovered_relations.json失败: {e}")

    def _apply_discovered(self):
        applied = 0
        for item in self._discovered:
            if not item.get("applied", False):
                continue
            self._apply_one(item)
            applied += 1
        if applied > 0:
            logger.info(f"✅ 已应用 {applied} 条历史发现到图谱")

    def _apply_one(self, item: Dict):
        src = item.get("src", "")
        dst = item.get("dst", "")
        relation = item.get("relation", "")
        weight = item.get("weight", 0.5)

        if not src or not dst:
            return

        # 新节点自动添加
        if src not in self.kg.graph:
            self.kg.add_node(src, domain=item.get("src_domain", ""), desc=item.get("src_desc", ""))
        if dst not in self.kg.graph:
            self.kg.add_node(dst, domain=item.get("dst_domain", ""), desc=item.get("dst_desc", ""))

        # 添加边（不覆盖已有的）
        if not self.kg.graph.has_edge(src, dst):
            self.kg.add_edge(src, dst, relation=relation, weight=weight)

    def discover(self, question: str, answer: str, seed_concepts: List[str],
                 related_concepts: List[Dict]) -> List[Dict]:
        """
        从一次问答中发现新概念和关系
        不依赖种子概念——即使种子为空，也能从问答中识别全新的概念
        """
        # 已有概念集合
        existing_nodes = set(self.kg.graph.nodes)

        # 构建抽取Prompt
        prompt = self._build_extract_prompt(question, answer, existing_nodes)

        # LLM抽取
        raw_output = self.llm.generate(prompt, max_tokens=800, temperature=0.1)
        logger.info(f"🔎 LLM发现原始输出: {raw_output[:200]}...")

        # 解析
        triples = self._parse_triples(raw_output)
        if not triples:
            logger.info("🔎 未发现新概念/关系")
            return []

        # 过滤低置信度
        filtered = [t for t in triples if t.get("confidence", 0) >= GRAPH_DISCOVER_THRESHOLD]

        # 写入图谱 + 持久化
        new_items = []
        new_nodes_count = 0
        new_edges_count = 0

        for t in filtered:
            src = t["src"]
            dst = t["dst"]
            src_is_new = src not in existing_nodes
            dst_is_new = dst not in existing_nodes

            item = {
                "src": src,
                "dst": dst,
                "relation": t["relation"],
                "weight": min(t.get("confidence", 0.5), 0.9),
                "confidence": t.get("confidence", 0.5),
                "src_domain": t.get("src_domain", ""),
                "src_desc": t.get("src_desc", ""),
                "dst_domain": t.get("dst_domain", ""),
                "dst_desc": t.get("dst_desc", ""),
                "question": question,
                "applied": True
            }

            # 应用到图谱
            self._apply_one(item)

            if src_is_new:
                new_nodes_count += 1
                existing_nodes.add(src)
            if dst_is_new:
                new_nodes_count += 1
                existing_nodes.add(dst)
            new_edges_count += 1
            new_items.append(item)

        if new_items:
            self._discovered.extend(new_items)
            self._save_discovered()
            logger.info(f"✨ 发现 {new_nodes_count} 个新概念 + {new_edges_count} 条新关系")

        return new_items

    def _build_extract_prompt(self, question: str, answer: str, existing_nodes: set) -> str:
        """
        构建关系抽取Prompt
        关键：不管图谱里有没有相关概念，都让LLM从问答中提取概念和关系
        """
        existing_list = sorted(existing_nodes)[:60]  # 限制长度
        existing_str = ", ".join(existing_list)

        prompt = f"""你是一个知识图谱构建助手。请从以下问答中提取关键概念和它们之间的关系。

【用户问题】{question}

【AI回答】{answer}

【图谱已有概念（部分）】{existing_str}

请严格按照以下JSON格式输出，不要输出其他内容：
```json
[
  {{
    "src": "源概念名",
    "dst": "目标概念名",
    "relation": "关系描述（如：包含、依赖、应用于、是...的基础、解释）",
    "confidence": 0.8,
    "src_domain": "源概念所属领域",
    "src_desc": "源概念一句话描述",
    "dst_domain": "目标概念所属领域",
    "dst_desc": "目标概念一句话描述"
  }}
]
```

提取规则：
1. 从问答中识别所有值得收录的专业概念，不管图谱里有没有
2. 概念不在已有列表中的，必须给出domain和desc，这是新概念
3. 概念已在已有列表中的，domain和desc可以留空
4. 只提取有明确语义关系的三元组，不要硬凑
5. confidence范围0.0~1.0，关系越明确值越高
6. 如果没有发现任何概念或关系，输出空列表 []
7. 最多输出8条关系
8. 特别注意跨领域的联系（如物理与AI的交叉）"""

        return prompt

    def _parse_triples(self, raw: str) -> List[Dict]:
        json_str = raw.strip()

        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1]
            json_str = json_str.split("```", 1)[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1]
            json_str = json_str.split("```", 1)[0]

        json_str = json_str.strip()

        try:
            triples = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str_fixed = json_str.replace(",]", "]").replace(",}", "}")
                triples = json.loads(json_str_fixed)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ 无法解析LLM输出的JSON: {json_str[:100]}")
                return []

        if not isinstance(triples, list):
            return []

        valid = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            if not t.get("src") or not t.get("dst") or not t.get("relation"):
                continue
            t["confidence"] = float(t.get("confidence", 0.5))
            valid.append(t)

        return valid

    @property
    def total_discovered(self) -> int:
        return len(self._discovered)

    def get_recent(self, n: int = 10) -> List[Dict]:
        return self._discovered[-n:]

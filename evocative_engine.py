"""
启发式探索引擎 (Evocative Engine) - 回答后主动发现隐藏关联并引导用户探索
核心理念：先回答，再启发。不是被动等用户问，而是主动指出"你可能没想到的关联"
"""
import logging
from typing import List, Dict, Optional, Tuple
from knowledge_graph import KnowledgeGraph
from llm_engine import LLMEngine

logger = logging.getLogger(__name__)


class EvocativeEngine:
    """回答后主动发现隐藏跨域关联，生成启发式反问"""

    def __init__(self, kg: KnowledgeGraph, llm: LLMEngine):
        self.kg = kg
        self.llm = llm

    def find_evocative(self, question: str, answer_text: str,
                       seed_concepts: List[str], related_concepts: List[Dict]) -> Optional[Dict]:
        """
        从已回答的问答中，找到最有启发性的隐藏跨域关联

        返回 None 表示没有发现值得启发的关联
        返回 Dict 包含：
          - bridge_concept: 桥接概念（连接用户话题和隐藏话题的关键节点）
          - hidden_concept: 隐藏的跨域概念
          - hidden_domain: 隐藏概念的领域
          - bridge_path: 概念路径 [seed → ... → hidden]
          - evocative_question: 启发式反问
          - reasoning: 为什么这个关联令人惊讶/有价值
        """
        if not seed_concepts and not related_concepts:
            return None

        # Step 1: 收集本次对话涉及的概念和领域
        involved_concepts = set(seed_concepts)
        involved_concepts.update(r["concept"] for r in related_concepts)
        involved_domains = set()
        for c in involved_concepts:
            info = self.kg.get_node_info(c)
            if info and info.get("domain"):
                involved_domains.add(info["domain"])

        # Step 2: 从涉及概念出发，2-3跳游走，找跨域但不在related中的节点
        hidden_gems = self._find_hidden_gems(involved_concepts, involved_domains, related_concepts)
        if not hidden_gems:
            logger.info("💡 未发现值得启发的隐藏关联")
            return None

        # Step 3: 用LLM评估哪个关联最令人惊讶，生成反问
        best = self._select_and_generate(question, answer_text, involved_concepts, involved_domains, hidden_gems)
        return best

    def _find_hidden_gems(self, involved_concepts: set, involved_domains: set,
                          related_concepts: List[Dict]) -> List[Dict]:
        """
        从涉及概念出发游走3跳，找到"跨域且不在related中"的隐藏宝石
        """
        related_names = {r["concept"] for r in related_concepts}
        # 种子+关联都是"已展示"的
        shown = involved_concepts | related_names

        gems = []
        visited = set(shown)
        frontier = [c for c in involved_concepts if c in self.kg.graph]

        for hop in range(3):
            next_frontier = []
            for node in frontier:
                if node not in self.kg.graph:
                    continue
                for neighbor in self.kg.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)

                    # 只收集：跨域 + 不在已展示中的
                    info = self.kg.get_node_info(neighbor)
                    if not info:
                        continue
                    n_domain = info.get("domain", "")
                    if n_domain and n_domain not in involved_domains and neighbor not in shown:
                        gems.append({
                            "concept": neighbor,
                            "domain": n_domain,
                            "desc": info.get("desc", ""),
                            "hop": hop + 1
                        })

            frontier = next_frontier

        # 优先近跳的（1跳>2跳>3跳），最多保留8个候选
        gems.sort(key=lambda g: g["hop"])
        return gems[:8]

    def _select_and_generate(self, question: str, answer_text: str,
                              involved_concepts: set, involved_domains: set,
                              gems: List[Dict]) -> Optional[Dict]:
        """
        用LLM从候选宝石中选出最令人惊讶的，生成启发式反问
        """
        # 找到桥接路径：种子到hidden的最短路径
        valid_seeds = [c for c in involved_concepts if c in self.kg.graph]

        # 构建候选描述
        gem_lines = []
        for i, g in enumerate(gems):
            gem_lines.append(f"{i+1}. {g['concept']}（{g['domain']}）: {g['desc']} — {g['hop']}跳可达")

        gems_text = "\n".join(gem_lines)
        domains_str = "、".join(involved_domains) if involved_domains else "未知"

        prompt = f"""你是一个善于发现跨领域隐藏关联的思维导师。用户刚问了一个关于「{domains_str}」的问题，你发现了一些用户可能没想到的跨域关联。

【用户问题】{question}

【已回答的核心领域】{domains_str}

【发现的隐藏跨域候选】
{gems_text}

请选出最令人惊讶且最有价值的1个关联，并生成一个启发式反问。

要求：
1. 反问要自然，像朋友聊天时的灵光一闪，不要教科书腔
2. 反问要指明具体的关联方向，让用户觉得"咦，这确实有关系"
3. 推理链要简短但有力，1-2句话说清楚为什么这个关联令人惊讶

请严格按以下JSON格式输出：
```json
{{
  "selected_index": 1,
  "evocative_question": "你有没有想过，这其实和XX有关？",
  "reasoning": "因为XX原理可以直接映射到YY现象..."
}}
```

如果没有哪个关联足够令人惊讶，输出：
```json
{{"selected_index": 0, "evocative_question": "", "reasoning": ""}}
```"""

        try:
            raw = self.llm.generate(prompt, max_tokens=300, temperature=0.7)
            result = self._parse_selection(raw)
            if not result or result.get("selected_index", 0) == 0:
                return None

            idx = result["selected_index"] - 1
            if idx < 0 or idx >= len(gems):
                return None

            gem = gems[idx]

            # 找桥接路径
            bridge_path = []
            if valid_seeds:
                path = self.kg.get_walk_path(valid_seeds[0], gem["concept"])
                if path:
                    bridge_path = path

            return {
                "bridge_concept": bridge_path[1] if len(bridge_path) > 1 else (valid_seeds[0] if valid_seeds else ""),
                "hidden_concept": gem["concept"],
                "hidden_domain": gem["domain"],
                "bridge_path": bridge_path,
                "evocative_question": result["evocative_question"],
                "reasoning": result["reasoning"]
            }

        except Exception as e:
            logger.warning(f"⚠️ 启发式生成失败: {e}")
            return None

    def explore_evocative(self, original_question: str, evocative: Dict) -> str:
        """
        用户点击启发式反问后，深入展开隐藏关联
        返回一段关于"为什么这个跨域关联令人惊讶"的深入解释
        """
        hidden = evocative["hidden_concept"]
        hidden_domain = evocative["hidden_domain"]
        bridge = evocative["bridge_concept"]
        path = evocative.get("bridge_path", [])
        reasoning = evocative.get("reasoning", "")

        path_str = " → ".join(path) if path else f"{bridge} → {hidden}"

        # 收集隐藏概念在图谱中的邻居信息
        neighbors = []
        if hidden in self.kg.graph:
            for nb in list(self.kg.graph.neighbors(hidden))[:8]:
                info = self.kg.get_node_info(nb)
                edge = self.kg.graph[hidden][nb]
                neighbors.append({
                    "concept": nb,
                    "domain": info.get("domain", "") if info else "",
                    "relation": edge.get("relation", "")
                })

        nb_lines = "\n".join(
            f"- {n['concept']}（{n['domain']}）— {n['relation']}"
            for n in neighbors
        ) if neighbors else "无直接邻居"

        prompt = f"""你是一个善于揭示跨领域深层联系的知识导游。用户对你的启发式反问「{evocative['evocative_question']}」很感兴趣，请深入展开这个关联。

【原问题】{original_question}

【跨域关联路径】{path_str}

【推理线索】{reasoning}

【{hidden}在图谱中的关联】
{nb_lines}

请用聊天的方式深入展开，要求：
1. 先说清楚"为什么这两件事有关系"——具体的逻辑链，不要笼统
2. 举一个生动的类比或例子，让关联变得直观
3. 如果图谱中有更多可探索的方向，自然地提一下
4. 风格：像博学的朋友分享一个让你兴奋的发现，不是写论文
5. 控制在300字以内"""

        try:
            return self.llm.generate(prompt, max_tokens=400, temperature=0.5)
        except Exception as e:
            logger.warning(f"⚠️ 启发式展开失败: {e}")
            return f"关于{bridge}和{hidden}的关联：{reasoning}（LLM展开失败，请稍后重试）"

    def _parse_selection(self, raw: str) -> Optional[Dict]:
        """解析LLM选择结果"""
        import json as _json
        s = raw.strip()
        if "```json" in s:
            s = s.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in s:
            s = s.split("```", 1)[1].split("```", 1)[0]
        s = s.strip()
        return _json.loads(s)

"""
联想式知识Agent - GraphRAG核心
用户问题 → 语义匹配种子概念 → 图游走扩展 → 增强Prompt → LLM生成回答 + 话题拓展推荐
支持Embedding语义匹配（优先）和关键词匹配（降级）
"""
import json
import re
import logging
from typing import List, Dict, Optional
from knowledge_graph import KnowledgeGraph
from llm_engine import LLMEngine
from embedding_matcher import EmbeddingMatcher
from config import (
    GRAPH_WALK_DEPTH, GRAPH_TOP_K_NEIGHBORS,
    CROSS_DOMAIN_WEIGHT_BOOST, TOPIC_MAX_CROSS_DOMAIN, TOPIC_MAX_SAME_DOMAIN
)

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """联想式知识图谱增强问答Agent"""

    def __init__(self, kg: KnowledgeGraph, llm: LLMEngine):
        self.kg = kg
        self.llm = llm
        # 初始化Embedding匹配器
        concepts = list(self.kg.graph.nodes)
        descriptions = [
            self.kg.get_node_info(c).get("desc", c) if self.kg.get_node_info(c) else c
            for c in concepts
        ]
        self.matcher = EmbeddingMatcher(concepts, descriptions)
        self._known_concepts = concepts
        logger.info(f"🔍 概念匹配模式: {self.matcher.mode}")

    def extract_seed_concepts(self, user_input: str) -> List[Dict]:
        """
        从用户输入中提取种子概念
        优先用Embedding语义匹配，降级到关键词匹配
        返回: [{"concept": str, "score": float, "source": str}, ...]
        """
        # Embedding匹配
        matches = self.matcher.match(user_input, top_k=5)
        results = []
        for concept, score in matches:
            info = self.kg.get_node_info(concept)
            results.append({
                "concept": concept,
                "score": score,
                "domain": info.get("domain", "") if info else "",
                "source": "embedding" if self.matcher.mode == "embedding" else "keyword"
            })

        # 如果匹配为空，尝试用描述中的关键词模糊匹配
        if not results:
            results = self._fuzzy_match(user_input)

        return results

    def _fuzzy_match(self, user_input: str) -> List[Dict]:
        """模糊匹配：检查用户输入是否包含概念描述中的关键词"""
        found = []
        input_lower = user_input.lower()
        for node in self._known_concepts:
            info = self.kg.get_node_info(node)
            if info and info.get("desc", ""):
                desc_keywords = info["desc"].replace("，", " ").replace("、", " ").replace("。", " ").split()
                for kw in desc_keywords:
                    if len(kw) >= 2 and kw in input_lower:
                        found.append({
                            "concept": node,
                            "score": 0.5,
                            "domain": info.get("domain", ""),
                            "source": "fuzzy"
                        })
                        break
        return found[:5]

    def expand_concepts(self, seed_concepts: List[str], depth: int = None) -> List[Dict]:
        """
        从种子概念出发，加权BFS游走扩展关联概念
        跨域概念额外加权，确保不被同域高分淹没
        """
        if not depth:
            depth = GRAPH_WALK_DEPTH

        if not seed_concepts:
            return []

        # 获取种子概念的领域
        seed_domains = set()
        for s in seed_concepts:
            info = self.kg.get_node_info(s)
            if info and info.get("domain"):
                seed_domains.add(info["domain"])

        # 多取一些候选，给跨域加权后重新排序
        expanded_top_k = max(GRAPH_TOP_K_NEIGHBORS * 2, 12)
        results = self.kg.weighted_walk(seed_concepts, depth=depth, top_k=expanded_top_k)

        # 跨域加权
        for r in results:
            r_domain = r.get("domain", "")
            if r_domain not in seed_domains and r_domain != "":
                r["score"] = round(r["score"] * CROSS_DOMAIN_WEIGHT_BOOST, 3)
                r["cross_domain"] = True
            else:
                r["cross_domain"] = False

        results.sort(key=lambda x: -x["score"])
        return results[:GRAPH_TOP_K_NEIGHBORS]

    def build_topic_suggestions(self, seed_concepts: List[str], related_concepts: List[Dict]) -> List[Dict]:
        """
        生成话题拓展推荐（独立于LLM回答，确保跨域概念可见）
        """
        if not related_concepts:
            return []

        suggestions = []
        cross_domain = [r for r in related_concepts if r.get("cross_domain")]
        same_domain = [r for r in related_concepts if not r.get("cross_domain")]

        for rc in cross_domain[:TOPIC_MAX_CROSS_DOMAIN]:
            suggestions.append({
                "concept": rc["concept"],
                "domain": rc["domain"],
                "cross_domain": True
            })
        for rc in same_domain[:TOPIC_MAX_SAME_DOMAIN]:
            suggestions.append({
                "concept": rc["concept"],
                "domain": rc["domain"],
                "cross_domain": False
            })

        return suggestions[:4]

    def build_enhanced_prompt(self, user_input: str, related_concepts: List[Dict], walk_path: List = None) -> str:
        """构建增强Prompt：注入图游走得到的关联概念"""

        concept_lines = []
        for rc in related_concepts:
            concept_lines.append(f"- {rc['concept']}（{rc['domain']}）: {rc['desc']}，关联度={rc['score']}")

        concepts_text = "\n".join(concept_lines) if concept_lines else "无额外关联概念"

        path_text = ""
        if walk_path:
            path_text = f"\n概念游走路径: {' → '.join(walk_path)}"

        prompt = f"""你是一个知识渊博的AI助手。回答用户问题时，请自然地融入以下关联概念（如果适用），帮助用户建立知识之间的联系。

【关联概念】
{concepts_text}
{path_text}

【用户问题】
{user_input}

【回答要求】
1. 直接回答用户问题
2. 自然地提及与问题相关的概念，帮助用户理解知识之间的关联
3. 如果存在跨领域的联系（如AI与物理学的交叉），请特别指出
4. 回答简洁清晰，避免冗余

回答："""

        return prompt

    def answer(self, user_input: str) -> Dict:
        """
        主入口：回答用户问题
        返回包含回答、种子概念、关联概念、游走路径、话题推荐的完整结果
        """
        logger.info(f"📝 收到问题: {user_input}")

        # Step 1: 提取种子概念（语义匹配）
        seed_results = self.extract_seed_concepts(user_input)
        seed_concepts = [r["concept"] for r in seed_results]
        seed_source = seed_results[0]["source"] if seed_results else "none"
        logger.info(f"🔑 种子概念: {seed_concepts} (来源: {seed_source})")

        # Step 2: 图游走扩展（跨域加权）
        related_concepts = []
        walk_paths = []
        if seed_concepts:
            related_concepts = self.expand_concepts(seed_concepts)
            if related_concepts:
                for seed in seed_concepts:
                    path = self.kg.get_walk_path(seed, related_concepts[0]["concept"])
                    if path:
                        walk_paths.append(path)
        logger.info(f"🔗 关联概念: {[r['concept'] for r in related_concepts]}")

        # Step 3: 话题拓展推荐
        topic_suggestions = self.build_topic_suggestions(seed_concepts, related_concepts)

        # Step 4: 构建增强Prompt
        enhanced_prompt = self.build_enhanced_prompt(
            user_input, related_concepts, walk_paths[0] if walk_paths else None
        )

        # Step 5: LLM生成回答
        answer_text = self.llm.generate(enhanced_prompt, max_tokens=500, temperature=0.3)
        logger.info(f"💬 生成回答: {answer_text[:80]}...")

        return {
            "question": user_input,
            "seed_concepts": seed_concepts,
            "seed_source": seed_source,
            "related_concepts": related_concepts,
            "walk_paths": walk_paths,
            "topic_suggestions": topic_suggestions,
            "answer": answer_text,
            "llm_mode": self.llm.current_mode,
            "match_mode": self.matcher.mode
        }


# CLI测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    kg = KnowledgeGraph()
    llm = LLMEngine()
    agent = KnowledgeAgent(kg, llm)

    print(f"\n{'=' * 60}")
    print(f"🧠 联想式知识图谱Agent（GraphRAG）")
    print(f"📊 图谱: {kg.stats()}")
    print(f"🧠 LLM模式: {llm.current_mode}")
    print(f"🔍 匹配模式: {agent.matcher.mode}")
    print(f"{'=' * 60}\n")

    while True:
        user_input = input("❓ ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = agent.answer(user_input)
        print(f"\n🔑 种子: {result['seed_concepts']} ({result['seed_source']})")
        print(f"🔗 关联: {[r['concept'] for r in result['related_concepts']]}")
        if result.get('topic_suggestions'):
            print(f"💡 话题: {[s['concept'] + ('(跨域)' if s['cross_domain'] else '') for s in result['topic_suggestions']]}")
        print(f"🤖 [{result['llm_mode']}] {result['answer']}\n")

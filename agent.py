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
    CROSS_DOMAIN_WEIGHT_BOOST, TOPIC_MAX_CROSS_DOMAIN, TOPIC_MAX_SAME_DOMAIN,
    GRAPH_DISCOVER_ENABLED
)
from graph_discoverer import GraphDiscoverer
from evocative_engine import EvocativeEngine

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
        # 图谱发现器
        self.discoverer = GraphDiscoverer(kg, llm) if GRAPH_DISCOVER_ENABLED else None
        # 启发式探索引擎
        self.evocative = EvocativeEngine(kg, llm)
        logger.info(f"🔍 概念匹配模式: {self.matcher.mode}")
        logger.info(f"🔎 自动发现: {'开启' if self.discoverer else '关闭'}")
        logger.info(f"💡 启发式探索: 已启用")

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

    def _llm_extract_seeds(self, user_input: str) -> List[Dict]:
        """种子为空时，用LLM从问题中提取关键概念"""
        prompt = f"""请从以下问题中提取2-4个核心专业概念，只输出概念名，用逗号分隔，不要输出其他内容。
问题：{user_input}"""
        try:
            raw = self.llm.generate(prompt, max_tokens=100, temperature=0.1)
            concepts = [c.strip() for c in raw.replace("，", ",").split(",") if c.strip()]
            results = []
            for c in concepts[:5]:
                info = self.kg.get_node_info(c)
                results.append({
                    "concept": c,
                    "score": 0.4,
                    "domain": info.get("domain", "") if info else "",
                    "source": "llm_extract"
                })
            return results
        except Exception as e:
            logger.warning(f"⚠️ LLM种子提取失败: {e}")
            return []

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

        prompt = f"""你是一个擅长联想和跨领域连接的知识助手。回答问题时，像一位博学的朋友在聊天——不是背诵教科书，而是帮用户看到概念之间的深层联系。

【关联概念】
{concepts_text}
{path_text}

【用户问题】
{user_input}

【回答要求】
1. 先直接回答问题，用通俗的语言讲清楚核心逻辑
2. 然后自然地引出关联概念，说明"这和XX有关联，因为..."
3. 如果存在跨领域的联系（如物理原理映射到AI算法），一定要点出来，这是最有价值的洞察
4. 用类比和直觉帮助理解，而不是堆砌公式和定义
5. 风格：像和聪明的朋友聊天，不是写论文

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

        # 种子为空时，用LLM从问题中提取概念
        if not seed_concepts:
            seed_results = self._llm_extract_seeds(user_input)
            seed_concepts = [r["concept"] for r in seed_results]
            seed_source = "llm_extract" if seed_concepts else "none"
            logger.info(f"🔑 LLM提取种子: {seed_concepts}")

        # 过滤：只保留图谱中存在的种子概念
        valid_seeds = [c for c in seed_concepts if c in self.kg.graph]
        dropped = [c for c in seed_concepts if c not in self.kg.graph]
        if dropped:
            logger.info(f"🔍 种子过滤: {dropped} 不在图谱中，尝试模糊匹配")
            # 对不在图谱中的种子，尝试模糊匹配到最近的概念
            for c in dropped:
                fuzzy = self.matcher.match(c, top_k=1)
                if fuzzy and fuzzy[0][1] >= 0.3:
                    valid_seeds.append(fuzzy[0][0])
                    logger.info(f"🔍 '{c}' → 模糊匹配到 '{fuzzy[0][0]}' (score={fuzzy[0][1]:.2f})")
        seed_concepts = valid_seeds
        if not seed_concepts:
            logger.info(f"🔑 无有效种子概念，跳过图游走")

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
        answer_text = ""
        llm_error = None
        try:
            answer_text = self.llm.generate(enhanced_prompt, max_tokens=500, temperature=0.3)
            logger.info(f"💬 生成回答: {answer_text[:80]}...")
        except Exception as e:
            llm_error = str(e)
            logger.warning(f"⚠️ LLM生成失败: {e}")
            # 降级：基于关联概念生成简要回答
            if related_concepts:
                concept_names = [r["concept"] + "（" + r["domain"] + "）" for r in related_concepts[:5]]
                answer_text = f"关于「{user_input}」，图谱中找到以下关联概念：{', '.join(concept_names)}。但LLM调用失败({llm_error})，无法生成详细回答。请检查API Key配置。"
            else:
                answer_text = f"无法回答「{user_input}」：未找到相关概念且LLM调用失败({llm_error})。请检查API Key配置。"

        # Step 6: 自动发现新关系（无论有无种子，都尝试从问答中扩展图谱）
        discovered = []
        if self.discoverer:
            try:
                discovered = self.discoverer.discover(
                    user_input, answer_text, seed_concepts, related_concepts
                )
            except Exception as e:
                logger.warning(f"⚠️ 图谱发现失败: {e}")

        # Step 7: 启发式探索——回答后主动发现隐藏跨域关联
        evocative_result = None
        if answer_text and not llm_error:
            try:
                evocative_result = self.evocative.find_evocative(
                    user_input, answer_text, seed_concepts, related_concepts
                )
                if evocative_result:
                    logger.info(f"💡 启发式发现: {evocative_result['hidden_concept']}({evocative_result['hidden_domain']}) ← {evocative_result['bridge_concept']}")
            except Exception as e:
                logger.warning(f"⚠️ 启发式探索失败: {e}")

        return {
            "question": user_input,
            "seed_concepts": seed_concepts,
            "seed_source": seed_source,
            "related_concepts": related_concepts,
            "walk_paths": walk_paths,
            "topic_suggestions": topic_suggestions,
            "answer": answer_text,
            "discovered_relations": discovered,
            "evocative": evocative_result,
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
    print(f"🔎 自动发现: {'开启' if agent.discoverer else '关闭'}")
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
        if result.get('discovered_relations'):
            disc = [d['src'] + '→' + d['dst'] + '(' + d['relation'] + ')' for d in result['discovered_relations']]
            print(f"✨ 发现: {disc}")
        print(f"🤖 [{result['llm_mode']}] {result['answer']}\n")

"""
对比实验框架 - 量化评估跨域加权策略效果
三组对比：无加权基线 / 跨域加权(核心创新) / 跨域加权+Embedding
核心指标：跨域概念可见率（跨域概念进入Top-K的比例）
"""
import json
import logging
from typing import List, Dict, Tuple
from knowledge_graph import KnowledgeGraph
from embedding_matcher import EmbeddingMatcher
from config import GRAPH_WALK_DEPTH, GRAPH_TOP_K_NEIGHBORS, CROSS_DOMAIN_WEIGHT_BOOST

logger = logging.getLogger(__name__)

# ===== 测试查询集 =====
TEST_QUERIES = [
    {
        "query": "Transformer和注意力机制有什么关系",
        "expected_seeds": ["Transformer", "注意力机制"],
        "expected_cross": ["认知科学", "复杂系统"],
        "category": "AI-同域"
    },
    {
        "query": "C++和图灵机有什么关系",
        "expected_seeds": ["C++", "图灵机"],
        "expected_cross": ["可计算性理论", "Agent", "复杂系统"],
        "category": "编程-跨域"
    },
    {
        "query": "漏洞扫描是怎么工作的",
        "expected_seeds": ["漏洞扫描"],
        "expected_cross": ["C++", "Agent", "静态分析"],
        "category": "安全-跨域"
    },
    {
        "query": "信息论和机器学习的关系",
        "expected_seeds": ["信息论"],
        "expected_cross": ["Embedding", "向量数据库", "密码学"],
        "category": "理论-AI"
    },
    {
        "query": "什么是RAG技术",
        "expected_seeds": ["RAG"],
        "expected_cross": ["知识图谱", "Embedding", "安全审计"],
        "category": "AI-同域"
    },
    {
        "query": "量子计算对密码学有什么影响",
        "expected_seeds": ["量子计算", "密码学"],
        "expected_cross": ["信息论", "数字签名", "安全审计"],
        "category": "理论-安全"
    },
    {
        "query": "Docker和微服务架构",
        "expected_seeds": ["Docker", "微服务"],
        "expected_cross": ["Kubernetes", "容器化", "负载均衡"],
        "category": "编程-系统"
    },
    {
        "query": "博弈论在强化学习中的应用",
        "expected_seeds": ["博弈论", "强化学习"],
        "expected_cross": ["MCTS", "Agent", "RLHF", "优化理论"],
        "category": "理论-AI"
    },
    {
        "query": "知识蒸馏和模型压缩",
        "expected_seeds": ["知识蒸馏"],
        "expected_cross": ["微调", "LoRA", "特征工程"],
        "category": "AI-同域"
    },
    {
        "query": "C++的内存安全和漏洞防范",
        "expected_seeds": ["C++", "内存管理"],
        "expected_cross": ["缓冲区溢出", "漏洞扫描", "静态分析", "Rust"],
        "category": "编程-安全"
    },
    {
        "query": "分布式系统的一致性问题",
        "expected_seeds": ["分布式系统", "一致性协议"],
        "expected_cross": ["复杂系统", "Kafka", "涌现"],
        "category": "系统-跨域"
    },
    {
        "query": "认知科学如何启发AI发展",
        "expected_seeds": ["认知科学"],
        "expected_cross": ["注意力机制", "LLM", "Agent", "复杂系统"],
        "category": "理论-AI"
    },
]


def _do_walk(kg: KnowledgeGraph, seeds: List[str], apply_cross_weight: bool) -> List[Dict]:
    """执行图游走，可选是否应用跨域加权"""
    if not seeds:
        return []

    seed_domains = set()
    for s in seeds:
        info = kg.get_node_info(s)
        if info and info.get("domain"):
            seed_domains.add(info["domain"])

    walk = kg.weighted_walk(seeds, depth=GRAPH_WALK_DEPTH, top_k=GRAPH_TOP_K_NEIGHBORS * 2)

    for r in walk:
        r_domain = r.get("domain", "")
        is_cross = r_domain not in seed_domains and r_domain != ""
        r["cross_domain"] = is_cross
        if apply_cross_weight and is_cross:
            r["score"] = round(r["score"] * CROSS_DOMAIN_WEIGHT_BOOST, 3)

    walk.sort(key=lambda x: -x["score"])
    return walk[:GRAPH_TOP_K_NEIGHBORS]


def _count_cross_concepts(related: List[Dict]) -> int:
    """统计Top-K中跨域概念数量"""
    return sum(1 for r in related if r.get("cross_domain"))


def _check_cross_in_expected(related: List[Dict], expected_cross: List[str]) -> List[str]:
    """检查跨域概念是否命中期望列表"""
    return [r["concept"] for r in related if r.get("cross_domain") and r["concept"] in expected_cross]


def run_keyword_match(concepts: List[str], descriptions: List[str], query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """纯关键词匹配"""
    results = []
    query_lower = query.lower()
    for concept in sorted(concepts, key=len, reverse=True):
        if concept.lower() in query_lower:
            results.append((concept, 0.8))
    for i, desc in enumerate(descriptions):
        if len(results) >= top_k:
            break
        concept = concepts[i]
        if concept in [r[0] for r in results]:
            continue
        desc_keywords = desc.replace("，", " ").replace("、", " ").replace("。", " ").split()
        for kw in desc_keywords:
            if len(kw) >= 2 and kw in query_lower:
                results.append((concept, 0.5))
                break
    return results[:top_k]


def run_benchmark() -> Dict:
    """运行三组对比实验"""
    kg = KnowledgeGraph()
    concepts = list(kg.graph.nodes)
    descriptions = [
        kg.get_node_info(c).get("desc", c) if kg.get_node_info(c) else c
        for c in concepts
    ]

    # 初始化Embedding匹配器
    embedding_matcher = EmbeddingMatcher(concepts, descriptions)
    has_embedding = embedding_matcher.ensure_loaded()

    results = []
    # 三组统计
    baseline_stats = {"cross_count": 0, "cross_in_expected": 0, "total_cross": 0}
    weighted_stats = {"cross_count": 0, "cross_in_expected": 0, "total_cross": 0}
    emb_stats = {"cross_count": 0, "cross_in_expected": 0, "total_cross": 0, "seed_recall": 0, "total_seeds": 0}

    for test in TEST_QUERIES:
        query = test["query"]
        expected_seeds = test["expected_seeds"]
        expected_cross = test["expected_cross"]
        category = test["category"]

        # 关键词提取种子
        kw_matches = run_keyword_match(concepts, descriptions, query, top_k=5)
        kw_seeds = [c for c, _ in kw_matches]

        # === 组1: 无跨域加权基线 ===
        baseline_related = _do_walk(kg, kw_seeds, apply_cross_weight=False)
        baseline_cross_count = _count_cross_concepts(baseline_related)
        baseline_cross_hit = _check_cross_in_expected(baseline_related, expected_cross)

        # === 组2: 跨域加权（核心创新）===
        weighted_related = _do_walk(kg, kw_seeds, apply_cross_weight=True)
        weighted_cross_count = _count_cross_concepts(weighted_related)
        weighted_cross_hit = _check_cross_in_expected(weighted_related, expected_cross)

        # === 组3: 跨域加权 + Embedding ===
        emb_seeds = []
        emb_related = []
        emb_cross_count = 0
        emb_cross_hit = []
        if has_embedding:
            emb_matches = embedding_matcher.match(query, top_k=5)
            emb_seeds = [c for c, _ in emb_matches]
            emb_seeds_found = [c for c in emb_seeds if c in expected_seeds]
            emb_related = _do_walk(kg, emb_seeds, apply_cross_weight=True)
            emb_cross_count = _count_cross_concepts(emb_related)
            emb_cross_hit = _check_cross_in_expected(emb_related, expected_cross)

            emb_stats["seed_recall"] += len(emb_seeds_found)
            emb_stats["total_seeds"] += len(expected_seeds)

        # 统计
        baseline_stats["cross_count"] += baseline_cross_count
        baseline_stats["cross_in_expected"] += len(baseline_cross_hit)
        baseline_stats["total_cross"] += len(expected_cross)

        weighted_stats["cross_count"] += weighted_cross_count
        weighted_stats["cross_in_expected"] += len(weighted_cross_hit)
        weighted_stats["total_cross"] += len(expected_cross)

        if has_embedding:
            emb_stats["cross_count"] += emb_cross_count
            emb_stats["cross_in_expected"] += len(emb_cross_hit)
            emb_stats["total_cross"] += len(expected_cross)

        results.append({
            "category": category,
            "query": query,
            "seeds": kw_seeds,
            "baseline_cross": [r["concept"] for r in baseline_related if r.get("cross_domain")][:3],
            "baseline_cross_hit": baseline_cross_hit,
            "weighted_cross": [r["concept"] for r in weighted_related if r.get("cross_domain")][:3],
            "weighted_cross_hit": weighted_cross_hit,
            "emb_seeds": emb_seeds if has_embedding else [],
            "emb_cross": [r["concept"] for r in emb_related if r.get("cross_domain")][:3] if has_embedding else [],
            "emb_cross_hit": emb_cross_hit if has_embedding else [],
        })

    total_queries = len(TEST_QUERIES)

    summary = {
        "graph_stats": kg.stats(),
        "baseline_no_weight": {
            "avg_cross_per_query": round(baseline_stats["cross_count"] / total_queries, 1),
            "cross_domain_recall": round(baseline_stats["cross_in_expected"] / max(baseline_stats["total_cross"], 1) * 100, 1),
        },
        "weighted_1_5x": {
            "avg_cross_per_query": round(weighted_stats["cross_count"] / total_queries, 1),
            "cross_domain_recall": round(weighted_stats["cross_in_expected"] / max(weighted_stats["total_cross"], 1) * 100, 1),
        },
        "weighted_plus_embedding": {
            "available": has_embedding,
            "avg_cross_per_query": round(emb_stats["cross_count"] / total_queries, 1) if has_embedding else 0,
            "cross_domain_recall": round(emb_stats["cross_in_expected"] / max(emb_stats["total_cross"], 1) * 100, 1) if has_embedding else 0,
            "seed_recall": round(emb_stats["seed_recall"] / max(emb_stats["total_seeds"], 1) * 100, 1) if has_embedding else 0,
        },
        "details": results
    }

    return summary


def print_benchmark_report(summary: Dict):
    """打印格式化的实验报告"""
    print("\n" + "=" * 70)
    print("📊 跨域联想GraphRAG 对比实验报告")
    print("=" * 70)

    gs = summary["graph_stats"]
    print(f"\n📐 图谱规模: {gs['nodes']}节点 / {gs['edges']}边 / {gs['cross_domain_edges']}跨域边")
    print(f"   领域分布: {gs['domains']}")

    b = summary["baseline_no_weight"]
    w = summary["weighted_1_5x"]
    e = summary["weighted_plus_embedding"]

    print(f"\n{'─' * 70}")
    print(f"{'指标':<24} {'无加权基线':<16} {'跨域加权1.5x':<16} {'加权+Embedding':<16}")
    print(f"{'─' * 70}")

    avg_b = b["avg_cross_per_query"]
    avg_w = w["avg_cross_per_query"]
    avg_e = e["avg_cross_per_query"] if e["available"] else "-"
    print(f"{'平均跨域概念数/查询':<24} {avg_b:<16} {avg_w:<16} {avg_e}")

    recall_b = b["cross_domain_recall"]
    recall_w = w["cross_domain_recall"]
    recall_e = e["cross_domain_recall"] if e["available"] else "-"
    print(f"{'跨域概念召回率':<24} {recall_b}%{'':<13} {recall_w}%{'':<13} {recall_e}")

    if e["available"]:
        print(f"{'种子概念召回率':<24} {'─':<16} {'─':<16} {e['seed_recall']}%")

    # 加权提升幅度
    if recall_b > 0:
        boost = round((recall_w - recall_b) / recall_b * 100, 1)
        print(f"\n{'🔥 跨域加权提升幅度':<24} +{boost}% (从{recall_b}% → {recall_w}%)")
    elif recall_w > 0:
        print(f"\n{'🔥 跨域加权效果':<24} 从0% → {recall_w}%")

    print(f"{'─' * 70}")

    print(f"\n📋 逐查询详情:")
    for d in summary["details"]:
        print(f"\n  [{d['category']}] {d['query']}")
        print(f"    无加权 → 跨域: {d['baseline_cross']}  命中: {d['baseline_cross_hit']}")
        print(f"    加  权 → 跨域: {d['weighted_cross']}  命中: {d['weighted_cross_hit']}")
        if summary["weighted_plus_embedding"]["available"]:
            print(f"    +Embed → 种子: {d['emb_seeds']}  跨域: {d['emb_cross']}  命中: {d['emb_cross_hit']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_benchmark()
    print_benchmark_report(summary)

    # 保存JSON结果
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n💾 结果已保存到 benchmark_result.json")

"""
联想式知识图谱 - 基于NetworkX实现
从graph_data.py加载120+节点扩展图谱
支持动态增删改、加权BFS游走、语义过滤
"""
import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
import json
from graph_data import NODES, EDGES


class KnowledgeGraph:
    """知识图谱：概念节点 + 关联边 + BFS游走"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """从graph_data.py构建知识网络"""
        # 添加节点
        for node_id, attrs in NODES.items():
            self.graph.add_node(node_id, **attrs)

        # 添加边 + 自动生成反向边
        for edge in EDGES:
            if len(edge) == 4:
                src, dst, relation, weight = edge
            else:
                src, dst, relation, weight = edge[0], edge[1], edge[2], 0.5

            self.graph.add_edge(src, dst, relation=relation, weight=weight)
            # 反向边（权重降低）
            if not self.graph.has_edge(dst, src):
                self.graph.add_edge(dst, src, relation=f"被{relation}", weight=weight * 0.6)

    def add_node(self, name: str, domain: str = "", desc: str = ""):
        """添加节点"""
        self.graph.add_node(name, domain=domain, desc=desc)

    def add_edge(self, src: str, dst: str, relation: str = "", weight: float = 0.5):
        """添加关联边"""
        self.graph.add_edge(src, dst, relation=relation, weight=weight)

    def get_node_info(self, name: str) -> Optional[Dict]:
        """获取节点信息"""
        if name in self.graph:
            return dict(self.graph.nodes[name])
        return None

    def get_neighbors(self, node: str) -> List[Tuple[str, Dict]]:
        """获取邻居节点及边属性"""
        if node not in self.graph:
            return []
        return [(n, self.graph[node][n]) for n in self.graph.neighbors(node)]

    def weighted_walk(self, start_nodes: List[str], depth: int = 2, top_k: int = 5) -> List[Dict]:
        """
        加权BFS游走：从起始节点出发，沿边权重游走
        返回关联概念列表，按累积权重排序
        """
        visited = {}  # node -> cumulative_weight

        # 初始化起始节点
        frontier = {n: 1.0 for n in start_nodes if n in self.graph}
        for n in frontier:
            visited[n] = 1.0

        for _ in range(depth):
            next_frontier = {}
            for node, cum_weight in frontier.items():
                for neighbor, edge_data in self.get_neighbors(node):
                    new_weight = cum_weight * edge_data.get("weight", 0.5)
                    if neighbor not in visited or new_weight > visited.get(neighbor, 0):
                        visited[neighbor] = new_weight
                        if neighbor not in next_frontier or new_weight > next_frontier[neighbor]:
                            next_frontier[neighbor] = new_weight
            frontier = next_frontier

        # 移除起始节点，按权重排序
        results = []
        for node, weight in sorted(visited.items(), key=lambda x: -x[1]):
            if node not in start_nodes:
                info = self.get_node_info(node)
                results.append({
                    "concept": node,
                    "score": round(weight, 3),
                    "domain": info.get("domain", "") if info else "",
                    "desc": info.get("desc", "") if info else ""
                })
                if len(results) >= top_k:
                    break

        return results

    def get_walk_path(self, start: str, end: str) -> List[str]:
        """获取两个节点之间的最短路径（用于可视化）"""
        try:
            return nx.shortest_path(self.graph, start, end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_all_nodes(self) -> List[Dict]:
        """获取所有节点（可视化用）"""
        result = []
        for node in self.graph.nodes:
            info = self.get_node_info(node)
            result.append({"id": node, **(info or {})})
        return result

    def get_all_edges(self) -> List[Dict]:
        """获取所有边（可视化用）"""
        result = []
        for src, dst, data in self.graph.edges(data=True):
            result.append({
                "source": src, "target": dst,
                "relation": data.get("relation", ""),
                "weight": data.get("weight", 0.5)
            })
        return result

    def get_cross_domain_edges(self) -> List[Dict]:
        """获取所有跨域边（用于高亮展示）"""
        result = []
        for src, dst, data in self.graph.edges(data=True):
            src_info = self.get_node_info(src)
            dst_info = self.get_node_info(dst)
            if src_info and dst_info and src_info.get("domain") != dst_info.get("domain"):
                result.append({
                    "source": src, "target": dst,
                    "relation": data.get("relation", ""),
                    "weight": data.get("weight", 0.5),
                    "src_domain": src_info.get("domain", ""),
                    "dst_domain": dst_info.get("domain", "")
                })
        return result

    def stats(self) -> Dict:
        """图谱统计信息"""
        domains = {}
        for node in self.graph.nodes:
            info = self.get_node_info(node)
            d = info.get("domain", "未分类") if info else "未分类"
            domains[d] = domains.get(d, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "domains": domains,
            "cross_domain_edges": len(self.get_cross_domain_edges())
        }


# 测试
if __name__ == "__main__":
    kg = KnowledgeGraph()
    stats = kg.stats()
    print(f"图谱统计: {stats['nodes']}节点, {stats['edges']}边")
    print(f"领域分布: {stats['domains']}")
    print(f"跨域边: {stats['cross_domain_edges']}")

    # 从C++出发游走
    results = kg.weighted_walk(["C++"], depth=2, top_k=8)
    print(f"\n从C++游走(depth=2):")
    for r in results:
        cross = "🌐" if r["domain"] not in ["编程", ""] else "📌"
        print(f"  {cross} {r['concept']} (score={r['score']}, domain={r['domain']})")

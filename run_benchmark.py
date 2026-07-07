"""重新运行benchmark并验证结果"""
import sys
import os
sys.path.insert(0, r"F:\newai2")
os.chdir(r"F:\newai2")

from benchmark import run_benchmark, print_benchmark_report
import json

print("正在运行benchmark...")
summary = run_benchmark()
print_benchmark_report(summary)

# 保存结果
with open("benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n结果已保存到 benchmark_result.json")

# 输出关键数据摘要（供简历使用）
b = summary["baseline_no_weight"]
w = summary["weighted_1_5x"]
e = summary["weighted_plus_embedding"]
gs = summary["graph_stats"]

recall_boost = round((w["cross_domain_recall"] - b["cross_domain_recall"]) / b["cross_domain_recall"] * 100, 1)
cross_boost = round((w["avg_cross_per_query"] - b["avg_cross_per_query"]) / b["avg_cross_per_query"] * 100, 1)

print("\n" + "=" * 50)
print("📋 简历关键数据摘要")
print("=" * 50)
print(f"图谱规模: {gs['nodes']}节点/{len(gs['domains'])}领域/{gs['cross_domain_edges']}跨域边")
print(f"跨域召回率: {b['cross_domain_recall']}% → {w['cross_domain_recall']}% (+{recall_boost}%)")
print(f"每查询跨域概念: {b['avg_cross_per_query']} → {w['avg_cross_per_query']} (+{cross_boost}%)")
if e["available"]:
    print(f"种子召回率: {e['seed_recall']}%")
print("=" * 50)

"""
Embedding语义匹配器 - 替代关键词匹配
基于sentence-transformers实现概念语义检索
支持自动降级：embedding不可用时回退到关键词匹配
懒加载：启动时不下载模型，首次提问时才加载
"""
import logging
from typing import List, Tuple, Optional
from config import EMBEDDING_MODE, EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class EmbeddingMatcher:
    """语义概念匹配器：用Embedding相似度替代关键词匹配"""

    def __init__(self, concepts: List[str], descriptions: List[str]):
        self.concepts = concepts
        self.descriptions = descriptions
        self._model = None
        self._embeddings = None
        self._available = False
        self._mode = "keyword"  # keyword / embedding
        self._load_attempted = False

        # EMBEDDING_MODE=keyword时直接跳过，auto/embedding时延迟到首次match再加载
        if EMBEDDING_MODE == "keyword":
            self._load_attempted = True  # 标记已决定，不再尝试
            logger.info("🔍 匹配模式: keyword（手动指定）")

    def _try_load_model(self):
        """尝试加载embedding模型，失败则降级到关键词模式"""
        try:
            import os
            # 自动设置HuggingFace镜像（国内网络）
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                logger.info("🌐 已设置HF镜像: hf-mirror.com")

            from sentence_transformers import SentenceTransformer
            logger.info(f"🔄 加载Embedding模型: {EMBEDDING_MODEL}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
            # 预计算所有概念的embedding
            texts = [f"{c}: {d}" for c, d in zip(self.concepts, self.descriptions)]
            self._embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._available = True
            self._mode = "embedding"
            logger.info(f"✅ Embedding模型加载完成，{len(self.concepts)}个概念已编码")
        except ImportError:
            logger.warning("⚠️ sentence-transformers未安装，使用关键词匹配模式")
            logger.warning("   安装命令: pip install sentence-transformers")
            self._mode = "keyword"
        except Exception as e:
            logger.warning(f"⚠️ Embedding模型加载失败({e})，使用关键词匹配模式")
            self._mode = "keyword"

    @property
    def mode(self) -> str:
        return self._mode

    def ensure_loaded(self) -> bool:
        """主动触发模型加载（用于benchmark等需要预加载的场景）"""
        if not self._load_attempted:
            self._load_attempted = True
            self._try_load_model()
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    def match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        语义匹配：返回与query最相关的top_k个概念
        首次调用时懒加载Embedding模型，加载失败自动降级
        返回: [(concept_name, similarity_score), ...]
        """
        # 懒加载：首次match时才尝试加载模型
        if not self._load_attempted:
            self._load_attempted = True
            self._try_load_model()

        if self._available:
            return self._embedding_match(query, top_k)
        else:
            return self._keyword_match(query, top_k)

    def _embedding_match(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Embedding语义匹配"""
        import numpy as np
        query_embedding = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        # 计算余弦相似度（已归一化，直接点积）
        similarities = (self._embeddings @ query_embedding.T).flatten()
        # 过滤低相似度
        top_indices = similarities.argsort()[-top_k * 2:][::-1]  # 多取一些用于过滤
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= EMBEDDING_SIMILARITY_THRESHOLD:
                results.append((self.concepts[idx], score))
            if len(results) >= top_k:
                break
        return results

    def _keyword_match(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """关键词匹配（降级方案）"""
        results = []
        query_lower = query.lower()
        # 按概念长度降序匹配
        for concept in sorted(self.concepts, key=len, reverse=True):
            if concept.lower() in query_lower:
                # 关键词匹配给固定0.8分
                results.append((concept, 0.8))
        # 模糊匹配：描述中的关键词
        for i, desc in enumerate(self.descriptions):
            if len(results) >= top_k:
                break
            concept = self.concepts[i]
            if concept in [r[0] for r in results]:
                continue
            # 描述中的关键词匹配
            desc_keywords = desc.replace("，", " ").replace("、", " ").replace("。", " ").split()
            for kw in desc_keywords:
                if len(kw) >= 2 and kw in query_lower:
                    results.append((concept, 0.5))
                    break
        return results[:top_k]

    def get_similarity(self, concept1: str, concept2: str) -> float:
        """获取两个概念之间的语义相似度（仅embedding模式）"""
        if not self._available:
            return 0.0
        import numpy as np
        idx1 = self.concepts.index(concept1) if concept1 in self.concepts else -1
        idx2 = self.concepts.index(concept2) if concept2 in self.concepts else -1
        if idx1 < 0 or idx2 < 0:
            return 0.0
        return float(np.dot(self._embeddings[idx1], self._embeddings[idx2]))

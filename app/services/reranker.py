"""
Reranker 重排序模块

提供抽象接口和基础实现，用于对初步检索结果进行精细化排序。
后续可接入 Cross-Encoder 等深度学习模型。
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import settings


@dataclass
class RerankerInput:
    """Reranker 输入数据结构"""
    id: int
    content: str
    document_id: int
    chunk_index: int
    page_number: Optional[int] = None
    metadata: dict = None
    # 原始检索分数
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class RerankerOutput:
    """Reranker 输出数据结构"""
    id: int
    content: str
    document_id: int
    chunk_index: int
    page_number: Optional[int] = None
    metadata: dict = None
    # 重排序后的分数
    final_score: float = 0.0
    # 各阶段分数明细（用于调试）
    score_breakdown: dict = None


class BaseReranker(ABC):
    """
    Reranker 抽象基类

    所有具体 Reranker 实现必须继承此类。
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[RerankerInput],
        top_k: int = 5
    ) -> List[RerankerOutput]:
        """
        对候选结果进行重排序

        Args:
            query: 查询字符串
            candidates: 候选结果列表
            top_k: 返回 Top-K 结果

        Returns:
            重排序后的结果列表
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查 Reranker 是否可用

        Returns:
            True if available, False otherwise
        """
        pass


class NullReranker(BaseReranker):
    """
    空 Reranker（默认实现）

    不做任何重排序，直接按 RRF 分数透传。
    用于向后兼容和禁用重排序的场景。
    """

    async def rerank(
        self,
        query: str,
        candidates: List[RerankerInput],
        top_k: int = 5
    ) -> List[RerankerOutput]:
        """透传结果，不进行重排序"""
        outputs = []
        for candidate in candidates[:top_k]:
            outputs.append(RerankerOutput(
                id=candidate.id,
                content=candidate.content,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                page_number=candidate.page_number,
                metadata=candidate.metadata or {},
                final_score=candidate.rrf_score,
                score_breakdown={
                    "rrf_score": candidate.rrf_score,
                    "vector_score": candidate.vector_score,
                    "keyword_score": candidate.keyword_score,
                    "rerank_score": 0.0
                }
            ))
        return outputs

    def is_available(self) -> bool:
        return True


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker 完整实现

    使用 sentence-transformers 的 CrossEncoder 对 Query-Document Pair 进行精细打分。
    默认加载 'BAAI/bge-reranker-v2-m3' 模型。

    性能优化：
    - 使用 asyncio.to_thread 将同步模型推理卸载到后台线程
    - 使用批量推理减少开销
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._initialized = False

    def _load_model(self):
        """懒加载 CrossEncoder 模型"""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            print(f"🔄 加载 CrossEncoder 模型: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=settings.EMBEDDING_DEVICE
            )
            print(f"✅ CrossEncoder 模型加载完成")
            self._initialized = True
        except Exception as e:
            print(f"❌ CrossEncoder 模型加载失败: {e}")
            self._initialized = False

    def _sync_predict(self, pairs: List[tuple]) -> List[float]:
        """
        同步的模型推理方法（在后台线程中执行）

        Args:
            pairs: [(query, doc), ...] 列表

        Returns:
            相似度分数列表（Sigmoid 概率值，0-1 范围）
        """
        if not self._model:
            return [0.0] * len(pairs)

        # 批量预测，返回的是相关性分数（已应用 sigmoid）
        scores = self._model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False
        )

        # 确保返回 Python float 列表
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        candidates: List[RerankerInput],
        top_k: int = 5
    ) -> List[RerankerOutput]:
        """
        使用 Cross-Encoder 进行重排序

        Args:
            query: 查询字符串
            candidates: 候选结果列表（来自 RRF 融合）
            top_k: 返回 Top-K 结果

        Returns:
            按 Reranker 分数排序的结果列表
        """
        if not self.is_available():
            print("[Reranker] 模型不可用，回退到 NullReranker")
            return await NullReranker().rerank(query, candidates, top_k)

        if not candidates:
            return []

        # 构建 Query-Document Pairs
        pairs = [(query, c.content) for c in candidates]

        print(f"[Reranker] 对 {len(candidates)} 个候选结果进行重排序...")

        # 🚨 关键：使用 asyncio.to_thread 将同步模型推理卸载到后台线程
        # 避免阻塞 FastAPI 的事件循环
        rerank_scores = await asyncio.to_thread(self._sync_predict, pairs)

        # 构建输出列表，使用 Reranker 分数作为最终分数
        outputs = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            outputs.append(RerankerOutput(
                id=candidate.id,
                content=candidate.content,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                page_number=candidate.page_number,
                metadata=candidate.metadata or {},
                # 使用 Reranker 分数覆盖 RRF 分数（0-1 范围的 Sigmoid 概率）
                final_score=rerank_score,
                score_breakdown={
                    "rerank_score": rerank_score,  # CrossEncoder 分数（主要）
                    "rrf_score": candidate.rrf_score,  # RRF 融合分数（参考）
                    "vector_score": candidate.vector_score,
                    "keyword_score": candidate.keyword_score
                }
            ))

        # 按 Reranker 分数从高到低排序
        outputs.sort(key=lambda x: x.final_score, reverse=True)

        print(f"[Reranker] 重排序完成，Top-1 分数: {outputs[0].final_score:.4f}")

        return outputs[:top_k]

    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not self._initialized:
            self._load_model()
        return self._initialized and self._model is not None


# Reranker 工厂
def create_reranker(reranker_type: str = None) -> BaseReranker:
    """
    创建 Reranker 实例

    Args:
        reranker_type: Reranker 类型（null, cross_encoder）

    Returns:
        Reranker 实例
    """
    reranker_type = reranker_type or settings.RERANKER_TYPE

    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    else:
        return NullReranker()


# 全局 Reranker 实例（懒加载）
_reranker_instance: Optional[BaseReranker] = None


def get_reranker() -> BaseReranker:
    """获取全局 Reranker 实例"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = create_reranker()
    return _reranker_instance


def reset_reranker():
    """重置 Reranker 实例（用于配置更新后）"""
    global _reranker_instance
    _reranker_instance = None

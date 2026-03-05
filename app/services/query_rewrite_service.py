"""
Query Rewrite 查询重写服务

基于 LLM 的查询语义扩充，解决短 Query 缺乏语义信息的问题。
使用严格格式控制，确保只输出重写后的字符串。
"""

import asyncio
from typing import Optional

from app.config import settings
from app.models.chat import ChatCompletionRequest, ChatMessage
from app.services.llm_service import vllm_client, VLLMClient


# 严苛的 System Prompt，强制 LLM 只输出重写结果
QUERY_REWRITE_PROMPT = """你是一个查询重写助手。你的任务是将用户的简短查询扩充为带有项目上下文的精准长句。

规则：
1. 分析用户查询的意图，识别其中的技术术语或项目专有名词
2. 将短查询扩充为包含上下文、具体概念的完整长句
3. 直接输出重写后的字符串，绝对不要包含任何解释、前缀或后缀
4. 禁止输出："好的"、"改写结果"、"这是"、"重写后"等任何说明性文字

示例：
输入：React
输出：Nexus-Agent 项目中 ReAct 智能体编排引擎的架构设计、核心原理与工作流

输入：架构设计
输出：Nexus-Agent 项目的系统架构总览，包括 Infra 层、State 层、Knowledge 层和 Action 层的设计原则

输入：分块
输出：Nexus-Agent 知识库模块中的文档语义切分策略、Chunking 算法与文本分块实现

输入：向量检索
输出：Nexus-Agent 项目中基于 pgvector 的向量相似度检索、Embedding 生成与混合检索实现

直接输出结果，不要解释。"""


class QueryRewriteService:
    """
    查询重写服务

    使用 VLLMClient 异步调用 LLM 进行查询语义扩充。
    """

    def __init__(self, llm_client: Optional[VLLMClient] = None):
        """
        初始化查询重写服务

        Args:
            llm_client: VLLM 客户端实例，默认使用全局实例
        """
        self.llm = llm_client or vllm_client

    async def rewrite(self, query: str, context_hint: str = "Nexus-Agent") -> str:
        """
        重写用户查询

        Args:
            query: 用户原始查询（短句）
            context_hint: 上下文提示，帮助 LLM 理解项目背景

        Returns:
            重写后的完整查询字符串（绝对无解释性文字）
        """
        if not query or len(query.strip()) == 0:
            return query

        # 如果查询已经很长，可能不需要重写
        if len(query) > 50 and ' ' in query:
            return query

        # 构建请求
        messages = [
            ChatMessage(role="system", content=QUERY_REWRITE_PROMPT),
            ChatMessage(role="user", content=f"输入：{query}\n输出：")
        ]

        request = ChatCompletionRequest(
            messages=messages,
            temperature=0.1,  # 低温度确保输出稳定
            max_tokens=256,
            stream=False
        )

        try:
            # 异步调用 LLM
            response = await self.llm.chat_completion(request)

            if not response or not response.choices:
                return query

            rewritten = response.choices[0].message.content.strip()

            # 后处理：去除可能的残留前缀/后缀
            rewritten = self._clean_output(rewritten)

            # Task 3: 显微镜日志 - 监控 Query Rewrite 效果
            print(f"==========\n[原Query]: {query}\n[重写后Query]: {rewritten}\n==========")

            # 如果重写结果异常（空或太短），回退到原始查询
            if len(rewritten) < len(query) * 0.8:
                return query

            return rewritten

        except Exception as e:
            # 重写失败时返回原始查询，不阻断主流程
            print(f"[QueryRewrite] 重写失败，使用原始查询: {e}")
            return query

    def _clean_output(self, text: str) -> str:
        """
        清洗 LLM 输出，去除可能的解释性文字
        """
        if not text:
            return text

        # 去除常见前缀
        prefixes_to_remove = [
            r"^(?:好的|好的，|好的，?)",
            r"^(?:这是|这是改写|这是重写|以下是|改写后|重写后)",
            r"^(?:改写结果|重写结果|结果|输出)[:：]\s*",
            r"^(?:查询|新查询)[:：]\s*",
            r"^['\"]+",  # 去除开头的引号
            r"^[-\*•]\s*",
        ]

        import re
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, "", text.strip(), flags=re.IGNORECASE)

        # 去除常见后缀
        suffixes_to_remove = [
            r"['\"]+\s*$",  # 去除结尾的引号
            r"\s*(?:这样|即可|就行|就可以了)\s*$",
            r"\s*（[^）]*）\s*$",  # 括号注释
            r"\s*\([^)]*\)\s*$",  # 英文括号注释
        ]

        for pattern in suffixes_to_remove:
            text = re.sub(pattern, "", text.strip())

        return text.strip()


# 全局查询重写服务实例
query_rewrite_service = QueryRewriteService()


async def rewrite_query(query: str, context_hint: str = "Nexus-Agent") -> str:
    """
    便捷函数：重写查询

    Args:
        query: 原始查询
        context_hint: 上下文提示

    Returns:
        重写后的查询
    """
    return await query_rewrite_service.rewrite(query, context_hint)

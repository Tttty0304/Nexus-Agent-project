"""
记忆服务：Redis 短期记忆 + PostgreSQL长期记忆

特性：
- Redis List 结构存储会话上下文
- 自动过期清理
- Token 预算控制
"""

import json
import time
from typing import List, Optional

from app.config import settings
from app.core.cache import cache


class MemoryService:
    """记忆服务"""
    
    def __init__(self, max_rounds: int = None):
        self.max_rounds = max_rounds or settings.MAX_SESSION_ROUNDS
    
    def _key(self, session_id: str) -> str:
        """生成 Redis Key"""
        return f"nexus:session:{session_id}"
    
    async def add_message(self, session_id: str, role: str, content: str):
        """
        添加消息到会话上下文
        
        使用 Redis Pipeline 批量操作：
        1. LPUSH 添加消息
        2. LTRIM 截断保留最近 N 轮
        3. EXPIRE 设置过期时间
        """
        key = self._key(session_id)
        message = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time()
        })
        
        pipe = cache.client.pipeline()
        pipe.lpush(key, message)
        pipe.ltrim(key, 0, self.max_rounds * 2 - 1)
        pipe.expire(key, settings.SESSION_TTL_DAYS * 24 * 3600)
        await pipe.execute()
    
    async def get_context(self, session_id: str) -> List[dict]:
        """
        获取会话上下文
        
        Returns:
            按时间正序排列的消息列表
        """
        key = self._key(session_id)
        raw = await cache.client.lrange(key, 0, -1)
        
        messages = []
        for m in raw:
            try:
                messages.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        
        return list(reversed(messages))
    
    async def clear_context(self, session_id: str):
        """清除会话上下文"""
        key = self._key(session_id)
        await cache.client.delete(key)
    
    async def get_formatted_history(
        self,
        session_id: str,
        max_tokens: int = 4000
    ) -> List[dict]:
        """
        获取格式化的对话历史，用于 LLM 上下文
        
        Args:
            session_id: 会话ID
            max_tokens: 最大token数限制
        
        Returns:
            OpenAI 格式的消息列表
        """
        messages = await self.get_context(session_id)
        
        # 简单的token预算控制（实际应使用 tiktoken）
        formatted = []
        total_len = 0
        
        for msg in reversed(messages):
            content_len = len(msg["content"])
            if total_len + content_len > max_tokens:
                break
            formatted.insert(0, {
                "role": msg["role"],
                "content": msg["content"]
            })
            total_len += content_len
        
        return formatted
    
    async def get_session_stats(self, session_id: str) -> dict:
        """获取会话统计信息"""
        key = self._key(session_id)
        length = await cache.client.llen(key)
        ttl = await cache.client.ttl(key)
        
        return {
            "message_count": length,
            "ttl_seconds": ttl,
            "ttl_days": ttl // 86400 if ttl > 0 else 0
        }


class TokenBudget:
    """Token 预算管理"""
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.reserved = 2048  # 预留响应空间
    
    def truncate(
        self,
        messages: List[dict],
        system_msgs: List[dict]
    ) -> List[dict]:
        """
        滑动窗口截断，优先保留系统提示和最近消息
        
        Args:
            messages: 历史消息
            system_msgs: 系统消息（优先保留）
        
        Returns:
            截断后的消息列表
        """
        # 计算固定部分（系统消息）的token数（估算）
        system_tokens = sum(len(m["content"]) // 4 for m in system_msgs)
        available = self.max_tokens - self.reserved - system_tokens
        
        # 从后向前累加，直到预算耗尽
        truncated, current = [], 0
        for msg in reversed(messages):
            tokens = len(msg["content"]) // 4  # 粗略估算
            if current + tokens > available:
                break
            truncated.insert(0, msg)
            current += tokens
        
        return system_msgs + truncated

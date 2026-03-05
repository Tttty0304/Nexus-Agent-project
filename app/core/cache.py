"""
Redis 缓存模块（满血版本）

特性：
- 连接池管理
- 令牌桶限流
- 会话上下文管理
- Pipeline 支持
"""

import asyncio
import json
import time
from typing import List, Optional, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.config import settings


class RedisCache:
    """Redis 缓存客户端"""
    
    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """建立 Redis 连接"""
        self._pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_POOL_MAX_CONNECTIONS,
            decode_responses=True
        )
        self._client = redis.Redis(connection_pool=self._pool)
        
        # 测试连接
        await self._client.ping()
        print(f"✅ Redis 已连接: {settings.REDIS_URL}")
    
    async def disconnect(self):
        """关闭 Redis 连接"""
        if self._pool:
            await self._pool.disconnect()
            print("✅ Redis 连接已关闭")
    
    @property
    def client(self) -> redis.Redis:
        """获取 Redis 客户端"""
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    async def ping(self) -> bool:
        """检查连接状态"""
        try:
            return await self._client.ping()
        except Exception:
            return False
    
    # ========== 会话上下文方法 ==========
    
    async def add_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        max_rounds: int = 10
    ):
        """
        添加会话消息到 Redis
        
        使用 Pipeline 批量操作：
        1. LPUSH 添加消息
        2. LTRIM 截断保留最近 N 轮
        3. EXPIRE 设置过期时间
        """
        key = f"nexus:session:{session_id}"
        message = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time()
        })
        
        pipe = self._client.pipeline()
        pipe.lpush(key, message)
        pipe.ltrim(key, 0, max_rounds * 2 - 1)  # 保留最近 N 轮（每轮2条消息）
        pipe.expire(key, settings.SESSION_TTL_DAYS * 86400)
        await pipe.execute()
    
    async def get_context(self, session_id: str) -> List[dict]:
        """获取会话上下文（按时间顺序）"""
        key = f"nexus:session:{session_id}"
        messages = await self._client.lrange(key, 0, -1)
        
        # 解析并反转顺序（LRANGE 返回的是逆序）
        result = []
        for msg in reversed(messages):
            try:
                result.append(json.loads(msg))
            except json.JSONDecodeError:
                continue
        return result
    
    async def clear_session(self, session_id: str):
        """清除会话"""
        key = f"nexus:session:{session_id}"
        await self._client.delete(key)
    
    # ========== 限流方法（令牌桶算法） ==========
    
    async def rate_limit_check(
        self,
        key: str,
        rate: float = None,
        capacity: int = None,
        burst: int = None
    ) -> bool:
        """
        令牌桶限流检查
        
        Args:
            key: 限流键（如用户ID或IP）
            rate: 每秒填充速率（默认从配置读取）
            capacity: 桶容量（默认从配置读取）
            burst: 突发容量
        
        Returns:
            True: 允许请求
            False: 限流
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True
        
        rate = rate or settings.RATE_LIMIT_REQUESTS_PER_MINUTE / 60
        capacity = capacity or settings.RATE_LIMIT_CAPACITY
        burst = burst or settings.RATE_LIMIT_BURST
        
        bucket_key = f"ratelimit:{key}"
        now = time.time()
        
        # Lua 脚本实现原子性令牌桶
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('hmget', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now
        
        -- 计算新令牌数
        local elapsed = now - last_update
        tokens = math.min(capacity, tokens + elapsed * rate)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('hmset', key, 'tokens', tokens, 'last_update', now)
            redis.call('expire', key, 60)
            return 1
        else
            redis.call('hmset', key, 'tokens', tokens, 'last_update', now)
            redis.call('expire', key, 60)
            return 0
        end
        """
        
        result = await self._client.eval(
            lua_script,
            1,  # num_keys
            bucket_key,
            rate,
            capacity,
            now
        )
        return bool(result)
    
    # ========== 通用缓存方法 ==========
    
    async def get(self, key: str) -> Optional[str]:
        """获取字符串值"""
        return await self._client.get(key)
    
    async def set(
        self,
        key: str,
        value: Union[str, bytes],
        expire: int = None
    ):
        """设置字符串值"""
        if expire:
            await self._client.setex(key, expire, value)
        else:
            await self._client.set(key, value)
    
    async def delete(self, key: str) -> int:
        """删除键"""
        return await self._client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self._client.exists(key) > 0
    
    async def ttl(self, key: str) -> int:
        """获取键剩余生存时间"""
        return await self._client.ttl(key)
    
    # ========== Pipeline 支持 ==========
    
    def pipeline(self):
        """获取 Pipeline 对象"""
        return self._client.pipeline()
    
    # ========== 分布式锁 ==========
    
    async def acquire_lock(
        self,
        lock_name: str,
        lock_timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: int = 5
    ) -> bool:
        """
        获取分布式锁
        
        Args:
            lock_name: 锁名称
            lock_timeout: 锁自动释放时间（秒）
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞等待超时时间
        
        Returns:
            True: 获取锁成功
            False: 获取锁失败
        """
        lock_key = f"lock:{lock_name}"
        identifier = f"{time.time()}-{id(self)}"
        
        if blocking:
            # 阻塞模式：使用 SET NX EX
            end_time = time.time() + blocking_timeout
            while time.time() < end_time:
                acquired = await self._client.set(
                    lock_key,
                    identifier,
                    nx=True,
                    ex=lock_timeout
                )
                if acquired:
                    return True
                await asyncio.sleep(0.1)
            return False
        else:
            # 非阻塞模式
            return await self._client.set(
                lock_key,
                identifier,
                nx=True,
                ex=lock_timeout
            )
    
    async def release_lock(self, lock_name: str) -> bool:
        """释放分布式锁"""
        lock_key = f"lock:{lock_name}"
        return await self._client.delete(lock_key) > 0


# 全局缓存实例
cache = RedisCache()


async def get_cache() -> RedisCache:
    """获取缓存实例"""
    return cache

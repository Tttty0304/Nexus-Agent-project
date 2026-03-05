"""
vLLM 客户端封装（修复 tool_choice 400 错误版）

错误原因分析：
1. ChatCompletionRequest 中 tool_choice 有默认值 "auto"
2. _build_payload 把这个默认值传给 vLLM
3. vLLM 规则：设置了 tool_choice 就必须设置 tools
4. 但请求中 tools 是 None，导致 400 错误

修复方案：
- 只有在 tools 不为 None 时，才添加 tool_choice 到请求体
"""

import json
from typing import AsyncIterator, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


class VLLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = (base_url or settings.VLLM_BASE_URL).rstrip('/')
        self.api_key = api_key or settings.VLLM_API_KEY
        self._client: httpx.AsyncClient = None
    
    async def connect(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(settings.VLLM_TIMEOUT, connect=5.0, read=30.0),
            http2=True
        )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
    
    def _get_model_name(self, request_model: str) -> str:
        """
        获取 vLLM 实际模型名称

        vLLM 中加载的模型 ID 可能是完整路径（如 /models/Qwen2-7B-Instruct-AWQ）
        而用户请求的是简单名称（如 Qwen2-7B-Instruct-AWQ）
        需要映射到正确的模型 ID
        """
        # 如果配置中指定了模型名称，优先使用
        if settings.VLLM_MODEL and settings.VLLM_MODEL != request_model:
            # 返回 vLLM 中实际注册的模型 ID
            return settings.VLLM_MODEL
        return request_model

    def _build_payload(self, request: ChatCompletionRequest) -> dict:
        """
        构建 vLLM 兼容的请求体

        关键修复：
        - 只有设置了 tools 时，才添加 tool_choice
        - vLLM 规则：tool_choice 和 tools 必须同时存在或同时不存在
        """
        # 使用正确的模型名称
        model_name = self._get_model_name(request.model)

        payload = {
            "model": model_name,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
        }
        
        # 基本参数
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        if request.stream is not None:
            payload["stream"] = request.stream
        
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        
        # ============================================
        # 关键修复：tools 和 tool_choice 必须配对
        # ============================================
        # 只有当用户明确提供了 tools 时，才添加 tool_choice
        if request.tools is not None and len(request.tools) > 0:
            payload["tools"] = request.tools
            # 只有提供了 tools，tool_choice 才有意义
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        # 如果 tools 为 None，则不添加 tool_choice
        # 即使 request.tool_choice 有默认值 "auto"，也不传给 vLLM
        
        return payload
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=lambda e: isinstance(e, (httpx.TimeoutException, httpx.ConnectError))
    )
    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncIterator[str]]:
        """调用 vLLM 聊天补全接口"""
        
        payload = self._build_payload(request)
        
        # 调试：打印请求体
        print(f"[DEBUG] vLLM Request: {json.dumps(payload, ensure_ascii=False)[:500]}")
        
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=payload,
                timeout=None if request.stream else settings.VLLM_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"[DEBUG] vLLM Error {response.status_code}: {response.text[:500]}")
            
            response.raise_for_status()
            
            if request.stream:
                return self._parse_stream(response.aiter_lines())
            
            return ChatCompletionResponse(**response.json())
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text[:500]
            except:
                pass
            
            if e.response.status_code == 429:
                raise RateLimitError("vLLM rate limited")
            elif e.response.status_code == 400:
                raise LLMError(f"vLLM Bad Request: {error_detail}")
            elif e.response.status_code >= 500:
                raise LLMUnavailableError(f"vLLM service error: {e.response.status_code}")
            raise LLMError(f"HTTP error: {e.response.status_code}, detail: {error_detail}")
    
    async def _parse_stream(self, line_iterator) -> AsyncIterator[str]:
        """解析 SSE 流式响应"""
        async for line in line_iterator:
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
    
    async def health_check(self) -> dict:
        """检查 vLLM 服务健康状态"""
        try:
            response = await self._client.get("/health", timeout=5.0)
            
            if response.status_code < 300:
                return {"status": "healthy", "detail": response.text or "OK"}
            else:
                return {"status": "degraded", "code": response.status_code}
                
        except httpx.ConnectError as e:
            return {"status": "unhealthy", "error": f"Connection failed: {str(e)}"}
        except httpx.TimeoutException:
            return {"status": "unhealthy", "error": "Connection timeout"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# 全局 vLLM 客户端实例
vllm_client = VLLMClient()

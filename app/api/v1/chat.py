"""
聊天 API：/v1/chat/*（满血版本）

功能：
- OpenAI 兼容的聊天补全接口
- 流式和非流式响应
- 会话历史保存
- RAG 增强检索
"""

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionUsage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ModelList,
    ModelInfo,
)
from app.models.user import User
from app.services.llm_service import vllm_client, RateLimitError, LLMUnavailableError
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService

router = APIRouter()


async def chat_stream_generator(
    request: ChatCompletionRequest,
    user: User,
    db: AsyncSession
) -> AsyncIterator[str]:
    """
    流式聊天响应生成器（支持多轮对话 + RAG 检索增强）
    """
    full_content = ""
    start_time = time.time()

    try:
        # 发送初始角色标记
        initial_chunk = ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=request.model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None
            )]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # 如果启用了 RAG，使用新的 generate_rag_response 方法
        if request.use_rag and len(request.messages) > 0:
            rag_service = RAGService()

            # 将 Pydantic 模型转换为 dict 列表，支持多轮对话上下文
            messages_dict = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            # 调用 RAG 流式生成（含检索 + System Prompt 注入）
            async for token in rag_service.generate_rag_response(
                session=db,
                messages=messages_dict,
                top_k=5,
                enable_query_rewrite=True
            ):
                full_content += token

                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        else:
            # 非 RAG 模式：直接调用 vLLM
            stream_request = request.model_copy(update={"stream": True})
            token_stream = await vllm_client.chat_completion(stream_request)

            async for token in token_stream:
                full_content += token

                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # 结束标记
        final_chunk = ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=request.model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        # 后台任务：保存会话历史
        if request.session_id:
            latency_ms = int((time.time() - start_time) * 1000)
            asyncio.create_task(_save_conversation_async(
                request.session_id,
                request.messages,
                full_content,
                user.id,
                latency_ms
            ))

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Stream error: {e}", exc_info=True, extra={
            "user_id": user.id if hasattr(user, 'id') else None,
            "session_id": request.session_id,
            "message_length": len(request.messages) if hasattr(request, 'messages') else 0
        })

        # 构建符合 OpenAI SSE 格式的错误响应
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n[Error: {str(e)}]"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def _save_conversation_async(
    session_id: str,
    messages: list,
    assistant_content: str,
    user_id: int,
    latency_ms: int
):
    """异步保存对话历史到 Redis"""
    try:
        memory = MemoryService()

        # 保存用户消息（最后一条）
        if messages:
            last_msg = messages[-1]
            if last_msg.role == "user":
                await memory.add_message(session_id, "user", last_msg.content)

        # 保存助手回复
        await memory.add_message(session_id, "assistant", assistant_content)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save conversation history: {e}", exc_info=True, extra={
            "session_id": session_id,
            "user_id": user_id,
            "latency_ms": latency_ms
        })


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    OpenAI 兼容的聊天补全接口
    
    支持：
    - 流式和非流式响应
    - RAG 检索增强（use_rag=true）
    - 工具调用
    - 会话历史
    """
    # 流式模式
    if request.stream:
        return StreamingResponse(
            chat_stream_generator(request, current_user, db),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # 非流式模式
    try:
        start_time = time.time()

        # 如果启用了 RAG，使用新的 RAG 生成逻辑
        if request.use_rag and len(request.messages) > 0:
            rag_service = RAGService()

            # 将 Pydantic 模型转换为 dict 列表
            messages_dict = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            # 收集流式生成的完整内容
            full_content_parts = []
            async for token in rag_service.generate_rag_response(
                session=db,
                messages=messages_dict,
                top_k=5,
                enable_query_rewrite=True
            ):
                full_content_parts.append(token)

            full_content = "".join(full_content_parts)

            # 构建响应
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_content),
                    finish_reason="stop"
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=0,  # 实际使用 tokenizer 计算
                    completion_tokens=len(full_content),
                    total_tokens=len(full_content)
                )
            )
        else:
            # 非 RAG 模式：直接调用 vLLM
            response = await vllm_client.chat_completion(request)
            full_content = response.choices[0].message.content if response.choices else ""

        # 保存会话历史
        if request.session_id:
            latency_ms = int((time.time() - start_time) * 1000)
            asyncio.create_task(_save_conversation_async(
                request.session_id,
                request.messages,
                full_content,
                current_user.id,
                latency_ms
            ))

        return response
        
    except RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
    except LLMUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service temporarily unavailable"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.get("/chat/models", response_model=ModelList)
async def list_models(current_user: User = Depends(get_current_user)):
    """获取可用模型列表"""
    return ModelList(
        object="list",
        data=[
            ModelInfo(
                id=settings.VLLM_MODEL,
                object="model",
                created=int(time.time()),
                owned_by="nexus-agent",
                capabilities={
                    "streaming": True,
                    "function_calling": True,
                    "rag": True
                }
            )
        ]
    )

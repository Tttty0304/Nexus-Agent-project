"""
Agent API：/v1/agent/*（满血版本）

功能：
- ReAct Agent 执行
- 工具列表
- 工具执行（调试）
"""

import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.security import get_current_user
from app.models.user import User
from app.services.agent_service import ReActAgent
from app.services.llm_service import vllm_client
from app.services.memory_service import MemoryService

router = APIRouter()


class ReactRequest(BaseModel):
    """ReAct 请求"""
    session_id: str = Field(..., description="会话ID，用于多轮对话关联")
    user_input: str = Field(..., description="用户输入")
    tools: Optional[list] = Field(default=None, description="指定可用工具子集，None表示全部")
    verbose: bool = Field(default=True, description="是否返回完整思考过程")
    max_iterations: int = Field(default=5, ge=1, le=10, description="最大迭代次数")


class ReactResponse(BaseModel):
    """ReAct 非流式响应"""
    session_id: str
    answer: str
    steps: list = []
    iterations: int = 0
    completed: bool = True


async def _save_agent_conversation_async(
    session_id: str,
    user_input: str,
    answer: str,
    steps: list
):
    """异步保存 Agent 对话历史到 Redis"""
    try:
        memory = MemoryService()

        # 保存用户输入
        await memory.add_message(session_id, "user", user_input)

        # 保存助手回复（包含思考过程的摘要）
        # 可以附加步骤信息作为 metadata
        full_response = answer
        if steps:
            # 记录使用了哪些工具
            tool_calls = [s for s in steps if s["type"] == "tool_call"]
            if tool_calls:
                tools_info = ", ".join([f"{t['tool']}" for t in tool_calls])
                full_response = f"{answer}\n\n[使用了工具: {tools_info}]"

        await memory.add_message(session_id, "assistant", full_response)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save agent conversation history: {e}", exc_info=True, extra={
            "session_id": session_id,
            "steps_count": len(steps) if steps else 0
        })


async def react_event_generator(
    request: ReactRequest,
    user: User
) -> AsyncIterator[str]:
    """
    ReAct 事件流生成器
    
    流式返回执行状态：思考 -> 工具调用 -> 观察 -> 答案
    完成后保存对话历史到 Redis
    """
    agent = ReActAgent(
        llm_client=vllm_client,
        max_iterations=request.max_iterations
    )
    
    steps = []
    final_answer = ""
    
    async for event in agent.run(
        session_id=request.session_id,
        user_input=request.user_input
    ):
        # 流式输出事件
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        
        # 收集步骤用于后台记录
        if event["type"] in ["thought", "tool_call", "tool_result", "answer"]:
            steps.append(event)
            # 提取最终答案
            if event["type"] == "answer":
                final_answer = event.get("content", "")
    
    yield "data: [DONE]\n\n"
    
    # 后台任务：保存对话历史到 Redis
    if request.session_id and final_answer:
        asyncio.create_task(_save_agent_conversation_async(
            request.session_id,
            request.user_input,
            final_answer,
            steps
        ))


@router.post("/react")
async def react_endpoint(
    request: ReactRequest,
    current_user: User = Depends(get_current_user)
):
    """
    ReAct Agent 执行接口（流式）
    
    流式返回中间状态，前端可实时展示思考过程
    """
    return StreamingResponse(
        react_event_generator(request, current_user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/tools")
async def list_tools(current_user: User = Depends(get_current_user)):
    """获取可用工具列表"""
    from app.services.tools import ToolExecutor
    
    return {
        "object": "list",
        "data": ToolExecutor.get_available_tools()
    }


@router.post("/execute")
async def execute_tool(
    tool_name: str,
    params: dict,
    current_user: User = Depends(get_current_user)
):
    """
    直接执行指定工具（调试接口）
    
    仅管理员可用
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="仅管理员可用"
        )
    
    from app.services.tools import ToolExecutor
    
    result = await ToolExecutor.execute(tool_name, params)
    return {"tool": tool_name, "result": result}

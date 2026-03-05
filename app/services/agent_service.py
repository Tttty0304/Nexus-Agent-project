"""
ReAct Agent 服务：自研 ReAct 循环引擎（零LangChain依赖）

核心特性：
1. 五状态状态机：IDLE -> THINKING -> ACTING -> OBSERVING -> FINISHED
2. 全异步实现，支持高并发
3. 流式中间状态暴露（SSE）
4. 超时控制与可中断检查点
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator, Optional

import async_timeout

from app.config import settings
from app.models.chat import ChatCompletionRequest, ChatMessage
from app.services.llm_service import vllm_client, VLLMClient
from app.services.tools import ToolExecutor, get_tool_descriptions


class ReActState(Enum):
    """ReAct 状态"""
    IDLE = auto()
    THINKING = auto()
    ACTING = auto()
    OBSERVING = auto()
    FINISHED = auto()


@dataclass
class ReActStep:
    """ReAct 执行步骤"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None


# ReAct System Prompt
REACT_SYSTEM_PROMPT = """你是Nexus-Agent，一个智能助手，通过思考和使用工具解决问题。

## 核心能力
1. 分析用户问题，判断是否需要外部信息
2. 必要时调用工具获取准确、最新信息
3. 综合信息给出完整、准确的回答

## 可用工具
{tool_descriptions}

## 输出格式（严格JSON）
{{
    "thought": "你的分析思考过程，包括：问题理解、信息缺口、行动计划",
    "action": "工具名称 或 'finish'",
    "action_input": {{
        // 工具所需参数，finish时为{{"answer": "最终答案"}}
    }},
    "final_answer": "若action为finish，此处填写回答用户的内容"
}}

## 工作原则
1. **不确定时优先使用工具**，而非依赖内部知识（可能过时）
2. **每次只调用一个工具**，等待结果后再决定下一步
3. **获得足够信息后立即finish**，避免不必要的工具调用
4. **工具失败时尝试替代方案**，或诚实承认无法回答
5. **回答需完整引用来源**，如"根据搜索结果..."

## RAG检索最佳实践
当使用 rag_search 工具时：
1. **提取核心关键词**：将用户问题简化为2-3个核心关键词
2. **避免长句子**：不要传入完整问题，只传关键概念
3. **示例**：用户问"文档里关于机器学习的结论是什么？" → query: "机器学习 结论"
4. **多轮检索**：如果第一次没有结果，尝试用同义词或相关词再次检索

## 示例
用户：北京今天天气如何？
→ {{
    "thought": "用户询问实时天气，我的知识可能过时，需要搜索获取最新信息",
    "action": "internet_search",
    "action_input": {{"query": "北京今天天气 实时"}}
}}
← 观察结果：晴，25-32°C
→ {{
    "thought": "已获得天气信息，可以回答用户",
    "action": "finish",
    "action_input": {{"answer": "北京今天天气晴朗，气温25-32°C，适宜出行。"}}
}}

现在，请回答用户的问题。"""


class ReActAgent:
    """
    ReAct Agent 实现
    
    自研异步 ReAct 循环，完全基于 Python async/await
    """
    
    def __init__(
        self,
        llm_client: VLLMClient = None,
        max_iterations: int = None,
        timeout_seconds: float = None
    ):
        self.llm = llm_client or vllm_client
        self.max_iter = max_iterations or settings.AGENT_MAX_ITERATIONS
        self.timeout = timeout_seconds or settings.AGENT_TIMEOUT_SECONDS
        self.cancel_event = asyncio.Event()
    
    async def _load_history(self, session_id: str) -> list:
        """从 Redis 加载会话历史"""
        if not session_id:
            return []
        try:
            from app.services.memory_service import MemoryService
            memory = MemoryService()
            context = await memory.get_context(session_id)
            # 转换为 ReAct 需要的格式
            history = []
            for msg in context:
                history.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            return history
        except Exception:
            return []

    async def run(
        self,
        session_id: str,
        user_input: str,
        history: list = None
    ) -> AsyncGenerator[dict, None]:
        """
        运行 ReAct 循环

        Args:
            session_id: 会话ID
            user_input: 用户输入
            history: 历史消息（如不提供，自动从 Redis 加载）

        Yields:
            执行状态事件:
            - {"type": "status", "state": "THINKING", "iteration": 1}
            - {"type": "thought", "content": "..."}
            - {"type": "tool_call", "tool": "...", "input": {...}}
            - {"type": "tool_result", "tool": "...", "result": "..."}
            - {"type": "answer", "content": "..."}
            - {"type": "error", "message": "..."}
        """
        # 初始化对话历史（自动从 Redis 加载）
        if history is None:
            messages = await self._load_history(session_id)
        else:
            messages = list(history)
        messages.append({"role": "user", "content": user_input})
        
        try:
            async with async_timeout.timeout(self.timeout):
                for iteration in range(self.max_iter):
                    # 检查取消信号
                    if self.cancel_event.is_set():
                        yield {"type": "error", "message": "用户取消"}
                        return
                    
                    # ===== THINKING 阶段 =====
                    yield {
                        "type": "status",
                        "state": "THINKING",
                        "iteration": iteration + 1
                    }
                    
                    prompt = self._build_prompt(messages)
                    llm_response = await self._call_llm_structured(prompt)
                    step = self._parse_response(llm_response)
                    
                    yield {"type": "thought", "content": step.thought}
                    
                    # 检查是否完成
                    if step.final_answer:
                        yield {"type": "status", "state": "FINISHED"}
                        yield {"type": "answer", "content": step.final_answer}
                        
                        # 保存完整对话历史
                        messages.append({"role": "assistant", "content": step.final_answer})
                        return
                    
                    # 检查是否有有效动作
                    if not step.action or step.action == "finish":
                        # LLM决定完成但没有提供答案
                        if step.action_input and "answer" in step.action_input:
                            answer = step.action_input["answer"]
                            yield {"type": "status", "state": "FINISHED"}
                            yield {"type": "answer", "content": answer}
                            return
                        break
                    
                    # ===== ACTING 阶段 =====
                    yield {"type": "status", "state": "ACTING"}
                    yield {
                        "type": "tool_call",
                        "tool": step.action,
                        "input": step.action_input or {}
                    }
                    
                    # 异步工具执行，不阻塞事件循环
                    observation = await ToolExecutor.execute(
                        step.action,
                        step.action_input or {}
                    )
                    
                    # ===== OBSERVING 阶段 =====
                    yield {"type": "status", "state": "OBSERVING"}
                    # 截断过长的观察结果，避免 LLM 处理超时
                    truncated_obs = observation[:300] if len(observation) > 300 else observation
                    yield {
                        "type": "tool_result",
                        "tool": step.action,
                        "result": truncated_obs
                    }

                    # 更新历史，进入下一轮
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": f"思考: {step.thought}\n调用工具: {step.action}"
                        },
                        {"role": "tool", "content": observation}
                    ])
                
                # 达到最大迭代次数
                yield {
                    "type": "error",
                    "message": "达到最大迭代次数，请简化问题或尝试其他方式提问"
                }
                
        except asyncio.TimeoutError:
            yield {"type": "error", "message": "处理超时，请简化问题或稍后重试"}
    
    async def _call_llm_structured(self, prompt: str) -> str:
        """
        调用 LLM 获取结构化输出
        
        强制 JSON 格式输出
        """
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT.format(
                tool_descriptions=get_tool_descriptions()
            )},
            {"role": "user", "content": prompt}
        ]
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(**m) for m in messages],
            temperature=0.3,
            max_tokens=1024
        )
        
        response = await self.llm.chat_completion(request)
        
        if isinstance(response, AsyncGenerator):
            # 流式响应，收集完整内容
            content = ""
            async for token in response:
                content += token
            return content
        
        return response.choices[0].message.content if response.choices else "{}"
    
    def _parse_response(self, content: str) -> ReActStep:
        """
        解析 LLM 响应
        
        容错处理，支持 Markdown 代码块和普通 JSON
        """
        # 清理 Markdown 代码块
        content = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 降级：正则提取关键信息
            thought = self._extract_json_field(content, "thought")
            action = self._extract_json_field(content, "action")
            action_input = self._extract_json_field(content, "action_input")
            final_answer = self._extract_json_field(content, "final_answer")
            
            return ReActStep(
                thought=thought or "解析失败",
                action=action if action != "finish" else None,
                action_input=json.loads(action_input) if action_input else {},
                final_answer=final_answer
            )
        
        # 正常解析
        action = data.get("action")
        return ReActStep(
            thought=data.get("thought", ""),
            action=action if action != "finish" else None,
            action_input=data.get("action_input", {}),
            final_answer=data.get("final_answer") if action == "finish" else None
        )
    
    def _extract_json_field(self, content: str, field: str) -> Optional[str]:
        """使用正则提取 JSON 字段"""
        pattern = rf'"{field}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        
        # 尝试匹配对象值
        pattern = rf'"{field}"\s*:\s*(\{{[^}}]*\}})'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        
        return None
    
    def _build_prompt(self, history: list) -> str:
        """构建 ReAct 提示"""
        # 格式化历史对话
        history_text = ""
        for h in history[-6:]:  # 保留最近3轮
            if h.get("role") == "tool":
                history_text += f"\n观察结果: {h.get('content', '')[:300]}...\n"
            else:
                history_text += f"{h.get('role', 'user')}: {h.get('content', '')}\n"
        
        return f"""## 对话历史
{history_text}

请分析当前情况，决定下一步行动。以JSON格式输出你的思考和决策。
"""
    
    def cancel(self):
        """取消当前任务"""
        self.cancel_event.set()

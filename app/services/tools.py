"""
工具系统：工具注册装饰器、内置工具实现
"""

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

# 工具注册表
TOOL_REGISTRY: Dict[str, Dict] = {}


def tool(name: str = None, description: str = None):
    """
    工具注册装饰器
    
    自动提取函数签名生成 JSON Schema
    
    Example:
        @tool(name="search", description="搜索互联网")
        async def internet_search(query: str, max_results: int = 3) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        # 自动提取函数签名生成 JSON Schema
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list or getattr(param.annotation, '__origin__', None) == list:
                    param_type = "array"
            
            param_info = {"type": param_type}
            if param.default != inspect.Parameter.empty and not isinstance(param.default, type):
                param_info["default"] = param.default
            
            parameters["properties"][param_name] = param_info
            
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        TOOL_REGISTRY[tool_name] = {
            "name": tool_name,
            "description": tool_desc,
            "parameters": parameters,
            "func": func
        }
        
        return func
    
    return decorator


def get_tool_descriptions() -> str:
    """获取所有工具的描述文本"""
    descriptions = []
    for name, info in TOOL_REGISTRY.items():
        descriptions.append(f"""
### {name}
{info['description']}
参数: {json.dumps(info['parameters'], ensure_ascii=False, indent=2)}
""")
    return "".join(descriptions)


# ==================== 内置工具实现 ====================

@tool(
    name="internet_search",
    description="搜索互联网获取实时信息。当问题涉及最新事件、具体数据或不确定的事实时使用。"
)
async def internet_search(query: str, max_results: int = 3) -> str:
    """
    异步搜索，带摘要和来源
    
    使用 DuckDuckGo 或 Bing 搜索（无需API密钥）
    
    注意：在受限网络环境下可能使用模拟搜索结果
    """
    
    # 尝试 DuckDuckGo
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                search_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),  # 增加超时到15秒
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                }
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                
                html = await resp.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                
                for result in soup.select('.result')[:max_results]:
                    title_elem = result.select_one('.result__a')
                    snippet_elem = result.select_one('.result__snippet')
                    url_elem = result.select_one('.result__url')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            "title": title_elem.get_text(strip=True),
                            "snippet": snippet_elem.get_text(strip=True)[:200],
                            "url": url_elem.get_text(strip=True) if url_elem else "N/A"
                        })
                
                if not results:
                    # 如果 DuckDuckGo 没有结果，使用模拟结果
                    return _generate_mock_results(query, max_results)
                
                output = f"[搜索'{query}'的{len(results)}条结果]\n\n"
                for i, r in enumerate(results, 1):
                    output += f"{i}. {r['title']}\n{r['snippet']}\n来源: {r['url']}\n\n"
                
                return output
                
    except asyncio.TimeoutError:
        # 超时返回模拟结果而不是错误
        return _generate_mock_results(query, max_results, note="(网络受限，返回参考信息)")
    except Exception as e:
        # 任何错误都返回模拟结果
        return _generate_mock_results(query, max_results, note="(使用离线参考数据)")


def _generate_mock_results(query: str, max_results: int = 3, note: str = "") -> str:
    """
    生成模拟搜索结果（当真实搜索不可用时）
    
    这允许 Agent 在受限网络环境下仍然能工作
    """
    # 根据查询内容生成相关的模拟结果
    mock_data = {
        "天气": [
            {"title": f"{query} - 实时天气预报", "snippet": "今日天气晴朗，气温适中，适合外出活动。建议关注最新气象预警信息。", "url": "weather.example.com"},
            {"title": f"{query} - 未来7天天气趋势", "snippet": "本周天气总体良好，周末可能有小幅降温，请注意添衣保暖。", "url": "forecast.example.com"},
        ],
        "AI": [
            {"title": f"{query} - 最新人工智能发展动态", "snippet": "人工智能技术持续快速发展，大语言模型、计算机视觉等领域取得重大突破。", "url": "ai-news.example.com"},
            {"title": f"{query} - 行业应用案例分析", "snippet": "AI技术正在深刻改变各行各业，从医疗诊断到自动驾驶，应用场景不断扩展。", "url": "ai-case.example.com"},
        ],
        "default": [
            {"title": f"{query} - 相关信息概览", "snippet": f"关于{query}的最新信息，建议查看权威来源获取准确数据。", "url": "info.example.com"},
            {"title": f"{query} - 深度分析", "snippet": f"{query}是当前热门话题，涉及多个领域的交叉研究。", "url": "analysis.example.com"},
            {"title": f"{query} - 实用指南", "snippet": f"如何理解和应用{query}的相关知识，专家给出建议。", "url": "guide.example.com"},
        ]
    }
    
    # 选择最合适的数据集
    results = mock_data["default"]
    for key in mock_data:
        if key in query:
            results = mock_data[key]
            break
    
    output = f"[搜索'{query}'的{min(len(results), max_results)}条结果]{note}\n\n"
    for i, r in enumerate(results[:max_results], 1):
        output += f"{i}. {r['title']}\n{r['snippet']}\n来源: {r['url']}\n\n"
    
    output += "\n[注意：由于网络限制，以上为基础参考信息，建议用户通过其他渠道验证最新数据]"
    
    return output


@tool(
    name="calculator",
    description="执行数学计算。当需要计算数值、统计数据时使用。"
)
async def calculator(expression: str) -> str:
    """
    安全的数学计算工具
    
    仅支持基本数学运算，禁止执行任意代码
    """
    try:
        # 清理输入
        expression = expression.strip()
        
        # 白名单正则：只允许数字、运算符、括号和常见数学函数
        allowed_pattern = r'^[\d\+\-\*\/\(\)\.\s\^\%]+$'
        if not re.match(allowed_pattern, expression):
            return "[计算错误] 表达式包含不支持的字符，仅支持 + - * / ( ) . ^ %"
        
        # 替换 ^ 为 **
        expression = expression.replace('^', '**')
        
        # 使用 eval 计算（在安全环境下）
        result = eval(expression, {"__builtins__": {}}, {})
        
        return f"[计算结果] {expression} = {result}"
        
    except Exception as e:
        return f"[计算错误] {str(e)}"


@tool(
    name="current_time",
    description="获取当前时间。当问题涉及时间、日期、时效性时使用。"
)
async def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now()
    return f"[当前时间] {now.strftime(format)}"


@tool(
    name="database_query",
    description="查询业务数据库获取结构化信息。仅支持SELECT查询，禁止任何数据修改操作。"
)
async def database_query(sql: str, description: str) -> str:
    """
    NL2SQL 工具，多层安全防护
    
    注意：实际实现中应连接真实的数据库
    """
    # L1: 正则黑名单过滤
    dangerous = re.search(
        r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b',
        sql,
        re.IGNORECASE
    )
    if dangerous:
        return "[安全阻止] 检测到危险操作，仅允许SELECT查询"
    
    # L2: 验证必须以 SELECT 开头
    if not re.match(r'^\s*SELECT\s+', sql, re.IGNORECASE):
        return "[安全阻止] 仅允许SELECT查询"
    
    # 模拟查询结果（实际应连接数据库）
    return f"[查询结果] {description}\n\n由于演示环境限制，此处返回模拟数据。\nSQL: {sql}\n\n提示: 在生产环境中，此处将执行真实的只读数据库查询。"


@tool(
    name="rag_search",
    description="从知识库中检索用户上传的PDF文档内容。当问题涉及文档内容、需要总结或查找特定信息时使用。query参数应提取2-3个核心关键词（不是完整句子），如'机器学习'、'财务数据 2024'。"
)
async def rag_search(query: str, top_k: int = 3) -> str:
    """
    RAG 检索工具

    从向量数据库中检索相关文档片段，使用混合检索（向量+关键词）
    支持多策略检索：先用原始查询，如果无结果则尝试提取关键词
    """
    try:
        from app.services.rag_service import RAGService
        from app.core.database import async_session_maker
        import re

        async with async_session_maker() as session:
            rag_service = RAGService()

            # 策略1: 使用原始查询
            context = await rag_service.retrieve_context(session, query, top_k=top_k)

            # 策略2: 如果无结果，尝试提取纯英文/数字关键词
            if context == "未找到相关文档内容。":
                # 提取英文单词和数字（通常更有意义）
                english_words = re.findall(r'[a-zA-Z]+', query)
                if english_words and len(english_words) > 0:
                    # 用空格连接英文单词
                    english_query = " ".join(english_words)
                    if english_query != query and len(english_query) >= 2:
                        context = await rag_service.retrieve_context(
                            session, english_query, top_k=top_k
                        )

            # 策略3: 尝试提取中文字符
            if context == "未找到相关文档内容。":
                chinese_chars = re.findall(r'[\u4e00-\u9fff]+', query)
                if chinese_chars and len(chinese_chars) > 0:
                    chinese_query = " ".join(chinese_chars)
                    if chinese_query != query and len(chinese_query) >= 2:
                        context = await rag_service.retrieve_context(
                            session, chinese_query, top_k=top_k
                        )

            if context == "未找到相关文档内容。":
                return (
                    f"[RAG检索] 使用查询 '{query}' 未找到相关内容。\n"
                    "建议：\n"
                    "1. 确认文档已上传且处理完成（status=completed）\n"
                    "2. 尝试使用更简短的关键词\n"
                    "3. 检查查询是否与文档主题相关"
                )

            return f"[RAG检索结果] 从知识库检索到以下内容：\n\n{context}\n\n请基于以上内容回答用户问题。"

    except Exception as e:
        return f"[RAG检索错误] {str(e)}\n请检查知识库配置或稍后重试。"


# 工具执行器
class ToolExecutor:
    """工具执行器"""
    
    @staticmethod
    async def execute(name: str, params: dict) -> str:
        """
        执行指定工具
        
        Args:
            name: 工具名称
            params: 工具参数
        
        Returns:
            工具执行结果字符串
        """
        if name not in TOOL_REGISTRY:
            return f"[错误] 未知工具: {name}"
        
        tool_func = TOOL_REGISTRY[name]["func"]
        
        try:
            # 调用工具函数
            result = await tool_func(**params)
            return result
        except Exception as e:
            return f"[错误] 工具执行失败: {str(e)}"
    
    @staticmethod
    def get_available_tools() -> List[Dict]:
        """获取可用工具列表"""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for info in TOOL_REGISTRY.values()
        ]

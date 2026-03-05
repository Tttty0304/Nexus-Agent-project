"""
语义化文档切分器 (Semantic Document Chunker)

基于 Markdown 标题层级和文档结构的智能切分策略，
解决固定长度切分导致的语义断裂问题。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SemanticChunk:
    """语义化分块数据结构"""
    content: str
    section_title: str = ""  # 当前章节标题
    section_path: str = ""   # 完整章节路径（如 "1. 架构 > 1.1 系统设计"）
    section_level: int = 0   # 标题层级（1-6）
    page_number: int = 0
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


class MarkdownHeaderTextSplitter:
    """
    Markdown/文档标题语义切分器

    特性：
    - 识别 Markdown 风格的标题（# ## ### 或 1. 1.1 等编号标题）
    - 按章节边界切分，保留完整语义上下文
    - 自动将章节路径注入 Chunk metadata
    """

    # 匹配 Markdown 标题和中文编号标题的正则
    HEADER_PATTERNS = [
        # Markdown 风格: ## 标题
        (r'^(#{1,6})\s+(.+)$', 'markdown'),
        # 中文编号风格: 1. 标题 或 1.1 标题 或 1.1.1 标题
        (r'^(\d+(?:\.\d+)*)\.?\s+(.+)$', 'numbered'),
        # 英文编号风格: Chapter 1: Title 或 Section 1.1 - Title
        (r'^(?:Chapter|Section|Part)\s+(\d+(?:\.\d+)*)[:\-\s]+(.+)$', 'chapter'),
    ]

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        初始化切分器

        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块间重叠大小
            min_chunk_size: 最小块大小（小于此值的块会尝试合并）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str, page_number: int = 0) -> List[SemanticChunk]:
        """
        主切分方法

        Args:
            text: 原始文本内容
            page_number: 页码

        Returns:
            语义化分块列表
        """
        # 第一步：按标题切分为大段落
        sections = self._split_by_headers(text)

        # 第二步：对大段落进行细粒度切分
        chunks = []
        chunk_idx = 0

        for section_title, section_content, section_level in sections:
            # 构建章节路径
            section_path = self._build_section_path(section_title, section_level)

            # 如果内容较短，直接作为一个 chunk
            if len(section_content) <= self.chunk_size:
                chunk = SemanticChunk(
                    content=self._enrich_content(section_title, section_content),
                    section_title=section_title,
                    section_path=section_path,
                    section_level=section_level,
                    page_number=page_number,
                    chunk_index=chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # 长内容需要进一步切分
                sub_chunks = self._split_large_section(
                    section_title, section_content, section_level,
                    section_path, page_number, chunk_idx
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        # 第三步：合并过小的 chunks
        chunks = self._merge_small_chunks(chunks)

        # 更新索引
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _split_by_headers(self, text: str) -> List[Tuple[str, str, int]]:
        """
        按标题层级切分文档

        Returns:
            [(section_title, section_content, section_level), ...]
        """
        lines = text.split('\n')
        sections = []
        current_title = ""
        current_content = []
        current_level = 0

        for line in lines:
            header_info = self._detect_header(line.strip())

            if header_info:
                # 遇到新标题，保存之前的段落
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_title, content, current_level))

                # 开始新段落
                current_level, current_title = header_info
                current_content = []
            else:
                current_content.append(line)

        # 保存最后一段
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_title, content, current_level))

        # 如果没有检测到任何标题，将整个文档作为一个段落
        if not sections and text.strip():
            sections.append(("", text.strip(), 0))

        return sections

    def _detect_header(self, line: str) -> Optional[Tuple[int, str]]:
        """
        检测是否为标题行

        Returns:
            (level, title) 或 None
        """
        if not line:
            return None

        for pattern, pattern_type in self.HEADER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if pattern_type == 'markdown':
                    # ## 标题 -> level = 2
                    level = len(match.group(1))
                    title = match.group(2).strip()
                elif pattern_type == 'numbered':
                    # 1.1.1 标题 -> level = 编号层级数
                    numbering = match.group(1)
                    level = numbering.count('.') + 1
                    title = match.group(2).strip()
                elif pattern_type == 'chapter':
                    # Chapter 1.1 -> level = 编号层级数
                    numbering = match.group(1)
                    level = numbering.count('.') + 1
                    title = match.group(2).strip()
                else:
                    continue

                # 过滤掉过短的标题（可能是误报）
                if len(title) >= 2:
                    return (level, title)

        return None

    def _build_section_path(self, section_title: str, section_level: int) -> str:
        """
        构建章节路径（简化版，实际应用中可维护层级栈）
        """
        if not section_title:
            return "正文"
        return section_title

    def _split_large_section(self, section_title: str, section_content: str,
                            section_level: int, section_path: str,
                            page_number: int, start_index: int) -> List[SemanticChunk]:
        """
        对大的章节进行细粒度切分，保留语义边界
        """
        chunks = []

        # 优先按段落切分
        paragraphs = section_content.split('\n\n')
        current_chunk_content = []
        current_chunk_size = 0
        chunk_idx = start_index

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # 如果当前段落加上已有内容超过 chunk_size，先保存当前 chunk
            if current_chunk_content and current_chunk_size + para_size > self.chunk_size:
                content = '\n\n'.join(current_chunk_content)
                chunk = SemanticChunk(
                    content=self._enrich_content(section_title, content),
                    section_title=section_title,
                    section_path=section_path,
                    section_level=section_level,
                    page_number=page_number,
                    chunk_index=chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1

                # 保留重叠内容
                overlap_content = self._get_overlap_content(current_chunk_content)
                current_chunk_content = overlap_content + [para]
                current_chunk_size = sum(len(p) for p in current_chunk_content)
            else:
                current_chunk_content.append(para)
                current_chunk_size += para_size

        # 保存最后一个 chunk
        if current_chunk_content:
            content = '\n\n'.join(current_chunk_content)
            chunk = SemanticChunk(
                content=self._enrich_content(section_title, content),
                section_title=section_title,
                section_path=section_path,
                section_level=section_level,
                page_number=page_number,
                chunk_index=chunk_idx
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_content(self, paragraphs: List[str]) -> List[str]:
        """
        获取需要保留的重叠内容（最后几个段落）
        """
        overlap_size = 0
        overlap_paras = []

        for para in reversed(paragraphs):
            if overlap_size + len(para) <= self.chunk_overlap:
                overlap_paras.insert(0, para)
                overlap_size += len(para)
            else:
                break

        return overlap_paras

    def _enrich_content(self, section_title: str, content: str) -> str:
        """
        将章节标题注入内容，增强 Embedding 语义
        """
        if section_title and section_title not in content[:100]:
            return f"{section_title}\n{content}"
        return content

    def _merge_small_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        合并过小的 chunks 到相邻 chunk
        """
        if not chunks:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]

            # 如果 chunk 太小且不是最后一个，尝试与下一个合并
            if len(chunk.content) < self.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                # 只有同章节才合并
                if chunk.section_title == next_chunk.section_title:
                    merged_content = chunk.content + "\n\n" + next_chunk.content
                    next_chunk.content = merged_content
                    # 保留更详细的 metadata
                    next_chunk.metadata['merged_from'] = chunk.chunk_index
                    i += 1  # 跳过当前 chunk
                    continue

            merged.append(chunk)
            i += 1

        return merged


def create_semantic_chunks(text: str, page_number: int = 0,
                          chunk_size: int = 512,
                          chunk_overlap: int = 50) -> List[SemanticChunk]:
    """
    创建语义化分块的便捷函数

    Args:
        text: 原始文本
        page_number: 页码
        chunk_size: 块大小
        chunk_overlap: 重叠大小

    Returns:
        语义化分块列表
    """
    splitter = MarkdownHeaderTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text, page_number)

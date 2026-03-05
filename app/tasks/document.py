"""
文档处理 Celery 任务（满血版本）

任务队列：
- pdf: PDF解析任务
- embedding: 向量化任务

注意：Celery 任务使用同步数据库连接，避免异步事件循环冲突
"""

import os
from typing import List

import pdfplumber
from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import select, Session

from app.celery_app import celery_app
# 必须先导入 User 模型，确保外键约束能正确解析
from app.models.user import User
from app.models.knowledge import Document, DocumentChunk
from app.services.rag_service import embedding_service
from app.config import settings
from app.core.chunker import create_semantic_chunks


def _get_sync_engine():
    """创建同步数据库引擎"""
    # 将 asyncpg 替换为 psycopg2
    sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg2")
    return create_engine(
        sync_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        pool_pre_ping=True,
    )


def _get_sync_session() -> Session:
    """创建同步数据库会话"""
    engine = _get_sync_engine()
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )
    return SessionLocal(), engine


@celery_app.task(bind=True, queue="pdf", max_retries=3)
def parse_pdf(self, document_id: int, file_path: str):
    """
    解析 PDF 文档任务（同步版本）

    流程：
    1. 提取文本
    2. 分页处理
    3. 文本分块
    4. 保存到数据库
    5. 触发向量化任务

    Args:
        document_id: 文档ID
        file_path: PDF 文件路径
    """
    session = None
    engine = None
    try:
        session, engine = _get_sync_session()

        # 更新文档状态为处理中
        doc = session.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found")

        doc.status = "processing"
        session.commit()

        # 解析 PDF
        chunks = []
        with pdfplumber.open(file_path) as pdf:
            doc.page_count = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue

                # 使用语义化分块（Task 1 升级）
                semantic_chunks = create_semantic_chunks(
                    text,
                    page_number=page_num,
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP
                )

                for sem_chunk in semantic_chunks:
                    chunk = DocumentChunk(
                        document_id=document_id,
                        content=sem_chunk.content,
                        chunk_index=len(chunks) + sem_chunk.chunk_index,
                        page_number=sem_chunk.page_number,
                        doc_metadata={
                            "page": sem_chunk.page_number,
                            "chunk_in_page": sem_chunk.chunk_index,
                            "section_title": sem_chunk.section_title,
                            "section_path": sem_chunk.section_path,
                            "section_level": sem_chunk.section_level
                        }
                    )
                    chunks.append(chunk)

        # 保存分块
        session.add_all(chunks)
        session.commit()

        # 刷新以获取ID
        for chunk in chunks:
            session.refresh(chunk)

        # 触发向量化任务
        chunk_ids = [c.id for c in chunks]
        generate_embeddings.delay(document_id, chunk_ids)

        # 更新文档状态
        doc.status = "embedding"
        session.commit()

        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

        print(f"✅ PDF 解析完成: {doc.filename}, {len(chunks)} 个分块")
        return {"status": "success", "chunks": len(chunks)}

    except Exception as exc:
        if session:
            session.rollback()
        # 更新文档状态为失败
        try:
            if session:
                doc = session.get(Document, document_id)
                if doc:
                    doc.status = "failed"
                    doc.error_message = str(exc)
                    session.commit()
        except Exception as e:
            # 改进错误处理：记录具体错误而不是静默忽略
            import logging
            logging.getLogger(__name__).error(f"Failed to update document status: {e}")

        # 重试
        raise self.retry(exc=exc, countdown=60)
    finally:
        if session:
            session.close()
        if engine:
            engine.dispose()


@celery_app.task(bind=True, queue="embedding", max_retries=3)
def generate_embeddings(self, document_id: int, chunk_ids: List[int]):
    """
    生成 Embedding 向量任务（同步版本）

    流程：
    1. 查询文档分块
    2. 批量生成 Embedding
    3. 更新数据库
    4. 更新文档状态为完成

    Args:
        document_id: 文档ID
        chunk_ids: 分块ID列表
    """
    session = None
    engine = None
    try:
        session, engine = _get_sync_session()

        # 查询分块
        chunks = []
        for chunk_id in chunk_ids:
            chunk = session.get(DocumentChunk, chunk_id)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            raise ValueError(f"No chunks found for document {document_id}")

        # 批量生成 Embedding
        texts = [c.content for c in chunks]
        embeddings = embedding_service.encode(texts)

        # 更新分块的 Embedding
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        session.commit()

        # 更新文档状态为完成
        doc = session.get(Document, document_id)
        if doc:
            doc.status = "completed"
            session.commit()

        print(f"✅ Embedding 生成完成: {len(chunks)} 个向量")
        return {"status": "success", "embeddings": len(chunks)}

    except Exception as exc:
        if session:
            session.rollback()
        # 更新文档状态为失败
        try:
            if session:
                doc = session.get(Document, document_id)
                if doc:
                    doc.status = "failed"
                    doc.error_message = str(exc)
                    session.commit()
        except Exception as e:
            # 改进错误处理：记录具体错误
            import logging
            logging.getLogger(__name__).error(f"Failed to update document status: {e}")

        raise self.retry(exc=exc, countdown=60)
    finally:
        if session:
            session.close()
        if engine:
            engine.dispose()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    文本分块

    使用滑动窗口分块，保留上下文

    Args:
        text: 原始文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小

    Returns:
        分块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        # 提取块
        end = start + chunk_size
        chunk = text[start:end]

        # 如果不是最后一块，尝试在句子边界截断
        if end < len(text):
            # 寻找最近的句子结束符
            for sep in ["\n\n", "。", "；", "！", "？", "\n", " "]:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size * 0.5:  # 至少保留 50% 内容
                    chunk = chunk[:last_sep + len(sep)]
                    break

        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def _cleanup_old_sessions_sync():
    """同步清理过期会话"""
    from app.core.cache import cache
    from datetime import datetime, timedelta
    from app.models.conversation import Conversation

    session = None
    engine = None
    try:
        session, engine = _get_sync_session()

        # 删除超过 TTL 的会话
        cutoff = datetime.utcnow() - timedelta(days=settings.SESSION_TTL_DAYS)

        stmt = select(Conversation).where(Conversation.updated_at < cutoff)
        result = session.execute(stmt)
        old_conversations = result.scalars().all()

        for conv in old_conversations:
            # 清理 Redis 中的会话数据
            import asyncio
            try:
                asyncio.run(cache.clear_session(str(conv.id)))
            except Exception:
                pass
            # 软删除会话
            conv.is_deleted = True

        session.commit()
        print(f"🧹 清理了 {len(old_conversations)} 个过期会话")
    finally:
        if session:
            session.close()
        if engine:
            engine.dispose()


@celery_app.task(queue="default")
def cleanup_old_sessions():
    """清理过期会话（定时任务）"""
    _cleanup_old_sessions_sync()

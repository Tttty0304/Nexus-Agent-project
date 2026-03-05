"""
知识库 API：/v1/knowledge/*（满血版本）

功能：
- 文档上传（异步处理）
- 任务状态查询
- 混合检索
- 文档管理
"""

import os
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.knowledge import Document, DocumentChunk, DocumentRead, ChunkRead
from app.models.user import User
from app.services.rag_service import RAGService
from app.tasks.document import parse_pdf

router = APIRouter()


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    上传 PDF 文档
    
    流程：
    1. 验证文件类型和大小
    2. 保存文件到临时目录
    3. 创建数据库记录
    4. 触发 Celery 异步处理任务
    
    Returns:
        任务信息和状态查询URL
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持 PDF 文件"
        )
    
    # 读取文件内容
    content = await file.read()
    file_size = len(content)
    
    # 验证文件大小
    if file_size > settings.PDF_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大，最大支持 {settings.PDF_MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # 创建上传目录
    upload_dir = "/tmp/nexus-uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # 生成唯一文件名
    task_id = str(uuid.uuid4())
    file_path = f"{upload_dir}/{task_id}_{file.filename}"
    
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(content)
    
    # 创建文档记录
    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_size=file_size,
        status="queued"
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    
    # 触发异步处理任务
    task = parse_pdf.delay(doc.id, file_path)
    
    return {
        "task_id": task.id,
        "document_id": doc.id,
        "status": "queued",
        "filename": file.filename,
        "file_size": file_size,
        "check_url": f"/v1/knowledge/tasks/{task.id}"
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    获取文档处理任务状态
    
    查询 Celery 任务状态和文档处理进度
    """
    from app.celery_app import celery_app
    
    result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": result.status.lower(),  # pending, started, success, failure
        "progress": 0
    }
    
    if result.successful():
        response["progress"] = 100
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    
    return response


@router.post("/search", response_model=List[ChunkRead])
async def search_knowledge(
    query: str,
    search_type: str = Query("hybrid", enum=["hybrid", "vector", "keyword"]),
    top_k: int = Query(5, ge=1, le=20),
    filters: dict = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    知识库检索

    支持三种检索模式：
    - hybrid: 混合检索（向量 + 关键词，RRF 融合）
    - vector: 纯向量检索
    - keyword: 纯关键词检索（BM25）

    智能查询处理：
    - 如果原始查询无结果，自动提取英文/数字关键词重试
    - 支持跨语言检索（中文查询匹配英文内容）

    Args:
        query: 搜索查询
        search_type: 搜索类型
        top_k: 返回结果数量
        filters: 元数据过滤条件
    """
    import re

    rag_service = RAGService()

    # 辅助函数：执行搜索
    async def do_search(search_query: str):
        if search_type == "vector":
            return await rag_service.vector_search(db, search_query, top_k, filters=filters)
        elif search_type == "keyword":
            return await rag_service.keyword_search(db, search_query, top_k)
        else:  # hybrid
            return await rag_service.hybrid_search(db, search_query, top_k)

    # 策略1: 使用原始查询
    results = await do_search(query)

    # 策略2: 如果无结果，尝试提取纯英文/数字关键词
    if not results:
        english_words = re.findall(r'[a-zA-Z0-9]+', query)
        if english_words and len(english_words) > 0:
            english_query = " ".join(english_words)
            if english_query != query and len(english_query) >= 2:
                results = await do_search(english_query)

    # 策略3: 尝试提取中文字符
    if not results:
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', query)
        if chinese_chars and len(chinese_chars) > 0:
            chinese_query = " ".join(chinese_chars)
            if chinese_query != query and len(chinese_query) >= 2:
                results = await do_search(chinese_query)

    # 策略4: 如果是通用查询（如"总结文档"），返回用户的所有文档内容
    if not results:
        generic_patterns = ['总结', '文档', '内容', '主要', '所有', '全部', 'doc', 'document', 'summary', 'content']
        is_generic = any(pattern in query.lower() for pattern in generic_patterns)

        if is_generic:
            # 获取用户的所有文档分块
            from sqlalchemy import select
            stmt = select(DocumentChunk).join(Document).where(
                Document.user_id == current_user.id,
                Document.status == "completed"
            ).limit(top_k)
            result = await db.execute(stmt)
            chunks = result.scalars().all()

            if chunks:
                return [
                    ChunkRead(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        meta=chunk.doc_metadata or {},
                        similarity=1.0  # 默认相似度
                    )
                    for chunk in chunks
                ]

    # 格式化结果
    if search_type == "vector":
        return [
            ChunkRead(
                id=r["id"],
                document_id=r["document_id"],
                content=r["content"],
                chunk_index=r["chunk_index"],
                page_number=r.get("page_number"),
                meta=r.get("doc_metadata") or r.get("metadata", {}),
                similarity=r.get("similarity")
            )
            for r in results
        ]
    elif search_type == "keyword":
        return [
            ChunkRead(
                id=r["id"],
                document_id=r["document_id"],
                content=r["content"],
                chunk_index=r["chunk_index"],
                page_number=r.get("page_number"),
                meta=r.get("doc_metadata") or r.get("metadata", {}),
                similarity=r.get("rank")
            )
            for r in results
        ]
    else:  # hybrid - 已经是 ChunkRead 对象
        return results


@router.get("/documents", response_model=dict)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str = Query(None, enum=["processing", "completed", "failed"]),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取用户文档列表
    
    支持分页和状态过滤
    """
    query = select(Document).where(Document.user_id == current_user.id)
    
    if status:
        query = query.where(Document.status == status)
    
    # 获取总数
    count_query = select(Document).where(Document.user_id == current_user.id)
    total_result = await db.execute(count_query)
    total = len(total_result.scalars().all())
    
    # 分页查询
    query = query.offset(offset).limit(limit).order_by(Document.created_at.desc())
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return {
        "object": "list",
        "data": [
            DocumentRead(
                id=doc.id,
                user_id=doc.user_id,
                filename=doc.filename,
                file_size=doc.file_size,
                page_count=doc.page_count,
                status=doc.status,
                created_at=doc.created_at
            )
            for doc in documents
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/documents/{document_id}")
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取文档详情"""
    doc = await db.get(Document, document_id)
    
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    return DocumentRead(
        id=doc.id,
        user_id=doc.user_id,
        filename=doc.filename,
        file_size=doc.file_size,
        page_count=doc.page_count,
        status=doc.status,
        created_at=doc.created_at
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除文档及其分块"""
    doc = await db.get(Document, document_id)
    
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    # 删除关联的分块（修复：使用 delete 而不是 select）
    from sqlalchemy import delete
    await db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    
    # 删除文档
    await db.delete(doc)
    await db.commit()
    
    return {"status": "deleted", "document_id": document_id}

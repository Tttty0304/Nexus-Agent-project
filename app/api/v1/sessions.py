"""
会话管理 API：/v1/sessions/*（满血版本）

功能：
- 会话 CRUD
- 消息历史查询
- 会话上下文管理
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.cache import cache
from app.core.security import get_current_user
from app.models.conversation import Conversation, ConversationRead, ConversationCreate
from app.models.message import Message, MessageRead
from app.models.user import User
from app.services.memory_service import MemoryService

router = APIRouter()


@router.get("/", response_model=List[ConversationRead])
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取当前用户的会话列表"""
    # 查询会话并统计消息数
    stmt = (
        select(
            Conversation,
            func.count(Message.id).label("message_count")
        )
        .join(Message, Message.conversation_id == Conversation.id, isouter=True)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.is_deleted == False)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    sessions = []
    for conv, msg_count in rows:
        session_data = {
            "id": conv.id,
            "user_id": conv.user_id,
            "title": conv.title,
            "model": conv.model,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        }
        sessions.append(ConversationRead(**session_data))
    
    return sessions


@router.post("/", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建新会话"""
    conversation = Conversation(
        user_id=current_user.id,
        title=request.title or "新对话",
        model=request.model
    )
    
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    
    return ConversationRead(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        model=conversation.model,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("/{session_id}", response_model=ConversationRead)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取会话详情"""
    conv = await db.get(Conversation, session_id)
    
    if not conv or conv.user_id != current_user.id or conv.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )
    
    return ConversationRead(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title,
        model=conv.model,
        created_at=conv.created_at,
        updated_at=conv.updated_at
    )


@router.get("/{session_id}/messages", response_model=List[MessageRead])
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    before_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取会话消息列表

    优先从 Redis 获取短期记忆，没有则从数据库查询
    """
    # 从 Redis 获取短期记忆（Redis 使用字符串 key）
    memory = MemoryService()
    context = await memory.get_context(session_id)
    
    if context:
        # 有 Redis 缓存，返回缓存的消息
        from datetime import datetime
        messages = []
        for msg in context:
            messages.append(MessageRead(
                id=0,
                conversation_id=0,
                role=msg["role"],
                content=msg["content"],
                tool_calls=None,
                tokens_input=None,
                tokens_output=None,
                latency_ms=None,
                created_at=datetime.fromtimestamp(msg.get("ts", 0))
            ))
        return messages[-limit:]
    
    # 从数据库查询
    try:
        conv_id = int(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    # 验证会话所有权
    conv = await db.get(Conversation, conv_id)
    if not conv or conv.user_id != current_user.id or conv.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )
    
    stmt = select(Message).where(Message.conversation_id == conv_id)
    
    if before_id:
        stmt = stmt.where(Message.id < before_id)
    
    stmt = stmt.order_by(Message.created_at.desc()).limit(limit)
    
    result = await db.execute(stmt)
    messages = result.scalars().all()
    
    return [MessageRead(
        id=m.id,
        conversation_id=m.conversation_id,
        role=m.role,
        content=m.content,
        tool_calls=m.tool_calls,
        tokens_input=m.tokens_input,
        tokens_output=m.tokens_output,
        latency_ms=m.latency_ms,
        created_at=m.created_at
    ) for m in reversed(messages)]


@router.delete("/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除会话（软删除）"""
    conv = await db.get(Conversation, session_id)
    
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )
    
    # 软删除
    conv.is_deleted = True
    await db.commit()
    
    # 清除 Redis 缓存
    memory = MemoryService()
    await memory.clear_context(str(session_id))
    
    return {"status": "deleted", "session_id": session_id}


@router.delete("/{session_id}/cache")
async def clear_session_cache(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """清除会话缓存"""
    memory = MemoryService()
    await memory.clear_context(session_id)
    
    return {"status": "ok", "message": "Session cache cleared"}

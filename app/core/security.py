"""
安全模块（满血版本）

功能：
- JWT Token 生成和验证
- 密码哈希（bcrypt）
- OAuth2 认证依赖
- 权限检查
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Union

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.models.user import User

# OAuth2 方案
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


def _sha256_hash(password: str) -> bytes:
    """将密码转换为 SHA256 哈希字节（避免 bcrypt 72 字节限制）"""
    return hashlib.sha256(password.encode('utf-8')).digest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（使用 SHA256 预处理避免 bcrypt 72 字节限制）"""
    # 先对密码进行 SHA256 哈希，得到 32 字节（256位），再用 bcrypt
    sha256_bytes = _sha256_hash(plain_password)
    # bcrypt 检查
    return bcrypt.checkpw(sha256_bytes, hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """获取密码哈希（使用 SHA256 预处理避免 bcrypt 72 字节限制）"""
    # 先对密码进行 SHA256 哈希，得到 32 字节（256位），再用 bcrypt
    sha256_bytes = _sha256_hash(password)
    # bcrypt 哈希，自动生成 salt
    hashed = bcrypt.hashpw(sha256_bytes, bcrypt.gensalt(rounds=12))
    return hashed.decode('utf-8')


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建 JWT Access Token
    
    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量
    
    Returns:
        JWT Token 字符串
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire, "type": "access"})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    创建 JWT Refresh Token
    
    Refresh Token 用于获取新的 Access Token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    解码 JWT Token
    
    Returns:
        Token  payload，如果验证失败返回 None
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    获取当前用户（依赖注入）
    
    验证 JWT Token 并返回对应的用户
    
    Raises:
        HTTPException: Token 无效或用户不存在
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    token_type: str = payload.get("type")
    
    if username is None or token_type != "access":
        raise credentials_exception
    
    # 查询用户
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户未激活"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前管理员用户"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user

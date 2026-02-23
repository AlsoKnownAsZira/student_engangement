"""
FastAPI dependency injection helpers — current user extraction, etc.
"""

from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.services import supabase_service

_bearer = HTTPBearer()


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """
    Validate the Supabase JWT and return ``{"user_id": …, "email": …}``.
    Raises 401 if the token is invalid / expired.
    """
    try:
        user = supabase_service.get_user_from_token(creds.credentials)
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

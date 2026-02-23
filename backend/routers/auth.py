"""
Auth router — signup / login / refresh (proxied to Supabase Auth).
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import (
    SignUpRequest,
    LoginRequest,
    AuthResponse,
    RefreshRequest,
)
from backend.services import supabase_service

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
async def signup(body: SignUpRequest):
    try:
        data = supabase_service.sign_up(body.email, body.password, body.full_name)
        return AuthResponse(**data)
    except Exception as e:
        log.exception("Signup failed for %s", body.email)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    try:
        data = supabase_service.sign_in(body.email, body.password)
        return AuthResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/refresh", response_model=AuthResponse)
async def refresh(body: RefreshRequest):
    try:
        data = supabase_service.refresh_session(body.refresh_token)
        return AuthResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

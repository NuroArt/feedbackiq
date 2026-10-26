"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.errors import AuthError


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Shared-secret auth on the workflow routes.

    Deliberately minimal: this service sits behind an ALB/ingress in a VPC and
    is called by internal schedulers, not browsers. `compare_digest` keeps the
    check constant-time. For multi-tenant use, swap this dependency for OIDC
    validation -- the routes don't change.
    """
    if not settings.api_key:
        return  # auth disabled (local dev)
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise AuthError("Missing or invalid X-API-Key header.")

"""
Request Size Limiting Middleware

Modul ini menyediakan middleware untuk membatasi ukuran request body.
Berguna untuk mencegah request yang terlalu besar yang bisa menyebabkan
memory issues atau denial of service.

Features:
    - Limit berdasarkan Content-Length header
    - Hanya berlaku untuk POST, PUT, PATCH methods
    - Configurable max size (default 10MB)
    - Return HTTP 413 jika request terlalu besar

Usage:
    from app.core.limit_request import RequestSizeLimitMiddleware
    
    # Di FastAPI app
    app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)

Note:
    Default limit 10MB cukup untuk kebanyakan LLM requests. Untuk use case
    dengan input sangat panjang (e.g., document processing), bisa dinaikkan.
"""

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # Default 10MB
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                return HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request body terlalu besar. Maksimum {self.max_size / (1024*1024):.1f}MB"
                )

        response = await call_next(request)
        return response

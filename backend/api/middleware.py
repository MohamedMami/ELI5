 # Custom middleware
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from core.config import settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse"""
    
    def __init__(self, app):
        super().__init__(app)
        # Store request times per IP
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
        # Track concurrent requests per IP
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        logger.info("Rate limiting middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Skip rate limiting for health checks
        if request.url.path.endswith('/health'):
            return await call_next(request)
        
        # Clean old requests (older than 1 minute)
        self._clean_old_requests(client_ip, current_time)
        
        # Check rate limit
        if len(self.requests[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
                    "retry_after": 60
                }
            )
        
        # Check concurrent requests
        if self.concurrent_requests[client_ip] >= settings.MAX_CONCURRENT_REQUESTS:
            logger.warning(f"Too many concurrent requests for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "too_many_concurrent_requests",
                    "detail": f"Too many concurrent requests. Maximum {settings.MAX_CONCURRENT_REQUESTS} concurrent requests.",
                    "retry_after": 10
                }
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        self.concurrent_requests[client_ip] += 1
        
        try:
            # Process request
            response = await call_next(request)
            return response
        finally:
            # Always decrement concurrent requests
            self.concurrent_requests[client_ip] -= 1
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies"""
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct client host
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, client_ip: str, current_time: float):
        """Remove requests older than 1 minute"""
        minute_ago = current_time - 60
        while (self.requests[client_ip] and 
               self.requests[client_ip][0] < minute_ago):
            self.requests[client_ip].popleft()

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Response: {response.status_code} "
                f"({process_time:.3f}s) {request.method} {request.url.path}"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log errors
            process_time = time.time() - start_time
            logger.error(
                f"Error: {str(e)} ({process_time:.3f}s) "
                f"{request.method} {request.url.path}"
            )
            raise

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
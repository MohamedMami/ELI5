# Caching logic
import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Dict
import hashlib
import logging
from datetime import datetime, timedelta
from core.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Handles caching with Redis fallback to in-memory cache"""
    
    def __init__(self):
        self.redis_client = None
        self._memory_cache = {}  # Fallback cache
        self._cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}
        
        # Initialize Redis if available
        if settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=False  # We'll handle encoding ourselves
                )
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed, using memory cache: {e}")
        else:
            logger.info("No Redis URL provided, using in-memory cache")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        
        key_string = ":".join(key_parts)
        # Hash long keys to avoid Redis key length limits
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        return key_string
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                # Try Redis first
                value = await self.redis_client.get(key)
                if value is not None:
                    self._cache_stats['hits'] += 1
                    return pickle.loads(value)
            else:
                # Use memory cache
                cache_entry = self._memory_cache.get(key)
                if cache_entry:
                    # Check if expired
                    if cache_entry['expires'] > datetime.now():
                        self._cache_stats['hits'] += 1
                        return cache_entry['value']
                    else:
                        # Remove expired entry
                        del self._memory_cache[key]
            
            self._cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            self._cache_stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if ttl is None:
                ttl = settings.CACHE_TTL
            
            if self.redis_client:
                # Use Redis
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)
            else:
                # Use memory cache
                expires = datetime.now() + timedelta(seconds=ttl)
                self._memory_cache[key] = {
                    'value': value,
                    'expires': expires
                }
                
                # Simple cleanup of expired entries (every 100 sets)
                if len(self._memory_cache) % 100 == 0:
                    await self._cleanup_memory_cache()
            
            self._cache_stats['sets'] += 1
            logger.debug(f"Cached value for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                return result > 0
            else:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
            return False
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            deleted_count = 0
            
            if self.redis_client:
                # Use Redis SCAN to find matching keys
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                    if keys:
                        deleted_count += await self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            else:
                # Memory cache pattern matching
                keys_to_delete = [
                    key for key in self._memory_cache.keys() 
                    if pattern.replace('*', '') in key
                ]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} cache entries matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern clear error for pattern '{pattern}': {e}")
            return 0
    
    async def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache"""
        try:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry['expires'] <= now
            ]
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.error(f"Memory cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_type': 'redis' if self.redis_client else 'memory',
            'stats': self._cache_stats.copy(),
            'memory_cache_size': len(self._memory_cache) if not self.redis_client else 0
        }
    
    # Convenience methods for common cache patterns
    
    def explanation_key(self, question: str, level: str, context_hash: str = "") -> str:
        """Generate cache key for explanations"""
        return self._generate_cache_key("explanation", question, level, context_hash)
    
    def document_key(self, document_id: str, operation: str = "") -> str:
        """Generate cache key for document operations"""
        return self._generate_cache_key("document", document_id, operation)

# Global cache manager instance
cache_manager = CacheManager()
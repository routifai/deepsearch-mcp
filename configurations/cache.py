#!/usr/bin/env python3
"""
Memory-Safe Caching System
Replaces unbounded data structures with proper cache management.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

class TTLCache:
    """Time-To-Live cache with size limits"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = []  # For LRU eviction
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        return datetime.now() - entry.created_at > self.ttl
    
    def _evict_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self._cache.items() 
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _evict_lru(self):
        """Evict least recently used entries when cache is full"""
        while len(self._cache) >= self.maxsize and self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._evict_expired()
        
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        if self._is_expired(entry):
            self._cache.pop(key)
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Move to end of access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry.value
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self._evict_expired()
        self._evict_lru()
        
        now = datetime.now()
        self._cache[key] = CacheEntry(value=value, created_at=now)
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._evict_expired()
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl.total_seconds(),
            "hit_rate": self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if not self._cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        if total_accesses == 0:
            return 0.0
        
        return len(self._cache) / total_accesses

class AnalysisCache:
    """Cache for query analysis results"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.cache = TTLCache(maxsize, ttl_seconds)
        self.hits = 0
        self.misses = 0
    
    def _cache_key(self, query: str, context: Dict = None) -> str:
        """Generate cache key for query and context"""
        data = {
            "query": query.lower().strip(),
            "context": context or {}
        }
        return hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    async def get_or_compute(self, query: str, compute_func, context: Dict = None):
        """Get from cache or compute and cache the result"""
        key = self._cache_key(query, context)
        
        # Try to get from cache
        result = self.cache.get(key)
        if result is not None:
            self.hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return result
        
        # Compute and cache
        self.misses += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        
        result = await compute_func(query, context)
        self.cache.set(key, result)
        return result
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class URLCache:
    """Memory-safe URL tracking to prevent infinite crawling"""
    
    def __init__(self, maxsize: int = 10000, ttl_seconds: int = 3600):
        self.seen_urls = TTLCache(maxsize, ttl_seconds)
        self.domain_counts = defaultdict(int)
        self.content_hashes = TTLCache(maxsize // 2, ttl_seconds)
        self._max_per_domain = 10
    
    def is_url_seen(self, url: str) -> bool:
        """Check if URL has been crawled recently"""
        return self.seen_urls.get(url) is not None
    
    def mark_url_seen(self, url: str):
        """Mark URL as seen"""
        self.seen_urls.set(url, True)
        
        # Update domain count
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        self.domain_counts[domain] += 1
    
    def is_domain_exhausted(self, url: str) -> bool:
        """Check if we've crawled too many pages from this domain"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return self.domain_counts[domain] >= self._max_per_domain
    
    def is_content_duplicate(self, content: str, title: str = "") -> bool:
        """Check if content is duplicate using hash"""
        # Create content fingerprint
        sample = (title + content[:500]).lower().strip()
        content_hash = hashlib.md5(sample.encode()).hexdigest()
        
        if self.content_hashes.get(content_hash):
            return True
        
        self.content_hashes.set(content_hash, True)
        return False
    
    def cleanup(self):
        """Manual cleanup - called periodically"""
        # Reset domain counts periodically
        if len(self.domain_counts) > 1000:
            self.domain_counts.clear()
            logger.info("Cleaned up domain counts cache")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "urls_seen": self.seen_urls.stats(),
            "content_hashes": self.content_hashes.stats(),
            "domain_count": len(self.domain_counts),
            "max_per_domain": self._max_per_domain
        }
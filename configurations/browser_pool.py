#!/usr/bin/env python3
"""
Fixed Browser Pool Management
Manages browser instances with correct Crawl4AI API usage.
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager
from crawl4ai import AsyncWebCrawler, BrowserConfig
from .config import config
from .exceptions import ResourceExhaustionError

logger = logging.getLogger(__name__)

class BrowserPool:
    """Manages a pool of browser instances for efficient reuse"""
    
    def __init__(self, pool_size: int = None):
        self.pool_size = pool_size or config.BROWSER_POOL_SIZE
        self.available_browsers = asyncio.Queue(maxsize=self.pool_size)
        self.total_browsers = 0
        self.in_use_count = 0
        self._initialized = False
        
        # Browser configuration
        self.browser_config = BrowserConfig(
            headless=True,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport_width=1920,
            viewport_height=1080,
            java_script_enabled=True,
            accept_downloads=False,
            ignore_https_errors=True
        )
    
    async def initialize(self):
        """Initialize the browser pool"""
        if self._initialized:
            return
        
        logger.info(f"Initializing browser pool with {self.pool_size} browsers")
        
        for i in range(self.pool_size):
            try:
                browser = AsyncWebCrawler(config=self.browser_config)
                # Note: Removed awarmup() as it doesn't exist in current Crawl4AI
                # Browsers will be warmed up on first use instead
                await self.available_browsers.put(browser)
                self.total_browsers += 1
                logger.debug(f"Created browser {i+1}/{self.pool_size}")
            except Exception as e:
                logger.error(f"Failed to create browser {i+1}: {e}")
                # Continue with fewer browsers rather than failing completely
        
        self._initialized = True
        logger.info(f"Browser pool initialized with {self.total_browsers} browsers")
    
    async def get_browser(self, timeout: float = 30.0) -> AsyncWebCrawler:
        """Get a browser from the pool"""
        if not self._initialized:
            await self.initialize()
        
        try:
            browser = await asyncio.wait_for(
                self.available_browsers.get(), 
                timeout=timeout
            )
            self.in_use_count += 1
            logger.debug(f"Browser acquired, {self.in_use_count} in use")
            return browser
        except asyncio.TimeoutError:
            raise ResourceExhaustionError(
                f"No browser available within {timeout} seconds",
                f"Pool size: {self.pool_size}, In use: {self.in_use_count}"
            )
    
    async def return_browser(self, browser: AsyncWebCrawler):
        """Return a browser to the pool"""
        try:
            await self.available_browsers.put(browser)
            self.in_use_count = max(0, self.in_use_count - 1)
            logger.debug(f"Browser returned, {self.in_use_count} in use")
        except Exception as e:
            logger.error(f"Failed to return browser to pool: {e}")
            # Don't crash if we can't return the browser
    
    @asynccontextmanager
    async def get_browser_context(self, timeout: float = 30.0):
        """Context manager for automatic browser return"""
        browser = None
        try:
            browser = await self.get_browser(timeout)
            yield browser
        finally:
            if browser:
                await self.return_browser(browser)
    
    async def close_all(self):
        """Close all browsers in the pool"""
        logger.info("Closing all browsers in pool")
        
        # Close browsers currently in the pool
        while not self.available_browsers.empty():
            try:
                browser = await self.available_browsers.get_nowait()
                await browser.aclose()
                self.total_browsers -= 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        
        logger.info(f"Closed browser pool, {self.total_browsers} browsers remaining")
    
    def get_stats(self) -> dict:
        """Get pool statistics"""
        return {
            "pool_size": self.pool_size,
            "total_browsers": self.total_browsers,
            "available_browsers": self.available_browsers.qsize(),
            "in_use_count": self.in_use_count,
            "initialized": self._initialized
        }

# Global browser pool instance
browser_pool = BrowserPool()

async def initialize_browser_pool():
    """Initialize the global browser pool"""
    await browser_pool.initialize()

async def cleanup_browser_pool():
    """Cleanup the global browser pool"""
    await browser_pool.close_all()
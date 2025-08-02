#!/usr/bin/env python3
"""
Simplified Web Content Fetcher
Single reliable content extraction method with proper resource management.
"""

import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime
from urllib.parse import urlparse
import re

from crawl4ai import CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter

from configurations.config import config
from configurations.browser_pool import browser_pool
from configurations.exceptions import FetchError, FetchTimeoutError, ContentExtractionError, ResourceExhaustionError

logger = logging.getLogger(__name__)

class WebFetcher:
    """Simplified web content fetcher with reliable extraction"""
    
    def __init__(self):
        self.timeout = config.TIMEOUT_SECONDS
        self.max_content_length = 100000  # 100KB limit for performance
        
    async def fetch_url(self, url: str, mode: str = "partial") -> str:
        """
        Fetch content from a single URL
        
        Args:
            url: URL to fetch
            mode: "snippet" (basic), "partial" (moderate), "complete" (full)
        """
        
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise FetchError(f"Invalid URL: {url}")
            
            logger.info(f"Fetching URL: {url} (mode: {mode})")
            
            # Get browser from pool
            async with browser_pool.get_browser_context(timeout=30.0) as browser:
                # Configure crawler based on mode
                config_obj = self._get_crawler_config(mode, url)
                
                # Fetch content
                result = await asyncio.wait_for(
                    browser.arun(url, config=config_obj),
                    timeout=self.timeout
                )
                
                if not getattr(result, 'success', False):
                    error_msg = getattr(result, 'error_message', 'Unknown error')
                    raise ContentExtractionError(f"Failed to fetch {url}: {error_msg}")
                
                # Extract content
                content = self._extract_content(result)
                
                # Format result
                return self._format_content(url, content, mode)
                
        except asyncio.TimeoutError:
            raise FetchTimeoutError(f"Timeout fetching {url}")
        except ResourceExhaustionError:
            raise  # Re-raise resource errors
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise FetchError(f"Error fetching {url}: {str(e)}")
    
    async def fetch_multiple_urls(self, urls: List[str], mode: str = "partial", 
                                 target_successful: Optional[int] = None) -> Dict[str, str]:
        """
        Fetch multiple URLs with adaptive strategy
        
        Args:
            urls: List of URLs to fetch
            mode: Fetch mode
            target_successful: Stop after this many successful fetches
        """
        
        if not urls:
            return {}
        
        logger.info(f"Fetching {len(urls)} URLs (target: {target_successful or 'all'})")
        
        # Limit concurrency to avoid overwhelming the system
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
        
        async def fetch_with_semaphore(url: str) -> tuple[str, str]:
            async with semaphore:
                try:
                    content = await self.fetch_url(url, mode)
                    return url, content
                except Exception as e:
                    return url, f"❌ Error: {str(e)}"
        
        results = {}
        successful_count = 0
        
        # Process URLs in batches if we have a target
        if target_successful and len(urls) > target_successful:
            # Process URLs until we get enough successful results
            remaining_urls = urls.copy()
            
            while successful_count < target_successful and remaining_urls:
                # Take batch of URLs
                batch_size = min(config.MAX_CONCURRENT, len(remaining_urls))
                current_batch = remaining_urls[:batch_size]
                remaining_urls = remaining_urls[batch_size:]
                
                # Process batch
                batch_tasks = [fetch_with_semaphore(url) for url in current_batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, Exception):
                        continue
                    
                    url, content = result
                    results[url] = content
                    
                    if not content.startswith("❌"):
                        successful_count += 1
                        if successful_count >= target_successful:
                            break
        else:
            # Fetch all URLs
            tasks = [fetch_with_semaphore(url) for url in urls]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in all_results:
                if isinstance(result, Exception):
                    continue
                url, content = result
                results[url] = content
                if not content.startswith("❌"):
                    successful_count += 1
        
        logger.info(f"Fetch completed: {successful_count} successful out of {len(results)} attempted")
        return results
    
    def _get_crawler_config(self, mode: str, url: str) -> CrawlerRunConfig:
        """Get crawler configuration based on mode and URL"""
        
        # Get site-specific CSS selectors
        css_selector = self._get_css_selector(url)
        
        # Content filter based on mode
        if mode == "snippet":
            content_filter = PruningContentFilter(threshold=0.3, threshold_type="fixed")
            word_threshold = 50
        elif mode == "partial":
            content_filter = PruningContentFilter(threshold=0.6, threshold_type="fixed")
            word_threshold = 100
        else:  # complete
            content_filter = PruningContentFilter(threshold=0.9, threshold_type="fixed")
            word_threshold = 1
        
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=config.PAGE_TIMEOUT * 1000,
            css_selector=css_selector,
            word_count_threshold=word_threshold,
            verbose=False
        )
    
    def _get_css_selector(self, url: str) -> str:
        """Get site-specific CSS selectors for better content extraction"""
        domain = urlparse(url).netloc.lower()
        
        if 'wikipedia.org' in domain:
            return '#mw-content-text, .mw-parser-output'
        elif 'github.com' in domain:
            return '.markdown-body, #readme'
        elif 'stackoverflow.com' in domain:
            return '.question, .answer, .post-text'
        elif 'medium.com' in domain:
            return 'article, .post-content'
        elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com']):
            return 'article, .story-body, .article-body'
        else:
            return 'article, main, .content, .post-content, .entry-content'
    
    def _extract_content(self, result) -> str:
        """Extract content from crawler result"""
        
        # Try different content sources in order of preference
        content = None
        
        # 1. Try markdown content
        if hasattr(result, 'markdown') and result.markdown:
            content = str(result.markdown)
        
        # 2. Try cleaned HTML
        elif hasattr(result, 'cleaned_html') and result.cleaned_html:
            content = self._html_to_text(str(result.cleaned_html))
        
        # 3. Fall back to text
        elif hasattr(result, 'text') and result.text:
            content = str(result.text)
        
        if not content or len(content.strip()) < 50:
            raise ContentExtractionError("No meaningful content found")
        
        # Clean and limit content
        content = self._clean_content(content)
        
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "... [Content truncated]"
        
        return content
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text"""
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert basic formatting
        html = re.sub(r'<h[1-6][^>]*>', '\n## ', html)
        html = re.sub(r'</h[1-6]>', '\n', html)
        html = re.sub(r'<p[^>]*>', '\n', html)
        html = re.sub(r'<br[^>]*>', '\n', html)
        
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        
        return html.strip()
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove common navigation text
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            # Skip navigation-like content
            lower_line = line.lower()
            if any(nav in lower_line for nav in ['skip to', 'menu', 'navigation', 'cookie']):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _format_content(self, url: str, content: str, mode: str) -> str:
        """Format the final content output"""
        
        # Extract title from content if possible
        lines = content.split('\n')
        title = "Content"
        for line in lines[:5]:
            if line.strip() and not line.startswith('#'):
                title = line.strip()[:100]
                break
        
        # Get domain for context
        domain = urlparse(url).netloc
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted = f"""# {title}

**Source:** {domain}
**URL:** {url}
**Extracted:** {timestamp}
**Mode:** {mode.title()}

---

{content}"""

        return formatted
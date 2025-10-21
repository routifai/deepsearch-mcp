"""
CHANGES FROM ORIGINAL:
1. Added MAX_CONTENT_LENGTH constant (25k chars - safe limit)
2. Added _truncate_content() method for post-filter truncation
3. Apply truncation in _browser_scrape() after getting filtered markdown
4. This ensures even filtered content stays within reasonable limits
5. Removed excessive comments, kept it clean
"""

import asyncio
import aiohttp
import time
import json
import gc
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import re
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Maximum content length per URL (characters) - REDUCED for memory management
MAX_CONTENT_LENGTH = 20000  # Reduced from 25k → 20k (~5k tokens)

class SimpleBrowserManager:
    """Simple browser lifecycle management with aggressive restart for memory"""
    def __init__(self):
        self.urls_processed = 0
        self.session_start = time.time()
        self.max_urls = 10  # CRITICAL: Reduced from 50 → 10 for memory management
        self.max_time = 900  # 15 minutes max
        
    def should_restart(self):
        return (self.urls_processed >= self.max_urls or 
                time.time() - self.session_start > self.max_time)
    
    def reset(self):
        self.urls_processed = 0
        self.session_start = time.time()
        # Force garbage collection on reset
        gc.collect()

@dataclass
class ScrapResult:
    url: str
    content: str
    metadata: Dict[str, Any]
    method: str
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0

class HybridScraper:
    def __init__(self):
        self.session = None
        self.crawler = None
        self.browser_manager = SimpleBrowserManager()
        
        # More aggressive content filtering
        self.content_filter = PruningContentFilter(
            threshold=0.55,  # Increased from 0.48 - more aggressive
            threshold_type="fixed",
            min_word_threshold=15,  # Increased from 10
        )
        
        self.markdown_generator = DefaultMarkdownGenerator(
            content_filter=self.content_filter,
            options={
                "ignore_links": False,
                "body_width": 0,
                "include_code": True,
                "include_tables": True,
                "citations": True,
            }
        )
        
        self.dynamic_indicators = [
            r'<script[^>]*>((?!<\/script>).)*react',
            r'<script[^>]*>((?!<\/script>).)*vue',
            r'<script[^>]*>((?!<\/script>).)*angular',
            r'<script[^>]*>((?!<\/script>).)*next\.js',
            r'document\.addEventListener\(["\']DOMContentLoaded',
            r'window\.onload',
            r'<div[^>]*id=["\']root["\']',
            r'<div[^>]*id=["\']app["\']',
            r'loading["\s]*[=:]["\s]*true',
            r'spa-|single.page',
        ]
        
        self.dynamic_pattern = re.compile('|'.join(self.dynamic_indicators), re.IGNORECASE)
        
    async def __aenter__(self):
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _init_session(self):
        """Initialize aiohttp session for fast HTTP requests"""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def _init_crawler(self):
        """Initialize Crawl4AI with performance optimizations"""
        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium",
            text_mode=True,
            light_mode=True,
            viewport_width=1280,
            viewport_height=800,
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
            ]
        )
        
        self.crawler = AsyncWebCrawler(config=browser_config)
        await self.crawler.__aenter__()
        # Only log in HTTP mode to avoid MCP JSON parsing errors
        import sys
        is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
        if not is_stdio_mode:
            print(f"Browser initialized (URLs processed: {self.browser_manager.urls_processed})")
    
    async def _restart_browser(self):
        """Restart browser when needed"""
        if self.crawler:
            # Only log in HTTP mode to avoid MCP JSON parsing errors
            import sys
            is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
            if not is_stdio_mode:
                print(f"🔄 Restarting browser after {self.browser_manager.urls_processed} URLs")
            try:
                await self.crawler.__aexit__(None, None, None)
            finally:
                self.crawler = None  # Critical: clear reference
            
            # Force garbage collection after closing browser
            gc.collect()
            
            await asyncio.sleep(1)  # Give it a moment
        await self._init_crawler()
        self.browser_manager.reset()
    
    def _truncate_content(self, content: str, max_length: int = MAX_CONTENT_LENGTH) -> Tuple[str, bool]:
        """
        Truncate content to max_length with smart truncation.
        Returns (truncated_content, was_truncated)
        """
        if len(content) <= max_length:
            return content, False
        
        # Smart truncation: try to end at paragraph or sentence
        truncated = content[:max_length]
        
        # Try to find last paragraph break (double newline)
        last_para = truncated.rfind('\n\n')
        if last_para > max_length * 0.75:  # At least 75% of content
            truncated = truncated[:last_para]
        else:
            # Try to find last sentence
            last_period = truncated.rfind('. ')
            if last_period > max_length * 0.75:
                truncated = truncated[:last_period + 1]
        
        truncated += "\n\n[Content truncated for size - this is a filtered excerpt]"
        return truncated, True
    
    def _extract_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """Extract basic metadata from HTML content"""
        metadata = {
            'url': url,
            'content_length': len(content),
            'has_forms': '<form' in content.lower(),
            'has_scripts': '<script' in content.lower(),
            'domain': urlparse(url).netloc,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            metadata['title'] = title_match.group(1).strip()[:200]
        
        # Extract meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', content, re.IGNORECASE)
        if desc_match:
            metadata['description'] = desc_match.group(1).strip()[:300]
        
        # Count links and images
        metadata['link_count'] = len(re.findall(r'<a[^>]*href=', content, re.IGNORECASE))
        metadata['image_count'] = len(re.findall(r'<img[^>]*src=', content, re.IGNORECASE))
        
        return metadata
    
    def _needs_browser(self, content: str, url: str) -> bool:
        """Determine if content needs browser rendering"""
        if len(content) < 1000:
            return True
            
        if self.dynamic_pattern.search(content):
            return True
            
        text_content = re.sub(r'<[^>]+>', '', content)
        if len(text_content.strip()) < 100:
            return True
            
        if any(pattern in content.lower() for pattern in [
            'id="root"', 'id="app"', 'class="app"',
            'loading...', 'please enable javascript'
        ]):
            return True
            
        return False
    
    async def _http_scrape(self, url: str) -> ScrapResult:
        """Fast HTTP-based scraping"""
        start_time = time.time()
        
        try:
            async with self.session.get(url, allow_redirects=True) as response:
                if response.status == 200:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    if self._needs_browser(content, url):
                        return ScrapResult(
                            url=url,
                            content="",
                            metadata={},
                            method="http_failed",
                            success=False,
                            error="Requires browser rendering",
                            response_time=response_time
                        )
                    
                    # Apply truncation to HTTP content too
                    content, was_truncated = self._truncate_content(content)
                    
                    metadata = self._extract_metadata(content, url)
                    metadata['status_code'] = response.status
                    metadata['response_time'] = response_time
                    metadata['was_truncated'] = was_truncated
                    
                    return ScrapResult(
                        url=url,
                        content=content,
                        metadata=metadata,
                        method="http",
                        success=True,
                        response_time=response_time
                    )
                else:
                    return ScrapResult(
                        url=url,
                        content="",
                        metadata={'status_code': response.status},
                        method="http",
                        success=False,
                        error=f"HTTP {response.status}",
                        response_time=time.time() - start_time
                    )
                    
        except Exception as e:
            return ScrapResult(
                url=url,
                content="",
                metadata={},
                method="http",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _browser_scrape(self, url: str) -> ScrapResult:
        """Browser-based scraping with Crawl4AI using content filters"""
        start_time = time.time()
        
        # Initialize browser if needed
        if self.crawler is None:
            await self._init_crawler()
        
        # Check if we should restart browser
        if self.browser_manager.should_restart():
            await self._restart_browser()
        
        try:
            # NOW PROPERLY USING the content filter and markdown generator!
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=10,
                page_timeout=15000,
                excluded_tags=["script", "style"],
                exclude_external_images=True,
                remove_overlay_elements=True,
                delay_before_return_html=1.0,
                verbose=False,
                
                # ✅ USING THE IMPORTED MODULES HERE:
                markdown_generator=self.markdown_generator,  # Uses PruningContentFilter internally
            )
            
            result = await self.crawler.arun(url, config=run_config)
            response_time = time.time() - start_time
            
            if result.success:
                # Extract the filtered content
                content = ""
                content_type = "unknown"
                
                if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                    content = result.markdown.fit_markdown
                    content_type = "filtered_markdown"
                elif hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
                    content = result.markdown.raw_markdown  
                    content_type = "raw_markdown"
                elif isinstance(result.markdown, str):
                    content = result.markdown
                    content_type = "string_markdown"
                else:
                    content = result.cleaned_html or result.html
                    content_type = "html_fallback"
                
                # ✅ CRITICAL: Apply post-filter truncation
                # Even filtered markdown can be too large for JSON responses
                original_length = len(content)
                content, was_truncated = self._truncate_content(content, MAX_CONTENT_LENGTH)
                
                metadata = {
                    'url': result.url or url,
                    'final_url': result.url,
                    'status_code': result.status_code,
                    'content_length': len(content),
                    'original_length': original_length,
                    'was_truncated': was_truncated,
                    'response_time': response_time,
                    'method': 'browser',
                    'content_type': content_type,
                    'scraped_at': datetime.now().isoformat(),
                    'has_js': bool(result.js_execution_result),
                    'content_filtered': content_type == "filtered_markdown",
                }
                
                # Extract links and media info
                if result.links:
                    metadata['links_found'] = len(result.links.get('internal', [])) + len(result.links.get('external', []))
                if result.media:
                    metadata['images_found'] = len(result.media.get('images', []))
                
                # Extract title
                if result.metadata and hasattr(result.metadata, 'title'):
                    metadata['title'] = result.metadata.title
                else:
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', result.html or '', re.IGNORECASE | re.DOTALL)
                    if title_match:
                        metadata['title'] = title_match.group(1).strip()[:200]
                
                # Update browser manager counter
                self.browser_manager.urls_processed += 1
                
                # Force garbage collection after each browser scrape
                gc.collect()
                
                return ScrapResult(
                    url=url,
                    content=content,
                    metadata=metadata,
                    method="browser",
                    success=True,
                    response_time=response_time
                )
            else:
                return ScrapResult(
                    url=url,
                    content="",
                    metadata={
                        'error': result.error_message,
                        'status_code': result.status_code,
                        'scraped_at': datetime.now().isoformat()
                    },
                    method="browser",
                    success=False,
                    error=result.error_message,
                    response_time=response_time
                )
                
        except Exception as e:
            return ScrapResult(
                url=url,
                content="",
                metadata={'scraped_at': datetime.now().isoformat()},
                method="browser",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _scrape_single(self, url: str) -> ScrapResult:
        """Scrape a single URL with hybrid approach"""
        # First attempt: Fast HTTP
        http_result = await self._http_scrape(url)
        
        # If HTTP worked, return it
        if http_result.success:
            return http_result
        
        # If HTTP failed or content needs browser, use Crawl4AI with content filtering
        return await self._browser_scrape(url)
    
    async def scrape_urls(self, urls: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs with hybrid approach and content filtering
        """
        if not urls:
            return []
        
        start_time = time.time()
        print(f"Starting scrape of {len(urls)} URLs with content filtering and truncation...")
        
        tasks = [self._scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        formatted_results = []
        successful_scrapes = 0
        filtered_content_count = 0
        truncated_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    'url': urls[i],
                    'content': '',
                    'metadata': {
                        'error': str(result),
                        'success': False,
                        'method': 'error',
                        'scraped_at': datetime.now().isoformat()
                    }
                })
            else:
                if result.success:
                    successful_scrapes += 1
                    if result.metadata.get('content_filtered'):
                        filtered_content_count += 1
                    if result.metadata.get('was_truncated'):
                        truncated_count += 1
                    
                formatted_results.append({
                    'url': result.url,
                    'content': result.content,
                    'metadata': {
                        **result.metadata,
                        'success': result.success,
                        'method': result.method,
                        'error': result.error,
                        'response_time': result.response_time
                    }
                })
        
        total_time = time.time() - start_time
        
        batch_metadata = {
            'total_batch_time': total_time,
            'urls_per_second': len(urls) / total_time if total_time > 0 else 0,
            'total_urls': len(urls),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': len(urls) - successful_scrapes,
            'content_filtered_count': filtered_content_count,
            'content_truncated_count': truncated_count,
            'average_response_time': total_time / len(urls),
            'timestamp': datetime.now().isoformat(),
            'max_content_length': MAX_CONTENT_LENGTH,
        }
        
        for result in formatted_results:
            result['metadata'].update(batch_metadata)
        
        # Force garbage collection after batch processing
        gc.collect()
        
        # Save to JSON file if specified
        if output_file:
            self._save_to_json(formatted_results, output_file, batch_metadata)
        
        return formatted_results
    
    def _save_to_json(self, results: List[Dict], filename: str, batch_metadata: Dict):
        """Save results to JSON file with proper formatting"""
        output_data = {
            'batch_metadata': batch_metadata,
            'results': results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            # Only log in HTTP mode to avoid MCP JSON parsing errors
            import sys
            is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
            if not is_stdio_mode:
                print(f"Results saved to {filename}")
        except Exception as e:
            import sys
            is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
            if not is_stdio_mode:
                print(f"Error saving to JSON: {e}")


class WebFetcher:
    """WebFetcher class for MCP server compatibility"""
    
    def __init__(self):
        self.scraper = None
    
    async def fetch_url(self, url: str, mode: str = "partial") -> str:
        """
        Fetch content from a single URL with automatic truncation
        
        Args:
            url: URL to fetch
            mode: Fetch mode (partial/full) - currently only partial is supported
            
        Returns:
            Extracted and truncated content from the URL
        """
        if not self.scraper:
            self.scraper = HybridScraper()
            await self.scraper.__aenter__()
        
        try:
            result = await self.scraper._scrape_single(url)
            if result.success:
                # Force garbage collection after fetch
                gc.collect()
                return result.content
            else:
                return f"Error fetching {url}: {result.error}"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    async def __aenter__(self):
        if not self.scraper:
            self.scraper = HybridScraper()
            await self.scraper.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.scraper:
            await self.scraper.__aexit__(exc_type, exc_val, exc_tb)

async def scrap_urls(urls: List[str], output_file: str = None) -> List[Dict[str, Any]]:
    """
    High-performance URL scraper with intelligent content filtering and truncation
    
    Args:
        urls: List of URLs to scrape
        output_file: Optional JSON file to save results
        
    Returns:
        List of dictionaries with filtered, truncated, high-quality content
    """
    async with HybridScraper() as scraper:
        return await scraper.scrape_urls(urls, output_file)


async def main():
    """Example usage"""
    test_urls = [
        "https://example.com",
        "https://news.ycombinator.com",
    ]
    
    print(f"Scraping {len(test_urls)} URLs...")
    results = await scrap_urls(test_urls)
    
    for result in results:
        metadata = result['metadata']
        print(f"\nURL: {result['url']}")
        print(f"   Success: {metadata['success']}")
        print(f"   Content: {len(result['content']):,} chars")
        if metadata.get('was_truncated'):
            print(f"   Truncated from {metadata.get('original_length', 0):,} chars")


if __name__ == "__main__":
    asyncio.run(main())
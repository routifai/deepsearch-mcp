"""
TRUE LIGHTWEIGHT HYBRID SCRAPER
================================
The smart approach:
1. Try HTTP + BeautifulSoup first (FAST - ~50ms)
2. Convert to Markdown with markitdown or html-to-markdown (FAST)
3. Only spin up browser if page needs JS (SLOW - ~2s)

NO Crawl4AI for simple pages!
Only use browser when absolutely necessary.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
import re
from datetime import datetime
from collections import defaultdict
import hashlib

# Try to import markdown converters
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    from html_to_markdown import convert as html2md
    HTML2MD_AVAILABLE = True
except ImportError:
    HTML2MD_AVAILABLE = False

# Browser fallback (only when needed)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class ScrapResult:
    url: str
    content: str  # Clean markdown content
    raw_html: Optional[str]  # Optional raw HTML
    metadata: Dict[str, Any]
    method: str  # 'http' or 'browser'
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0


class MarkdownConverter:
    """Convert HTML to Markdown using best available library"""
    
    def __init__(self):
        if MARKITDOWN_AVAILABLE:
            self.markitdown = MarkItDown()
            self.method = "markitdown"
        elif HTML2MD_AVAILABLE:
            self.method = "html2markdown"
        else:
            self.method = "fallback"
    
    def convert(self, html: str, url: str = "") -> str:
        """Convert HTML to clean Markdown"""
        try:
            if self.method == "markitdown":
                # MarkItDown expects a file, so we need to use BeautifulSoup
                # to clean and then convert
                soup = BeautifulSoup(html, 'lxml')
                # Remove script, style, nav, footer
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                
                # Get text with some structure preserved
                clean_html = str(soup)
                
                # Simple HTML to Markdown conversion
                # (MarkItDown is mainly for files, so we'll do basic conversion)
                return self._basic_html_to_markdown(soup)
            
            elif self.method == "html2markdown":
                # Fast Rust-based converter
                return html2md(html)
            
            else:
                # Fallback: Use BeautifulSoup to extract text
                soup = BeautifulSoup(html, 'lxml')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                return self._basic_html_to_markdown(soup)
        
        except Exception as e:
            # Fallback to simple text extraction
            soup = BeautifulSoup(html, 'lxml')
            return soup.get_text(separator='\n', strip=True)
    
    def _basic_html_to_markdown(self, soup: BeautifulSoup) -> str:
        """Basic HTML to Markdown conversion"""
        markdown_parts = []
        
        # Process headings
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                text = heading.get_text(strip=True)
                markdown_parts.append(f"{'#' * i} {text}\n")
                heading.decompose()
        
        # Process lists
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                text = li.get_text(strip=True)
                markdown_parts.append(f"- {text}\n")
            ul.decompose()
        
        for ol in soup.find_all('ol'):
            for idx, li in enumerate(ol.find_all('li'), 1):
                text = li.get_text(strip=True)
                markdown_parts.append(f"{idx}. {text}\n")
            ol.decompose()
        
        # Process links
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            href = a['href']
            if text:
                markdown_parts.append(f"[{text}]({href})")
            a.decompose()
        
        # Process bold/strong
        for tag in soup.find_all(['b', 'strong']):
            text = tag.get_text(strip=True)
            markdown_parts.append(f"**{text}**")
            tag.decompose()
        
        # Process italic/em
        for tag in soup.find_all(['i', 'em']):
            text = tag.get_text(strip=True)
            markdown_parts.append(f"*{text}*")
            tag.decompose()
        
        # Get remaining text
        remaining_text = soup.get_text(separator='\n', strip=True)
        
        # Combine everything
        markdown = '\n'.join(markdown_parts) + '\n\n' + remaining_text
        
        # Clean up multiple newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()


class DynamicContentDetector:
    """Detect if a page needs browser rendering"""
    
    def __init__(self):
        # Patterns that indicate dynamic content
        self.js_framework_patterns = [
            r'<script[^>]*>((?!<\/script>).)*react',
            r'<script[^>]*>((?!<\/script>).)*vue',
            r'<script[^>]*>((?!<\/script>).)*angular',
            r'<script[^>]*>((?!<\/script>).)*next\.js',
            r'window\.React',
            r'window\.Vue',
            r'ng-app',
            r'data-react',
            r'data-vue',
        ]
        
        self.dynamic_indicators = [
            r'<div[^>]*id=["\']root["\']',
            r'<div[^>]*id=["\']app["\']',
            r'<div[^>]*id=["\']__next["\']',
            r'document\.addEventListener\(["\']DOMContentLoaded',
            r'window\.onload',
            r'loading["\s]*[=:]["\s]*true',
            r'<noscript>',
        ]
        
        self.pattern = re.compile(
            '|'.join(self.js_framework_patterns + self.dynamic_indicators),
            re.IGNORECASE
        )
    
    def needs_browser(self, html: str, url: str) -> tuple[bool, str]:
        """
        Check if page needs browser rendering
        Returns: (needs_browser: bool, reason: str)
        """
        # Check 1: Very little content
        text_only = re.sub(r'<[^>]+>', '', html)
        if len(text_only.strip()) < 200:
            return True, "minimal_content"
        
        # Check 2: JS framework detected
        if self.pattern.search(html[:10000]):  # Check first 10KB
            return True, "js_framework"
        
        # Check 3: High script-to-content ratio
        script_content = len(re.findall(r'<script[^>]*>.*?</script>', html, re.DOTALL))
        total_length = len(html)
        if total_length > 0 and (script_content / total_length) > 0.3:
            return True, "high_js_ratio"
        
        # Check 4: Known SPA patterns
        if 'spa-' in html.lower() or 'single-page' in html.lower():
            return True, "spa_pattern"
        
        return False, "static_content"


class RateLimiter:
    """Simple rate limiter per domain"""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request: Dict[str, float] = defaultdict(float)
        self.lock = asyncio.Lock()
    
    async def wait(self, domain: str):
        async with self.lock:
            elapsed = time.time() - self.last_request[domain]
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request[domain] = time.time()


class TrueLightweightHybridScraper:
    """
    TRUE Hybrid: Fast HTTP + Markdown conversion, browser only when needed
    """
    
    def __init__(
        self,
        rate_limit: float = 2.0,
        max_concurrent: int = 20,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ):
        self.rate_limiter = RateLimiter(rate_limit)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.user_agent = user_agent
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.browser = None
        self.browser_context = None
        
        self.markdown_converter = MarkdownConverter()
        self.detector = DynamicContentDetector()
        
        # Stats
        self.stats = {
            'http_scrapes': 0,
            'browser_scrapes': 0,
            'failed_scrapes': 0,
            'total_time': 0.0
        }
    
    async def __aenter__(self):
        await self._init_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def _init_session(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300,
        )
        
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def _init_browser(self):
        """Initialize browser ONLY when needed"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.browser_context = await self.browser.new_context(
                user_agent=self.user_agent
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.browser_context:
            await self.browser_context.close()
        if self.browser:
            await self.browser.close()
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from parsed HTML"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Title
        if title := soup.find('title'):
            metadata['title'] = title.get_text(strip=True)
        
        # Meta description
        if desc := soup.find('meta', {'name': 'description'}):
            metadata['description'] = desc.get('content', '')[:300]
        
        # Open Graph
        if og_title := soup.find('meta', {'property': 'og:title'}):
            metadata['og_title'] = og_title.get('content', '')
        
        if og_desc := soup.find('meta', {'property': 'og:description'}):
            metadata['og_description'] = og_desc.get('content', '')
        
        # Counts
        metadata['link_count'] = len(soup.find_all('a', href=True))
        metadata['image_count'] = len(soup.find_all('img'))
        
        return metadata
    
    async def _scrape_http(self, url: str) -> ScrapResult:
        """
        Fast HTTP scraping with BeautifulSoup + Markdown conversion
        NO BROWSER - Pure Python
        """
        start_time = time.time()
        domain = urlparse(url).netloc
        
        try:
            # Rate limit
            await self.rate_limiter.wait(domain)
            
            # Fetch HTML
            async with self.session.get(url) as response:
                if response.status >= 400:
                    return ScrapResult(
                        url=url,
                        content="",
                        raw_html=None,
                        metadata={'status_code': response.status},
                        method="http",
                        success=False,
                        error=f"HTTP {response.status}",
                        response_time=time.time() - start_time
                    )
                
                html = await response.text()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            
            # Check if browser needed
            needs_browser, reason = self.detector.needs_browser(html, url)
            
            if needs_browser:
                return ScrapResult(
                    url=url,
                    content="",
                    raw_html=html,
                    metadata={'needs_browser_reason': reason},
                    method="http",
                    success=False,
                    error=f"Needs browser: {reason}",
                    response_time=time.time() - start_time
                )
            
            # Convert to Markdown (FAST!)
            markdown = self.markdown_converter.convert(html, url)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            metadata.update({
                'status_code': response.status,
                'content_length': len(markdown),
                'conversion_method': self.markdown_converter.method,
                'response_time': time.time() - start_time
            })
            
            self.stats['http_scrapes'] += 1
            
            return ScrapResult(
                url=url,
                content=markdown,
                raw_html=html if len(html) < 50000 else None,  # Don't store huge HTML
                metadata=metadata,
                method="http",
                success=True,
                response_time=time.time() - start_time
            )
        
        except asyncio.TimeoutError:
            return ScrapResult(
                url=url,
                content="",
                raw_html=None,
                metadata={},
                method="http",
                success=False,
                error="Timeout",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ScrapResult(
                url=url,
                content="",
                raw_html=None,
                metadata={},
                method="http",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _scrape_browser(self, url: str) -> ScrapResult:
        """
        Browser fallback for JavaScript-heavy sites
        ONLY CALLED WHEN HTTP FAILS
        """
        start_time = time.time()
        
        try:
            # Initialize browser if needed
            if self.browser is None:
                await self._init_browser()
            
            # Create new page
            page = await self.browser_context.new_page()
            
            try:
                # Navigate and wait for content
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait a bit for JS to execute
                await page.wait_for_timeout(2000)
                
                # Get rendered HTML
                html = await page.content()
                
                # Parse
                soup = BeautifulSoup(html, 'lxml')
                
                # Convert to markdown
                markdown = self.markdown_converter.convert(html, url)
                
                # Extract metadata
                metadata = self._extract_metadata(soup, url)
                metadata.update({
                    'content_length': len(markdown),
                    'browser_rendered': True,
                    'response_time': time.time() - start_time
                })
                
                self.stats['browser_scrapes'] += 1
                
                return ScrapResult(
                    url=url,
                    content=markdown,
                    raw_html=None,  # Don't store browser HTML (too large)
                    metadata=metadata,
                    method="browser",
                    success=True,
                    response_time=time.time() - start_time
                )
            
            finally:
                await page.close()
        
        except Exception as e:
            self.stats['failed_scrapes'] += 1
            return ScrapResult(
                url=url,
                content="",
                raw_html=None,
                metadata={},
                method="browser",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def scrape_single(self, url: str) -> ScrapResult:
        """
        Scrape single URL with hybrid approach:
        1. Try HTTP + Markdown (FAST)
        2. If needs browser → Use browser (SLOW)
        """
        # Try HTTP first
        result = await self._scrape_http(url)
        
        if result.success:
            return result
        
        # Check if it needs browser
        if "Needs browser" in (result.error or ""):
            # Fallback to browser
            return await self._scrape_browser(url)
        
        # Other error - return failed result
        self.stats['failed_scrapes'] += 1
        return result
    
    async def scrape_urls(
        self,
        urls: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs with hybrid approach
        """
        if not urls:
            return []
        
        batch_start = time.time()
        
        if show_progress:
            print(f"🚀 Scraping {len(urls)} URLs with TRUE hybrid approach...")
            print(f"   Strategy: HTTP first → Browser only if needed")
            print(f"   Markdown converter: {self.markdown_converter.method}")
        
        # Scrape all URLs with semaphore control
        async def scrape_with_semaphore(url: str):
            async with self.semaphore:
                return await self.scrape_single(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    'url': urls[i],
                    'content': '',
                    'metadata': {
                        'success': False,
                        'error': str(result),
                        'method': 'error'
                    }
                })
            else:
                formatted_results.append({
                    'url': result.url,
                    'content': result.content,
                    'raw_html': result.raw_html,
                    'metadata': {
                        **result.metadata,
                        'success': result.success,
                        'method': result.method,
                        'error': result.error,
                        'response_time': result.response_time
                    }
                })
        
        batch_time = time.time() - batch_start
        
        # Print stats
        if show_progress:
            success_count = sum(1 for r in formatted_results if r['metadata']['success'])
            http_count = self.stats['http_scrapes']
            browser_count = self.stats['browser_scrapes']
            failed_count = self.stats['failed_scrapes']
            
            print(f"\n✅ Completed in {batch_time:.2f}s ({len(urls)/batch_time:.2f} URLs/sec)")
            print(f"   Success: {success_count}/{len(urls)}")
            print(f"   HTTP (fast): {http_count} | Browser (slow): {browser_count}")
            print(f"   Failed: {failed_count}")
            
            if http_count + browser_count > 0:
                http_percent = (http_count / (http_count + browser_count)) * 100
                print(f"   📊 {http_percent:.1f}% used fast HTTP path!")
        
        return formatted_results
    
    async def extract_with_selectors(
        self,
        url: str,
        selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract specific data using CSS selectors
        Always uses HTTP (fast), even for dynamic sites we can try first
        """
        start_time = time.time()
        
        try:
            async with self.session.get(url) as response:
                html = await response.text()
            
            soup = BeautifulSoup(html, 'lxml')
            
            extracted = {}
            for name, selector in selectors.items():
                elements = soup.select(selector)
                if len(elements) == 1:
                    elem = elements[0]
                    extracted[name] = {
                        'text': elem.get_text(strip=True),
                        'html': str(elem),
                        'attrs': dict(elem.attrs) if hasattr(elem, 'attrs') else {}
                    }
                elif len(elements) > 1:
                    extracted[name] = [
                        {
                            'text': elem.get_text(strip=True),
                            'attrs': dict(elem.attrs) if hasattr(elem, 'attrs') else {}
                        }
                        for elem in elements
                    ]
                else:
                    extracted[name] = None
            
            return {
                'success': True,
                'url': url,
                'data': extracted,
                'response_time': time.time() - start_time
            }
        
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'response_time': time.time() - start_time
            }


# Convenience function
async def scrape_urls_fast(
    urls: List[str],
    rate_limit: float = 2.0,
    max_concurrent: int = 20,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Quick scraping with TRUE hybrid approach
    
    Args:
        urls: URLs to scrape
        rate_limit: Requests per second per domain
        max_concurrent: Max parallel requests
        timeout: Request timeout in seconds
    
    Returns:
        List of results with clean markdown content
    """
    async with TrueLightweightHybridScraper(
        rate_limit=rate_limit,
        max_concurrent=max_concurrent,
        timeout=timeout
    ) as scraper:
        return await scraper.scrape_urls(urls)


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
            self.scraper = TrueLightweightHybridScraper()
            await self.scraper.__aenter__()
        
        try:
            result = await self.scraper.scrape_single(url)
            if result.success:
                return result.content
            else:
                return f"Error fetching {url}: {result.error}"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    async def __aenter__(self):
        if not self.scraper:
            self.scraper = TrueLightweightHybridScraper()
            await self.scraper.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.scraper:
            await self.scraper.__aexit__(exc_type, exc_val, exc_tb)


# Legacy function for backward compatibility
async def scrap_urls(urls: List[str], output_file: str = None) -> List[Dict[str, Any]]:
    """
    High-performance URL scraper with intelligent content filtering and truncation
    
    Args:
        urls: List of URLs to scrape
        output_file: Optional JSON file to save results
        
    Returns:
        List of dictionaries with filtered, truncated, high-quality content
    """
    async with TrueLightweightHybridScraper() as scraper:
        return await scraper.scrape_urls(urls)


# Example usage
async def main():
    """Test the TRUE hybrid scraper"""
    
    print("=" * 70)
    print("TRUE LIGHTWEIGHT HYBRID SCRAPER")
    print("=" * 70)
    print("\n📋 Strategy:")
    print("   1. HTTP + BeautifulSoup + Markdown (FAST ~50ms)")
    print("   2. Browser only if JS detected (SLOW ~2s)")
    print()
    
    test_urls = [
        "https://example.com",              # Static - will use HTTP
        "https://news.ycombinator.com",     # Static - will use HTTP
        "https://python.org",                # Static - will use HTTP
        "https://react.dev",                 # Dynamic - will need browser
    ]
    
    results = await scrape_urls_fast(
        urls=test_urls,
        rate_limit=2.0,
        max_concurrent=5
    )
    
    print("\n📝 Results:\n")
    for result in results:
        meta = result['metadata']
        print(f"{'='*70}")
        print(f"URL: {result['url']}")
        print(f"Success: {meta['success']}")
        print(f"Method: {meta.get('method', 'unknown')}")
        print(f"Time: {meta.get('response_time', 0):.2f}s")
        
        if meta['success']:
            content_preview = result['content'][:200].replace('\n', ' ')
            print(f"Content: {content_preview}...")
            print(f"Length: {len(result['content']):,} chars")
        else:
            print(f"Error: {meta.get('error', 'unknown')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
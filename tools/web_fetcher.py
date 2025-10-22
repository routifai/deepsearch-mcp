"""
Optimized Hybrid Web Scraper

A high-performance web scraper that uses HTTP requests for static content
and Playwright browser automation as a fallback for JavaScript-heavy sites.

Features:
- Fast HTTP scraping with BeautifulSoup
- Intelligent detection of dynamic content
- Playwright fallback for JavaScript-heavy sites
- Rate limiting and concurrent request management
- SSL configuration support
- Minimal memory footprint
"""

import asyncio
import aiohttp
import time
import logging
import ssl
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from collections import defaultdict

# ============================================================================
# LOGGING CONFIGURATION - CONSOLE ONLY
# ============================================================================

logger = logging.getLogger('HybridScraper')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


@dataclass
class ScrapResult:
    url: str
    content: str
    raw_html: Optional[str]
    metadata: Dict[str, Any]
    method: str
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0


class TextExtractor:
    """Fast text extraction without markdown overhead"""
    
    def __init__(self):
        logger.info("📝 Text extractor initialized (optimized mode)")
    
    def extract(self, html: str, url: str = "") -> str:
        """Extract clean text from HTML - FAST"""
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove unwanted tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                tag.decompose()
            
            # Simple text extraction
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            logger.debug(f"   Extracted {len(clean_text):,} chars from {len(html):,} chars HTML")
            
            return clean_text
        
        except Exception as e:
            logger.warning(f"   Text extraction error: {e}")
            return ""


class DynamicContentDetector:
    """Detect if a page needs browser rendering"""
    
    def __init__(self):
        self.js_framework_patterns = [
            r'<script[^>]*>((?!<\/script>).)*react',
            r'<script[^>]*>((?!<\/script>).)*vue',
            r'<script[^>]*>((?!<\/script>).)*angular',
            r'window\.React',
            r'window\.Vue',
            r'ng-app',
            r'data-react',
        ]
        
        self.dynamic_indicators = [
            r'<div[^>]*id=["\']root["\']',
            r'<div[^>]*id=["\']app["\']',
            r'<div[^>]*id=["\']__next["\']',
            r'<noscript>',
        ]
        
        self.pattern = re.compile(
            '|'.join(self.js_framework_patterns + self.dynamic_indicators),
            re.IGNORECASE
        )
        
        logger.info("🔍 Dynamic content detector initialized")
    
    def needs_browser(self, html: str, url: str) -> tuple[bool, str]:
        """Check if page needs browser rendering"""
        logger.debug(f"   🔍 Running detection for {url}")
        
        # Check 1: Minimal content
        text_only = re.sub(r'<[^>]+>', '', html)
        text_length = len(text_only.strip())
        
        logger.debug(f"      ├─ Content length: {text_length} chars")
        
        if text_length < 200:
            logger.warning(f"      └─ ❌ MINIMAL CONTENT detected ({text_length} < 200 chars)")
            return True, "minimal_content"
        
        # Check 2: JS framework
        match = self.pattern.search(html[:10000])
        if match:
            detected = match.group(0)[:50]
            logger.warning(f"      └─ ❌ JS FRAMEWORK detected: {detected}...")
            return True, "js_framework"
        
        logger.debug(f"      ├─ No JS frameworks found")
        
        # Check 3: Script ratio
        scripts = re.findall(r'<script[^>]*>.*?</script>', html, re.DOTALL)
        script_size = sum(len(s) for s in scripts)
        total_size = len(html)
        ratio = script_size / total_size if total_size > 0 else 0
        
        logger.debug(f"      ├─ Script ratio: {ratio:.1%}")
        
        if ratio > 0.3:
            logger.warning(f"      └─ ❌ HIGH SCRIPT RATIO ({ratio:.1%} > 30%)")
            return True, "high_js_ratio"
        
        # Check 4: SPA patterns
        if 'spa-' in html.lower() or 'single-page' in html.lower():
            logger.warning(f"      └─ ❌ SPA PATTERN detected")
            return True, "spa_pattern"
        
        logger.info(f"      └─ ✅ STATIC CONTENT - HTTP will work!")
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
                wait_time = self.min_interval - elapsed
                logger.debug(f"      ⏱️  Rate limiting {domain}: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_request[domain] = time.time()


class OptimizedHybridScraper:
    """Optimized Hybrid Scraper - Fast HTTP + Simple text extraction"""
    
    def __init__(
        self,
        rate_limit: float = 2.0,
        max_concurrent: int = 20,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        verify_ssl: bool = True
    ):
        self.rate_limiter = RateLimiter(rate_limit)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.user_agent = user_agent
        self.verify_ssl = verify_ssl
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.browser = None
        self.browser_context = None
        
        self.text_extractor = TextExtractor()
        self.detector = DynamicContentDetector()
        
        self.stats = {
            'http_scrapes': 0,
            'browser_scrapes': 0,
            'failed_scrapes': 0
        }
        
        logger.info(f"🚀 Scraper initialized")
        logger.info(f"   Rate limit: {rate_limit} req/s")
        logger.info(f"   Max concurrent: {max_concurrent}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   SSL verify: {verify_ssl}")
    
    async def __aenter__(self):
        # Create SSL context
        if self.verify_ssl:
            ssl_context = ssl.create_default_context()
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("⚠️  SSL verification disabled")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=100)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': self.user_agent}
        )
        
        logger.info("✅ HTTP session created")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            logger.info("🔒 HTTP session closed")
        
        if self.browser_context:
            await self.browser_context.close()
            logger.info("🔒 Browser context closed")
        
        if self.browser:
            await self.browser.close()
            logger.info("🔒 Browser closed")
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract basic metadata"""
        metadata = {'url': url}
        
        try:
            if soup.title:
                metadata['title'] = soup.title.string.strip() if soup.title.string else ""
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                metadata['description'] = meta_desc['content'].strip()
            
        except Exception as e:
            logger.debug(f"   Metadata extraction error: {e}")
        
        return metadata
    
    async def _scrape_http(self, url: str) -> ScrapResult:
        """Fast HTTP scraping with simple text extraction"""
        start_time = time.time()
        
        try:
            domain = urlparse(url).netloc
            await self.rate_limiter.wait(domain)
            
            logger.info(f"⚡ HTTP: Fetching {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"   ⚠️  Status {response.status}")
                    return ScrapResult(
                        url=url, content="", raw_html=None, metadata={},
                        method="http", success=False,
                        error=f"HTTP {response.status}",
                        response_time=time.time() - start_time
                    )
                
                html = await response.text()
                logger.debug(f"   ✅ Got {len(html):,} bytes")
            
            # Check if browser needed
            needs_browser, reason = self.detector.needs_browser(html, url)
            
            if needs_browser:
                logger.warning(f"   ⚠️  Needs browser: {reason}")
                return ScrapResult(
                    url=url, content="", raw_html=None,
                    metadata={'detection_reason': reason},
                    method="http", success=False,
                    error=f"Needs browser: {reason}",
                    response_time=time.time() - start_time
                )
            
            # Extract text (FAST - no markdown conversion)
            soup = BeautifulSoup(html, 'lxml')
            text = self.text_extractor.extract(html, url)
            
            metadata = self._extract_metadata(soup, url)
            metadata.update({
                'content_length': len(text),
                'browser_rendered': False,
                'response_time': time.time() - start_time
            })
            
            self.stats['http_scrapes'] += 1
            
            elapsed = time.time() - start_time
            logger.info(f"✅ HTTP SUCCESS: {url} ({elapsed:.2f}s, {len(text):,} chars)")
            
            return ScrapResult(
                url=url, content=text, raw_html=None,
                metadata=metadata, method="http",
                success=True, response_time=elapsed
            )
        
        except asyncio.TimeoutError:
            self.stats['failed_scrapes'] += 1
            logger.error(f"❌ HTTP TIMEOUT: {url}")
            return ScrapResult(
                url=url, content="", raw_html=None, metadata={},
                method="http", success=False,
                error="Timeout", response_time=time.time() - start_time
            )
        
        except Exception as e:
            self.stats['failed_scrapes'] += 1
            logger.error(f"❌ HTTP FAILED: {url} - {str(e)[:100]}")
            return ScrapResult(
                url=url, content="", raw_html=None, metadata={},
                method="http", success=False,
                error=str(e), response_time=time.time() - start_time
            )
    
    async def _scrape_browser(self, url: str) -> ScrapResult:
        """Browser scraping fallback"""
        start_time = time.time()
        
        try:
            # Lazy load Playwright
            if not self.browser:
                logger.info(f"🌐 Initializing browser (first use)...")
                from playwright.async_api import async_playwright
                
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(headless=True)
                self.browser_context = await self.browser.new_context(
                    user_agent=self.user_agent
                )
                logger.info(f"   ✅ Browser ready")
            
            logger.info(f"🌐 BROWSER: Loading {url}")
            
            page = await self.browser_context.new_page()
            
            try:
                logger.debug(f"   🔄 Navigating...")
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                
                logger.debug(f"   ⏳ Waiting 2s for JavaScript...")
                await page.wait_for_timeout(2000)
                
                logger.debug(f"   📥 Getting rendered HTML...")
                html = await page.content()
                logger.debug(f"   ✅ Got {len(html):,} bytes")
                
                soup = BeautifulSoup(html, 'lxml')
                
                logger.debug(f"   📝 Extracting text...")
                text = self.text_extractor.extract(html, url)
                
                metadata = self._extract_metadata(soup, url)
                metadata.update({
                    'content_length': len(text),
                    'browser_rendered': True,
                    'response_time': time.time() - start_time
                })
                
                self.stats['browser_scrapes'] += 1
                
                elapsed = time.time() - start_time
                logger.info(f"✅ BROWSER SUCCESS: {url} ({elapsed:.2f}s, {len(text):,} chars)")
                
                return ScrapResult(
                    url=url, content=text, raw_html=None,
                    metadata=metadata, method="browser",
                    success=True, response_time=elapsed
                )
            
            finally:
                await page.close()
                logger.debug(f"   🔒 Browser page closed")
        
        except Exception as e:
            self.stats['failed_scrapes'] += 1
            logger.error(f"❌ BROWSER FAILED: {url} - {str(e)[:100]}")
            return ScrapResult(
                url=url, content="", raw_html=None, metadata={},
                method="browser", success=False,
                error=str(e), response_time=time.time() - start_time
            )
    
    async def scrape_single(self, url: str) -> ScrapResult:
        """Scrape single URL with hybrid approach"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 Starting: {url}")
        
        result = await self._scrape_http(url)
        
        if result.success:
            return result
        
        if result.error and "Needs browser" in result.error:
            logger.warning(f"↪️  Falling back to browser...")
            return await self._scrape_browser(url)
        
        self.stats['failed_scrapes'] += 1
        logger.error(f"❌ FAILED: {url}")
        return result
    
    async def scrape_urls(
        self,
        urls: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs"""
        if not urls:
            return []
        
        batch_start = time.time()
        
        if show_progress:
            logger.info(f"\n{'='*70}")
            logger.info(f"🚀 BATCH SCRAPE: {len(urls)} URLs")
            logger.info(f"   Strategy: HTTP first → Browser fallback")
            logger.info(f"{'='*70}\n")
        
        async def scrape_with_semaphore(url: str):
            async with self.semaphore:
                return await self.scrape_single(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    'url': urls[i], 'content': '',
                    'metadata': {'success': False, 'error': str(result), 'method': 'error'}
                })
            else:
                formatted_results.append({
                    'url': result.url, 'content': result.content,
                    'raw_html': result.raw_html,
                    'metadata': {**result.metadata, 'success': result.success,
                                'method': result.method, 'error': result.error,
                                'response_time': result.response_time}
                })
        
        batch_time = time.time() - batch_start
        
        if show_progress:
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 BATCH COMPLETE")
            logger.info(f"{'='*70}")
            
            success_count = sum(1 for r in formatted_results if r['metadata']['success'])
            http_count = self.stats['http_scrapes']
            browser_count = self.stats['browser_scrapes']
            failed_count = self.stats['failed_scrapes']
            
            logger.info(f"⏱️  Total time: {batch_time:.2f}s ({len(urls)/batch_time:.2f} URLs/sec)")
            logger.info(f"✅ Success: {success_count}/{len(urls)}")
            logger.info(f"⚡ HTTP (fast): {http_count}")
            logger.info(f"🌐 Browser (slow): {browser_count}")
            logger.info(f"❌ Failed: {failed_count}")
            
            if http_count + browser_count > 0:
                http_percent = (http_count / (http_count + browser_count)) * 100
                logger.info(f"📈 HTTP usage: {http_percent:.1f}%")
            
            logger.info(f"{'='*70}\n")
        
        return formatted_results


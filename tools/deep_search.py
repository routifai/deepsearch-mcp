#!/usr/bin/env python3
"""
Deep Search Tool - Pure Crawl4AI Implementation 🚀
FIXED VERSION: Uses built-in Crawl4AI link extraction for REAL deep crawling
"""

import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from collections import defaultdict

# Fixed Crawl4AI imports for v0.7.x
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.content_filter_strategy import PruningContentFilter

# Import your modules
from configurations.config import config as global_config
from configurations.cache import URLCache
from tools.search_engine import search_engine, SearchCategory
from configurations.exceptions import handle_error

logger = logging.getLogger(__name__)

@dataclass
class DeepSearchConfig:
    """Deep search configuration for different bragging levels"""
    max_depth: int = 3
    max_pages: int = 30
    max_per_domain: int = 8
    max_links_per_page: int = 8
    concurrent_crawlers: int = 5
    delay_seconds: float = 0.2
    timeout_seconds: int = 180

@dataclass
class DeepSearchStats:
    """Stats that will impress your colleagues"""
    pages_crawled: int = 0
    domains_hit: Set[str] = field(default_factory=set)
    total_words_extracted: int = 0
    total_processing_time: float = 0.0
    successful_extractions: int = 0
    failed_extractions: int = 0
    average_page_time: float = 0.0
    keywords_found: Dict[str, int] = field(default_factory=dict)
    depth_distribution: Dict[int, int] = field(default_factory=dict)
    links_extracted: int = 0
    links_followed: int = 0

class IntelligentCrawl4AIExtractor:
    """Pure Crawl4AI-based intelligent extraction using built-in features"""
    
    def __init__(self, keywords: List[str]):
        self.keywords = [k.lower() for k in keywords]
        self.primary_keywords = set(self.keywords[:3])
        self.secondary_keywords = set(self.keywords[3:])
        
        # Pre-compile filter patterns for better performance
        self.filter_patterns = self._get_filter_patterns()
        
    def _get_filter_patterns(self) -> List[str]:
        """Get regex patterns for filtering out unwanted links"""
        return [
            r'/login',
            r'/register', 
            r'/search\?',
            r'/tag/',
            r'/category/',
            r'\.pdf$',
            r'\.jpg$',
            r'\.png$',
            r'\.gif$',
            r'\.zip$',
            r'\.doc$',
            r'\.docx$',
            r'javascript:',
            r'mailto:',
            r'#',
            r'\?page=',
            r'\?p=',
            r'/admin',
            r'/user/',
            r'/profile',
            r'/settings',
            r'/privacy',
            r'/terms',
            r'/contact',
            r'/about$',
            r'/home$'
        ]
    
    def _get_content_selectors(self, url: str) -> str:
        """Get site-specific CSS selectors for content areas"""
        domain = urlparse(url).netloc.lower()
        
        if 'wikipedia.org' in domain:
            return '#mw-content-text, .mw-parser-output'
        elif 'github.com' in domain:
            return '.markdown-body, #readme, .Box-body'
        elif 'stackoverflow.com' in domain:
            return '.question, .answer, .post-text'
        elif 'medium.com' in domain or 'substack.com' in domain:
            return 'article, .post-content'
        elif any(news in domain for news in ['bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com', 'bloomberg.com']):
            return 'article, .story-body, .article-body, .story-content'
        elif 'yahoo.com' in domain:
            return 'article, .caas-body, .story-body'
        elif 'reddit.com' in domain:
            return '.Post, .thing, .content'
        elif 'arxiv.org' in domain:
            return '.abs, .abstract, .full-text'
        elif any(edu in domain for edu in ['.edu', 'stanford', 'mit']):
            return 'article, main, .content, .research'
        else:
            # Generic selectors prioritizing content areas - avoid complex attribute selectors
            return 'article, main, .content, .post-content, .entry-content'
    
    def extract_and_score_links(self, crawl_result, base_url: str, current_depth: int) -> List[Dict]:
        """Extract and score links using Crawl4AI's built-in links extraction"""
        
        scored_links = []
        
        # Use Crawl4AI's built-in links attribute - format: {"internal": [...], "external": [...]}
        if not hasattr(crawl_result, 'links') or not crawl_result.links:
            logger.debug(f"No links found in crawl result for {base_url}")
            return scored_links
        
        # Combine internal and external links
        all_links = []
        if 'internal' in crawl_result.links:
            all_links.extend(crawl_result.links['internal'])
        if 'external' in crawl_result.links:
            all_links.extend(crawl_result.links['external'])
        
        logger.debug(f"Found {len(all_links)} total links from {base_url} (internal: {len(crawl_result.links.get('internal', []))}, external: {len(crawl_result.links.get('external', []))})")
        
        for link_data in all_links:
            # Extract link information from Crawl4AI's link format
            href = link_data.get('href', '') or link_data.get('url', '')
            text = link_data.get('text', '').strip()
            title = link_data.get('title', '')
            
            if not href or not text:
                continue
            
            try:
                # Ensure absolute URL
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                # Filter out unwanted links
                if self._should_filter_link(full_url, text):
                    continue
                
                # Score this link
                score = self._calculate_intelligence_score(text, full_url, title, current_depth)
                
                if score > 0.3:  # Threshold for inclusion
                    scored_links.append({
                        'url': full_url,
                        'text': text,
                        'title': title,
                        'score': score,
                        'depth': current_depth + 1,
                        'domain': parsed.netloc
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing link {href}: {e}")
                continue
        
        # Sort by score and return top links
        scored_links.sort(key=lambda x: x['score'], reverse=True)
        logger.debug(f"Scored {len(scored_links)} intelligent links from {len(all_links)} raw links")
        return scored_links
    
    def _should_filter_link(self, url: str, text: str) -> bool:
        """Check if link should be filtered out"""
        for pattern in self.filter_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def _calculate_intelligence_score(self, text: str, url: str, title: str, depth: int) -> float:
        """Advanced scoring algorithm for link intelligence"""
        score = 0.0
        text_lower = text.lower()
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Primary keyword matches (high value)
        for keyword in self.primary_keywords:
            if keyword in text_lower:
                score += 3.0
            if keyword in title_lower:
                score += 2.0
            if keyword in url_lower:
                score += 1.5
        
        # Secondary keyword matches (medium value)
        for keyword in self.secondary_keywords:
            if keyword in text_lower:
                score += 1.5
            if keyword in title_lower:
                score += 1.0
        
        # Content quality indicators
        quality_words = ['research', 'study', 'analysis', 'report', 'article', 'detailed', 'comprehensive']
        if any(word in text_lower for word in quality_words):
            score += 2.0
        
        # Depth penalty (prefer not too deep)
        if depth >= 3:
            score *= 0.7
        elif depth >= 2:
            score *= 0.85
        
        # Length sweet spot
        word_count = len(text_lower.split())
        if 2 <= word_count <= 8:
            score += 1.0
        elif word_count > 15:
            score -= 0.5
        
        # Domain diversity bonus
        if 'wikipedia' not in url_lower:
            score += 0.5
        
        # Penalize obviously low-quality links
        junk_indicators = ['click here', 'read more', 'continue reading', 'next page']
        if any(junk in text_lower for junk in junk_indicators):
            score -= 1.0
        
        return max(0.0, score)

class PureCrawl4AIDeepSearch:
    """Pure Crawl4AI implementation using v0.7.x API with REAL deep crawling"""
    
    def __init__(self):
        self.url_cache = URLCache(maxsize=5000, ttl_seconds=7200)
        
    async def execute_deep_search(self, query: str, mode: str = "intensive") -> Dict[str, Any]:
        """Execute deep search using pure Crawl4AI approach"""
        start_time = time.time()
        
        # Configure based on mode
        if mode == "standard":
            deep_config = DeepSearchConfig(max_depth=2, max_pages=20, concurrent_crawlers=3)
        elif mode == "intensive":
            deep_config = DeepSearchConfig(max_depth=3, max_pages=40, concurrent_crawlers=5)
        elif mode == "ultra":
            deep_config = DeepSearchConfig(max_depth=4, max_pages=60, concurrent_crawlers=8)
        else:
            deep_config = DeepSearchConfig()
        
        logger.info(f"🚀 PURE CRAWL4AI DEEP SEARCH: '{query}' | Mode: {mode.upper()}")
        logger.info(f"📊 Config: {deep_config.max_pages} pages, {deep_config.max_depth} depth, {deep_config.concurrent_crawlers} crawlers")
        
        # Initialize components
        stats = DeepSearchStats()
        keywords = self._extract_keywords(query)
        extractor = IntelligentCrawl4AIExtractor(keywords)
        
        logger.info(f"🧠 Keywords for intelligence: {keywords}")
        
        # Get strategic starting URLs
        starting_urls = await self._get_strategic_starting_urls(query, keywords)
        
        if not starting_urls:
            return self._create_failure_response(query, "No starting URLs found", time.time() - start_time)
        
        logger.info(f"🎯 Strategic starting URLs: {len(starting_urls)}")
        
        # Execute pure Crawl4AI crawling
        crawl_results = await self._execute_pure_crawl4ai_crawl(
            starting_urls, deep_config, extractor, stats, query
        )
        
        # Generate results
        processing_time = time.time() - start_time
        summary = self._generate_intelligence_summary(query, crawl_results, stats, mode, processing_time)
        
        logger.info(f"🎉 PURE CRAWL4AI COMPLETE: {stats.pages_crawled} pages, {len(stats.domains_hit)} domains, {processing_time:.1f}s")
        
        return {
            "query": query,
            "mode": mode,
            "success": True,
            "processing_time": processing_time,
            "stats": {
                "pages_crawled": stats.pages_crawled,
                "domains_hit": len(stats.domains_hit),
                "total_words": stats.total_words_extracted,
                "successful_extractions": stats.successful_extractions,
                "average_page_time": stats.average_page_time,
                "keywords_found": dict(stats.keywords_found),
                "depth_distribution": dict(stats.depth_distribution),
                "crawl_speed": stats.pages_crawled / processing_time if processing_time > 0 else 0,
                "links_extracted": stats.links_extracted,
                "links_followed": stats.links_followed
            },
            "results": crawl_results,
            "summary": summary
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords for intelligent targeting"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'how', 'new', 'now', 
            'see', 'two', 'way', 'who', 'what', 'when', 'where', 'why', 'this', 'that'
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:8]
    
    async def _get_strategic_starting_urls(self, query: str, keywords: List[str]) -> List[str]:
        """Get starting URLs using multiple search strategies"""
        all_urls = []
        
        # Strategy 1: Direct search
        try:
            direct_results = await search_engine.search(query, SearchCategory.GENERAL, num_results=5)
            all_urls.extend([r.url for r in direct_results])
        except Exception as e:
            logger.warning(f"Direct search failed: {e}")
        
        # Strategy 2: Enhanced search with keywords
        if keywords:
            try:
                enhanced_query = f"{query} {' '.join(keywords[:3])}"
                enhanced_results = await search_engine.search(enhanced_query, SearchCategory.GENERAL, num_results=3)
                all_urls.extend([r.url for r in enhanced_results])
            except Exception as e:
                logger.warning(f"Enhanced search failed: {e}")
        
        # Strategy 3: Academic search if relevant
        if any(word in query.lower() for word in ['research', 'study', 'analysis', 'report', 'paper']):
            try:
                academic_results = await search_engine.search(query, SearchCategory.ACADEMIC, num_results=3)
                all_urls.extend([r.url for r in academic_results])
            except Exception as e:
                logger.warning(f"Academic search failed: {e}")
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))
        return unique_urls[:10]
    
    async def _execute_pure_crawl4ai_crawl(self, starting_urls: List[str], deep_config: DeepSearchConfig,
                                          extractor: IntelligentCrawl4AIExtractor, stats: DeepSearchStats,
                                          original_query: str) -> List[Dict[str, Any]]:
        """Execute crawling using pure Crawl4AI approach"""
        
        crawl_queue = [(url, 0) for url in starting_urls]
        crawled_results = []
        domain_counts = defaultdict(int)
        
        # Concurrency control
        semaphore = asyncio.Semaphore(deep_config.concurrent_crawlers)
        
        while crawl_queue and len(crawled_results) < deep_config.max_pages:
            # Process batch
            batch_size = min(deep_config.concurrent_crawlers, len(crawl_queue), 
                           deep_config.max_pages - len(crawled_results))
            
            current_batch = []
            for _ in range(batch_size):
                if crawl_queue:
                    current_batch.append(crawl_queue.pop(0))
            
            if not current_batch:
                break
            
            logger.info(f"🔄 Pure Crawl4AI batch: {len(current_batch)} URLs at depths {[d for _, d in current_batch]}")
            
            # Process batch with pure Crawl4AI
            batch_tasks = [
                self._crawl_single_url_pure_crawl4ai(url, depth, deep_config, extractor, 
                                                   semaphore, stats, original_query)
                for url, depth in current_batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and extract new links
            for (url, depth), result in zip(current_batch, batch_results):
                if isinstance(result, Exception):
                    stats.failed_extractions += 1
                    logger.warning(f"❌ Failed to crawl {url}: {result}")
                    continue
                
                if result:
                    crawled_results.append(result)
                    stats.pages_crawled += 1
                    domain = urlparse(url).netloc
                    stats.domains_hit.add(domain)
                    domain_counts[domain] += 1
                    stats.depth_distribution[depth] = stats.depth_distribution.get(depth, 0) + 1
                    
                    # Get intelligent links from result
                    if depth < deep_config.max_depth and 'intelligent_links' in result:
                        new_links = result['intelligent_links']
                        stats.links_extracted += len(new_links)
                        
                        # Add high-scoring links to queue (with domain limits)
                        added_count = 0
                        for link_info in new_links[:deep_config.max_links_per_page]:
                            link_url = link_info['url']
                            link_domain = urlparse(link_url).netloc
                            
                            # Check domain limits
                            if domain_counts[link_domain] >= deep_config.max_per_domain:
                                continue
                            
                            if not self._should_skip_url(link_url, depth + 1, deep_config):
                                crawl_queue.append((link_url, depth + 1))
                                added_count += 1
                                stats.links_followed += 1
                        
                        if added_count > 0:
                            logger.info(f"🔗 Pure Crawl4AI: Added {added_count} intelligent links from {domain}")
            
            # Delay between batches
            if deep_config.delay_seconds > 0:
                await asyncio.sleep(deep_config.delay_seconds)
        
        # Update final stats
        if stats.pages_crawled > 0:
            stats.average_page_time = stats.total_processing_time / stats.pages_crawled
        
        return crawled_results
    
    async def _crawl_single_url_pure_crawl4ai(self, url: str, depth: int, deep_config: DeepSearchConfig,
                                            extractor: IntelligentCrawl4AIExtractor, semaphore: asyncio.Semaphore,
                                            stats: DeepSearchStats, original_query: str) -> Optional[Dict[str, Any]]:
        """Crawl single URL using pure Crawl4AI with intelligent extraction"""
        
        async with semaphore:
            start_time = time.time()
            
            try:
                # Check cache
                if self.url_cache.is_url_seen(url):
                    return None
                
                # Configure browser for this crawl
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=False
                )
                
                # Configure crawl with content filtering - links are extracted by default
                crawl_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    css_selector=extractor._get_content_selectors(url),
                    word_count_threshold=100,
                    verbose=False,
                    page_timeout=30000,  # 30 seconds timeout
                    exclude_external_links=False,  # We want external links for deep crawling
                    remove_overlay_elements=True,
                    process_iframes=False,  # Skip iframes for speed
                )
                
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    logger.debug(f"🔍 Crawling {url}")
                    result = await crawler.arun(url, config=crawl_config)
                    
                    if not getattr(result, 'success', False):
                        stats.failed_extractions += 1
                        logger.warning(f"❌ Crawl failed for {url}: {getattr(result, 'error_message', 'Unknown error')}")
                        return None
                    
                    # Extract content
                    content = ""
                    if hasattr(result, 'markdown') and result.markdown:
                        content = str(result.markdown)
                    elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                        content = str(result.cleaned_html)
                    else:
                        content = getattr(result, 'text', '')
                    
                    if len(content.strip()) < 100:
                        stats.failed_extractions += 1
                        logger.warning(f"❌ Insufficient content for {url}: {len(content.strip())} chars")
                        return None
                    
                    # Extract and score links using built-in Crawl4AI functionality
                    intelligent_links = extractor.extract_and_score_links(result, url, depth)
                
                # Mark as crawled
                self.url_cache.mark_url_seen(url)
                
                # Update stats
                word_count = len(content.split())
                stats.total_words_extracted += word_count
                stats.successful_extractions += 1
                
                processing_time = time.time() - start_time
                stats.total_processing_time += processing_time
                
                # Count keyword occurrences
                content_lower = content.lower()
                for keyword in original_query.lower().split():
                    if len(keyword) > 3:
                        count = content_lower.count(keyword)
                        if count > 0:
                            stats.keywords_found[keyword] = stats.keywords_found.get(keyword, 0) + count
                
                logger.info(f"✅ [{depth}] Pure Crawl4AI: {urlparse(url).netloc} ({word_count} words, {len(intelligent_links)} links, {processing_time:.1f}s)")
                
                return {
                    'url': url,
                    'depth': depth,
                    'content': content,
                    'word_count': word_count,
                    'processing_time': processing_time,
                    'domain': urlparse(url).netloc,
                    'timestamp': datetime.now().isoformat(),
                    'intelligent_links': intelligent_links,
                    'links_count': len(intelligent_links)
                }
                
            except Exception as e:
                stats.failed_extractions += 1
                logger.warning(f"❌ Pure Crawl4AI crawl failed for {url}: {e}")
                return None
    
    def _should_skip_url(self, url: str, depth: int, deep_config: DeepSearchConfig) -> bool:
        """Intelligent URL filtering"""
        if self.url_cache.is_url_seen(url):
            return True
        
        if depth > deep_config.max_depth:
            return True
        
        return False
    
    def _generate_intelligence_summary(self, query: str, results: List[Dict], 
                                     stats: DeepSearchStats, mode: str, processing_time: float) -> str:
        """Generate impressive summary with pure Crawl4AI metrics"""
        
        if not results:
            return f"No results found for deep search: '{query}'"
        
        # Sort by relevance
        results.sort(key=lambda x: (x.get('word_count', 0), -x.get('depth', 0)), reverse=True)
        
        # Calculate metrics
        pages_per_second = stats.pages_crawled / processing_time if processing_time > 0 else 0
        words_per_page = stats.total_words_extracted / stats.pages_crawled if stats.pages_crawled > 0 else 0
        link_efficiency = stats.links_followed / stats.links_extracted if stats.links_extracted > 0 else 0
        
        summary = f"""# 🚀 PURE CRAWL4AI INTELLIGENCE REPORT: "{query}"

## 📊 PERFORMANCE METRICS  
**Mode:** {mode.upper()} Pure Crawl4AI Engine v0.7.x  
**Processing Time:** {processing_time:.1f}s  
**Crawl Speed:** {pages_per_second:.1f} pages/second  
**Pages Analyzed:** {stats.pages_crawled}  
**Domains Infiltrated:** {len(stats.domains_hit)}  
**Content Extracted:** {stats.total_words_extracted:,} words  
**Average Content/Page:** {words_per_page:.0f} words  
**Success Rate:** {(stats.successful_extractions/max(1, stats.pages_crawled))*100:.1f}%  

## 🔗 LINK INTELLIGENCE  
**Links Discovered:** {stats.links_extracted}  
**Links Followed:** {stats.links_followed}  
**Link Efficiency:** {link_efficiency*100:.1f}% (quality filter)  
**Architecture:** Built-in Crawl4AI Link Extraction  

## 🧠 CONTENT ANALYSIS  
**Keyword Distribution:**
{chr(10).join(f"  • {keyword}: {count} occurrences" for keyword, count in sorted(stats.keywords_found.items(), key=lambda x: x[1], reverse=True)[:5])}

**Depth Distribution:**
{chr(10).join(f"  • Depth {depth}: {count} pages" for depth, count in sorted(stats.depth_distribution.items()))}

**Domain Intelligence:**
{chr(10).join(f"  • {domain}" for domain in sorted(stats.domains_hit)[:8])}

## 🎯 TOP INTELLIGENCE SOURCES
"""
        
        # Add top results
        for i, result in enumerate(results[:5], 1):
            summary += f"""
### {i}. {urlparse(result['url']).netloc.upper()} | Depth {result['depth']}
**URL:** {result['url']}  
**Content Volume:** {result['word_count']:,} words  
**Links Extracted:** {result.get('links_count', 0)}  
**Processing Time:** {result['processing_time']:.1f}s  

**Content Sample:**
{result['content'][:400]}{"..." if len(result['content']) > 400 else ""}
"""
        
        if len(results) > 5:
            summary += f"\n... and {len(results) - 5} additional intelligence sources"
        
        summary += f"""

---
*🎭 Pure Crawl4AI Intelligence System v0.7.x*  
*⚡ Built-in link extraction: {processing_time:.1f}s | {len(stats.domains_hit)} domains | {stats.links_extracted} links discovered*  
*🧠 Using native Crawl4AI link extraction - Maximum efficiency*
"""
        
        return summary
    
    def _create_failure_response(self, query: str, error: str, processing_time: float) -> Dict[str, Any]:
        """Create failure response"""
        return {
            "query": query,
            "success": False,
            "error": error,
            "processing_time": processing_time,
            "summary": f"Pure Crawl4AI deep search failed: {error}"
        }

# Global pure Crawl4AI instance
deep_search_tool = PureCrawl4AIDeepSearch()
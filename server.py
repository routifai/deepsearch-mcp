#!/usr/bin/env python3
"""
DeepSearch MCP Server
Modern FastMCP 2.x implementation with proper error handling and Context support.
Following MCP best practices: simple tools, not autonomous agents.

CHANGES:
- Added MAX_CONTENT_PER_URL limit (20k chars per URL)
- Added MAX_TOTAL_RESPONSE_SIZE limit (800k chars total)
- Truncate individual content before adding to response
- Track total response size and stop when limit reached
- Better token estimation (1 token ≈ 4 chars)
"""

import json
import logging
import os
import sys
import asyncio
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from collections import OrderedDict
from dotenv import load_dotenv
from fastmcp import FastMCP, Context

# Load environment variables
load_dotenv()

from configurations.config import config, validate_startup_config
from tools.search_engine import search_engine, SearchCategory
from tools.web_fetcher import WebFetcher
from configurations.exceptions import handle_error, ConfigurationError

# Setup logging - force stderr for stdio mode to prevent JSON-RPC interference
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Always log to stderr to avoid interfering with stdio MCP protocol
)
logger = logging.getLogger(__name__)

# Validate configuration
if not validate_startup_config():
    logger.error("❌ Invalid configuration. Please check your .env file.")
    exit(1)

# Initialize FastMCP
mcp = FastMCP("deepsearch-mcp")
web_fetcher = WebFetcher()

# ============================================================================
# Memory-Efficient Content Cache with Size Limits
# ============================================================================
class MemoryEfficientCache:
    """
    LRU cache with strict size limits to prevent memory bloat.
    Max 10 items, 10 minutes TTL, auto-cleanup on access.
    """
    def __init__(self, max_items: int = 10, ttl_minutes: int = 10):
        self.cache: OrderedDict[str, Tuple[str, datetime]] = OrderedDict()
        self.max_items = max_items
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, url: str) -> str | None:
        # Clean expired entries on every access
        self._cleanup_expired()
        
        if url in self.cache:
            content, timestamp = self.cache[url]
            if datetime.now() - timestamp < self.ttl:
                # Move to end (most recently used)
                self.cache.move_to_end(url)
                return content
            else:
                # Expired - remove it
                del self.cache[url]
        return None
    
    def set(self, url: str, content: str):
        # Remove if exists (will re-add at end)
        if url in self.cache:
            del self.cache[url]
        
        # Add to end
        self.cache[url] = (content, datetime.now())
        
        # Enforce max size - remove oldest
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self):
        """Clear cache and force garbage collection"""
        self.cache.clear()
        gc.collect()

content_cache = MemoryEfficientCache(max_items=10, ttl_minutes=10)

# ============================================================================
# Response Size Limits (Reduced for 2Gi memory limit)
# ============================================================================
MAX_CONTENT_PER_URL = 20000  # Reduced from 25k → 20k chars (≈5k tokens)
MAX_TOTAL_RESPONSE_SIZE = 600000  # Reduced from 800k → 600k chars (≈150k tokens)
TARGET_URLS_TO_FETCH = 3  # Reduced from 5 → 3 URLs


def truncate_content(content: str, max_chars: int, url: str) -> tuple[str, bool]:
    """
    Truncate content to max_chars with smart truncation.
    Returns (truncated_content, was_truncated)
    """
    if len(content) <= max_chars:
        return content, False
    
    # Smart truncation: try to end at paragraph or sentence
    truncated = content[:max_chars]
    
    # Try to find last paragraph break (double newline)
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.75:  # At least 75% of content
        truncated = truncated[:last_para]
    else:
        # Try to find last sentence
        last_period = truncated.rfind('. ')
        if last_period > max_chars * 0.75:
            truncated = truncated[:last_period + 1]
    
    truncated += f"\n\n[Content truncated - Full content at: {url}]"
    return truncated, True


# ============================================================================
# MCP Tools with Context Support
# ============================================================================

@mcp.tool()
async def get_current_date(ctx: Context = None) -> str:
    """Get the current date and year.
    
    IMPORTANT: LLMs do not have access to real-time date information.
    Call this tool to get the current date when users ask about:
    - "latest" or "recent" products/news/events
    - "current" information
    - "today", "this year", "this month"
    - Any time-sensitive queries
    
    Returns:
        JSON with current date, year, month, and search guidance
    """
    from datetime import datetime
    
    now = datetime.now()
    
    result = {
        'current_date': now.strftime('%Y-%m-%d'),
        'current_date_long': now.strftime('%B %d, %Y'),
        'current_year': now.year,
        'current_month': now.strftime('%B'),
        'current_day': now.day,
        'day_of_week': now.strftime('%A'),
        'timestamp': now.isoformat(),
        'guidance': f'When searching for latest/recent information, include year {now.year} in your search query'
    }
    
    if ctx:
        await ctx.info(f"Current date: {now.strftime('%Y-%m-%d %H:%M:%S')} (Year: {now.year})")
    
    return json.dumps(result, indent=2)


@mcp.tool()
async def web_search(query: str, num_results: int = 5, auto_fetch: bool = True, ctx: Context = None) -> str:
    """Search the web and automatically fetch content from results.
    
    oCRITICAL: DO NOT ASSUME THE CURRENT DATE OR YEAR
    If the user asks about "latest", "recent", "current", or time-sensitive information:
    1. FIRST call get_current_date() to get the actual current year
    2. THEN include that year in your search query
    
    Example workflow:
    - User: "what is the latest iPhone?"
    - You: Call get_current_date() → returns 2025
    - You: Call web_search("latest iPhone 2025 specifications")
    
    DO NOT search "latest iPhone 2023" or assume any year - always get current date first!
    
    AUTO-FETCHING:
    When auto_fetch=True (default), this tool will:
    1. Search for results
    2. Automatically fetch content from at least 5 URLs
    3. Apply smart content truncation (25k chars per URL, 800k total)
    4. Return both search results and scraped content
    
    When auto_fetch=False, only search results with snippets are returned.
    
    USAGE EXAMPLES:
    
    SNIPPET-ONLY QUERIES (set auto_fetch=False):
    - "What are the top 10 companies in AI?"
    - "List of best restaurants in Paris"
    - "Recent news headlines about climate change"
    - "Compare prices of iPhone vs Samsung"
    - "What are the trending topics today?"
    
    FULL-CONTENT QUERIES (auto_fetch=True recommended):
    - "How does machine learning work?"
    - "Explain the process of photosynthesis"
    - "What are the latest iPhone specifications?" (Remember: call get_current_date() first!)
    - "How to implement authentication in React?"
    - "What are the health benefits of exercise?"
    
    Args:
        query: Search query string. Include year for temporal queries (get from get_current_date()).
        num_results: Number of search results to return (3-10, default: 5, minimum enforced: 3)
        auto_fetch: Whether to automatically fetch content (default: True)
        ctx: Context for logging and user feedback
        
    Returns:
        JSON with search results, scraped content, and metadata
    """
    
    # Simple guidance based on query patterns
    query_lower = query.lower()
    snippet_indicators = ['list', 'top', 'best', 'compare', 'headlines', 'news', 'trending', 'prices', 'cost', 'reviews', 'ratings']
    is_likely_snippet_query = any(indicator in query_lower for indicator in snippet_indicators)
    
    # Enforce minimum 3 results for better quality
    original_num_results = num_results
    num_results = min(max(3, num_results), 10)
    
    if ctx:
        if original_num_results < 3:
            await ctx.info(f"Requested {original_num_results} results, but minimum is 3. Using {num_results} results.")
        
        await ctx.info(f"Searching: '{query}' ({num_results} results, auto_fetch={auto_fetch})")
        if is_likely_snippet_query and auto_fetch:
            await ctx.info(f"This query might work well with auto_fetch=False for faster results")
    
    try:
        # Step 1: Perform search
        results = await search_engine.search(
            query=query,
            category=SearchCategory.GENERAL,
            num_results=num_results
        )
        
        if not results:
            if ctx:
                await ctx.warning("No results found")
            return json.dumps({
                'query': query,
                'success': False,
                'results': [],
                'scraped_content': [],
                'message': f"No results found for query: {query}"
            }, indent=2)
        
        # Format search results
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append({
                'rank': i,
                'title': r.title,
                'url': r.url,
                'snippet': r.snippet
            })
        
        response = {
            'query': query,
            'success': True,
            'search_date': datetime.now().strftime("%Y-%m-%d"),
            'results_count': len(results),
            'results': formatted_results,
            'scraped_content': [],
            'auto_fetch_enabled': auto_fetch
        }
        
        # Step 2: Auto-fetch content if enabled
        if auto_fetch:
            if ctx:
                await ctx.info(f"Auto-fetching content from URLs with size limits...")
            
            scraped_content = []
            successful_scrapes = 0
            failed_urls = []
            total_response_chars = 0
            urls_truncated = 0
            
            # Parallel fetching for 3-5x speed improvement
            fetch_tasks = [web_fetcher.fetch_url(r.url, mode="partial") for r in results[:TARGET_URLS_TO_FETCH]]
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Process results
            for i, (result, content) in enumerate(zip(results[:TARGET_URLS_TO_FETCH], fetch_results)):
                # Check if we've hit our limits
                if successful_scrapes >= TARGET_URLS_TO_FETCH and total_response_chars >= MAX_TOTAL_RESPONSE_SIZE * 0.8:
                    if ctx:
                        await ctx.info(f"Reached target: {successful_scrapes} URLs, {total_response_chars:,} chars")
                    break
                
                if total_response_chars >= MAX_TOTAL_RESPONSE_SIZE:
                    if ctx:
                        await ctx.warning(f"Response size limit reached ({MAX_TOTAL_RESPONSE_SIZE:,} chars)")
                    break
                    
                if ctx:
                    await ctx.info(f"Processing {i+1}/{len(fetch_results)}: {result.url}")
                
                try:
                    # Handle exceptions from parallel fetching
                    if isinstance(content, Exception):
                        failed_urls.append({
                            'url': result.url,
                            'title': result.title,
                            'error': str(content)
                        })
                        if ctx:
                            await ctx.warning(f"Fetching failed: {str(content)}")
                        continue
                    
                    # Check if scraping was successful
                    if len(content) > 100 and not content.startswith("Error fetching"):
                        # Truncate individual content to max per URL
                        truncated_content, was_truncated = truncate_content(
                            content, 
                            MAX_CONTENT_PER_URL, 
                            result.url
                        )
                        
                        if was_truncated:
                            urls_truncated += 1
                        
                        content_length = len(truncated_content)
                        estimated_tokens = content_length // 4
                        
                        # Check if adding this would exceed total limit
                        if total_response_chars + content_length > MAX_TOTAL_RESPONSE_SIZE:
                            # Calculate how much we can include
                            remaining_chars = MAX_TOTAL_RESPONSE_SIZE - total_response_chars
                            if remaining_chars > 5000:  # Only include if we can get meaningful content
                                truncated_content, _ = truncate_content(
                                    truncated_content,
                                    remaining_chars,
                                    result.url
                                )
                                content_length = len(truncated_content)
                                urls_truncated += 1
                            else:
                                if ctx:
                                    await ctx.warning(f"⚠️ Skipping - would exceed total size limit")
                                break
                        
                        scraped_content.append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet,
                            'content': truncated_content,
                            'content_length': content_length,
                            'estimated_tokens': estimated_tokens,
                            'rank': result.rank,
                            'was_truncated': was_truncated,
                            'success': True
                        })
                        
                        successful_scrapes += 1
                        total_response_chars += content_length
                        
                        if ctx:
                            truncate_info = " (truncated)" if was_truncated else ""
                            await ctx.info(f"Fetched {content_length:,} chars (~{estimated_tokens:,} tokens){truncate_info}")
                    else:
                        failed_urls.append({
                            'url': result.url,
                            'title': result.title,
                            'error': 'Insufficient content or error'
                        })
                        if ctx:
                            await ctx.warning(f"Failed to fetch sufficient content")
                            
                except Exception as e:
                    failed_urls.append({
                        'url': result.url,
                        'title': result.title,
                        'error': str(e)
                    })
                    if ctx:
                        await ctx.warning(f"Processing failed: {str(e)}")
            
            # Update response with scraped content
            response['scraped_content'] = scraped_content
            response['failed_urls'] = failed_urls
            response['scraping_stats'] = {
                'successful_fetches': successful_scrapes,
                'failed_fetches': len(failed_urls),
                'total_chars': total_response_chars,
                'estimated_tokens': total_response_chars // 4,
                'urls_truncated': urls_truncated,
                'max_chars_per_url': MAX_CONTENT_PER_URL,
                'max_total_chars': MAX_TOTAL_RESPONSE_SIZE
            }
            
            if successful_scrapes >= TARGET_URLS_TO_FETCH:
                if ctx:
                    await ctx.info(f"Successfully fetched {successful_scrapes} URLs (~{total_response_chars // 4:,} tokens)")
            else:
                if ctx:
                    await ctx.warning(f"Only fetched {successful_scrapes} URLs (target: {TARGET_URLS_TO_FETCH})")
            
            # Force garbage collection after fetching to release memory
            gc.collect()
        else:
            response['next_steps'] = "Call fetch_url() on relevant URLs to get full content"
        
        if ctx:
            await ctx.info(f"Search completed: {len(results)} results")
        
        logger.info(f"Search completed: {len(results)} results")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = handle_error(e, "web_search")
        logger.error(f"Search failed: {error_msg}")
        
        if ctx:
            await ctx.error(f"Search failed: {error_msg}")
        
        return json.dumps({
            'query': query,
            'success': False,
            'error': error_msg
        }, indent=2)


@mcp.tool()
async def fetch_url(url: str, ctx: Context = None) -> str:
    """Fetch and extract content from a specific URL.
    
    PURPOSE:
    Retrieves full content from a webpage, extracting clean readable text
    from HTML. Use this after web_search() to get detailed information.
    
    TEMPORAL NOTE: If analyzing time-sensitive content, consider calling 
    get_current_date() first to understand the temporal context.
    
    CONTENT LIMITS:
    Content is automatically truncated to 25k characters (≈6k tokens) to ensure
    responses stay under size limits. For full content, visit the URL directly.
    
    TOOL CHAINING PATTERN:
    Standard workflow:
    1. (Optional) get_current_date() if dealing with time-sensitive queries
    2. web_search(query) returns URLs
    3. fetch_url(url1) gets first source content
    4. fetch_url(url2) gets second source for verification
    5. fetch_url(url3) gets additional perspective if needed
    6. Synthesize comprehensive answer from all sources
    
    Args:
        url: Complete URL to fetch (must be valid HTTP/HTTPS URL)
        ctx: Context for logging and user feedback
        
    Returns:
        JSON with content, metadata, and status
    """
    
    if ctx:
        await ctx.info(f"Fetching: {url}")
    
    try:
        # Check cache first
        cached = content_cache.get(url)
        if cached:
            if ctx:
                await ctx.info(f"Using cached content")
            return json.dumps({'url': url, 'success': True, 'cached': True, 'content': cached}, indent=2)
        
        content = await web_fetcher.fetch_url(url, mode="partial")
        
        # Truncate content to prevent oversized responses
        truncated_content, was_truncated = truncate_content(content, MAX_CONTENT_PER_URL, url)
        
        # Cache the content
        content_cache.set(url, truncated_content)
        
        response = {
            'url': url,
            'success': True,
            'fetched_date': datetime.now().strftime("%Y-%m-%d"),
            'content_length': len(truncated_content),
            'original_length': len(content),
            'was_truncated': was_truncated,
            'content': truncated_content
        }
        
        if ctx:
            truncate_info = f" (truncated from {len(content):,})" if was_truncated else ""
            await ctx.info(f"Fetched {len(truncated_content):,} characters{truncate_info}")
        
        logger.info(f"Fetch completed: {len(truncated_content)} characters")
        
        # Force garbage collection after fetching
        gc.collect()
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = handle_error(e, "fetch_url")
        logger.error(f"Fetch failed: {error_msg}")
        
        if ctx:
            await ctx.error(f"Fetch failed: {error_msg}")
        
        return json.dumps({
            'url': url,
            'success': False,
            'error': error_msg,
            'suggestion': "Try another URL from search results"
        }, indent=2)


@mcp.tool()
async def health(ctx: Context = None) -> str:
    """Check server health and status"""
    
    if ctx:
        await ctx.info("Checking server health...")
    
    try:
        search_status = search_engine.get_status()
        provider_info = config.get_status_info()
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0-modern',
            'components': {
                'search_engine': search_status,
                'web_fetcher': 'operational'
            },
            'config': provider_info,
            'content_limits': {
                'max_chars_per_url': MAX_CONTENT_PER_URL,
                'max_total_response_chars': MAX_TOTAL_RESPONSE_SIZE,
                'estimated_max_tokens': MAX_TOTAL_RESPONSE_SIZE // 4
            },
            'features': [
                'Multi-provider search (SerpAPI, Google CSE, Tavily)',
                'Clean content extraction (Crawl4AI)',
                'Smart content truncation',
                'Context-aware logging',
                'Structured JSON responses',
                'Parallel URL fetching (3-5x faster)',
                'Simple content cache (eliminates duplicate fetches)',
                'Fixed browser memory leaks for long sessions'
            ]
        }
        
        if ctx:
            await ctx.info("Server is healthy")
        
        return json.dumps(health_data, indent=2)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        if ctx:
            await ctx.error(f"Health check failed: {e}")
        
        return json.dumps({
            'status': 'degraded',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, indent=2)


@mcp.tool()
async def get_server_info(ctx: Context = None) -> str:
    """Get server information and capabilities"""
    
    if ctx:
        await ctx.info("Getting server information...")
    
    provider_info = config.get_status_info()
    primary_provider = provider_info["search_providers"]["primary"]
    available_providers = provider_info["search_providers"]["available"]
    
    server_info = {
        'name': 'DeepSearch MCP Server',
        'version': '2.0-modern',
        'description': 'Modern FastMCP 2.x server with Context support and smart content limits',
        'philosophy': 'Simple tools, not autonomous agents',
        'tools': [
            {
                'name': 'get_current_date',
                'description': 'Get current date/year - MUST call this before searching for latest/recent information'
            },
            {
                'name': 'web_search',
                'description': 'Search the web and automatically fetch content from at least 5 URLs with smart truncation'
            },
            {
                'name': 'fetch_url',
                'description': 'Fetch and extract content from URLs (auto-truncated to 25k chars)'
            },
            {
                'name': 'health',
                'description': 'Check server health and status'
            },
            {
                'name': 'get_server_info',
                'description': 'Get server information and capabilities'
            }
        ],
        'important_usage_note': 'DO NOT ASSUME current date/year. Always call get_current_date() first when dealing with time-sensitive queries.',
        'search_providers': {
            'primary': primary_provider,
            'available': available_providers
        },
        'content_limits': {
            'max_chars_per_url': MAX_CONTENT_PER_URL,
            'max_total_response': MAX_TOTAL_RESPONSE_SIZE,
            'target_urls_to_fetch': TARGET_URLS_TO_FETCH,
            'estimated_max_tokens': MAX_TOTAL_RESPONSE_SIZE // 4
        },
        'features': [
            'Temporal awareness via get_current_date() tool',
            'Context-aware logging and user feedback',
            'Structured JSON responses',
            'Multi-provider search support',
            'Clean content extraction with Crawl4AI',
            'Smart content truncation to prevent oversized responses',
            'Error handling with helpful suggestions',
            'Auto-fetch with size limits',
            'Guaranteed minimum 5 URL scraping (when available)',
            'Parallel URL fetching (3-5x faster)',
            'Simple content cache (eliminates duplicate fetches)',
            'Fixed browser memory leaks for long sessions'
        ]
    }
    
    if ctx:
        await ctx.info("Server info retrieved")
    
    return json.dumps(server_info, indent=2)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if running in stdio mode (for Claude Desktop) or HTTP mode
    is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
    
    # Only log startup info in HTTP mode to avoid MCP JSON parsing errors
    if not is_stdio_mode:
        logger.info("=" * 80)
        logger.info("DeepSearch MCP Server v2.0 (Modern FastMCP)")
        logger.info("Features: Context support, structured responses, smart content limits")
        
        # Get provider info
        provider_info = config.get_status_info()
        primary_provider = provider_info["search_providers"]["primary"]
        available_providers = provider_info["search_providers"]["available"]
        
        logger.info(f"Search: {', '.join(available_providers)} (primary: {primary_provider})")
        logger.info("Tools: get_current_date, web_search (auto-fetch), fetch_url, health, get_server_info")
        logger.info(f"Limits: {MAX_CONTENT_PER_URL:,} chars/URL, {MAX_TOTAL_RESPONSE_SIZE:,} chars total")
        logger.info("Philosophy: Simple tools, not autonomous agents")
        logger.info("Temporal Awareness: LLM must call get_current_date() for time-sensitive queries")
        logger.info("=" * 80)
    
    try:
        if is_stdio_mode:
            # Stdio mode for Claude Desktop - no startup logging to avoid JSON parsing errors
            mcp.run(transport="stdio")
        else:
            # HTTP mode for standalone use
            logger.info(f"Running in HTTP mode: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
            mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp")
    except KeyboardInterrupt:
        if not is_stdio_mode:
            logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
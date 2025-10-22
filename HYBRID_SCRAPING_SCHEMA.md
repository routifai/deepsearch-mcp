# Hybrid Scraping System Schema

## Overview
The DeepSearch MCP system implements a sophisticated hybrid scraping architecture that combines multiple strategies for optimal web content extraction. The system intelligently chooses between HTTP requests, proxy connections, and browser automation based on content requirements and availability.

## Architecture Components

### 1. Core Scraping Engine (`OptimizedHybridScraper`)

#### **Strategy Selection Flow**
```
URL Input → Domain Analysis → Strategy Selection → Content Extraction
```

#### **Multi-Tier Approach**
1. **HTTP Direct** (Fastest - ~2-5 seconds)
2. **HTTP with Proxy** (Fallback for blocked content)
3. **Browser Automation** (JavaScript-heavy sites)

### 2. Content Detection System (`DynamicContentDetector`)

#### **Detection Criteria**
- **Minimal Content**: < 200 characters of text
- **JS Framework Detection**: React, Vue, Angular patterns
- **Script Ratio**: > 30% JavaScript content
- **SPA Patterns**: Single-page application indicators

#### **Detection Logic**
```python
def needs_browser(html: str, url: str) -> tuple[bool, str]:
    # Check 1: Content length
    if text_length < 200:
        return True, "minimal_content"
    
    # Check 2: JS frameworks
    if js_framework_detected:
        return True, "js_framework"
    
    # Check 3: Script ratio
    if script_ratio > 0.3:
        return True, "high_js_ratio"
    
    # Check 4: SPA patterns
    if spa_indicators:
        return True, "spa_pattern"
    
    return False, "static_content"
```

### 3. Proxy Management System

#### **Proxy Configuration**
- **Environment Variables**: `PROXY_URL`, `PROXY_USER`, `PROXY_PASS`
- **Authentication**: Automatic credential injection
- **Domain Bypass**: Special handling for `abc.com` domains

#### **Proxy Strategy**
```
1. Try with proxy first (if configured)
2. Fall back to direct connection if proxy fails
3. Browser fallback with/without proxy as needed
```

### 4. Rate Limiting & Concurrency

#### **Rate Limiter**
- **Per-domain limiting**: 2 requests/second default
- **Concurrent requests**: 20 max concurrent
- **Timeout handling**: 10-second default

#### **Concurrency Control**
```python
async def scrape_urls(urls: List[str]):
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks)
```

### 5. Content Processing Pipeline

#### **Text Extraction (`TextExtractor`)**
1. **HTML Parsing**: BeautifulSoup with lxml
2. **Tag Removal**: Scripts, styles, navigation, footer
3. **Text Cleaning**: Whitespace normalization
4. **Content Validation**: Length and quality checks

#### **Metadata Extraction**
- **Title**: Page title extraction
- **Description**: Meta description
- **Content Metrics**: Length, tokens, response time
- **Method Tracking**: HTTP/Proxy/Browser used

### 6. MCP Server Integration

#### **Tool Architecture**
```
MCP Server (FastMCP 2.x)
├── web_search() - Search + auto-fetch
├── fetch_url() - Single URL extraction
├── get_current_date() - Temporal awareness
├── health() - System status
└── get_server_info() - Capabilities
```

#### **Content Limits**
- **Per URL**: 20,000 characters (≈5,000 tokens)
- **Total Response**: 600,000 characters (≈150,000 tokens)
- **Target URLs**: 3 URLs per search
- **Smart Truncation**: Paragraph/sentence boundaries

### 7. Search Engine Integration

#### **Multi-Provider Support**
- **SerpAPI**: Primary Google search
- **Google CSE**: Custom search engine
- **Tavily**: AI-powered search
- **Automatic Fallback**: Provider switching on failure

#### **Search Categories**
- **General**: Standard web search
- **News**: Recent news articles
- **Academic**: Scholarly content
- **Technical**: Stack Overflow, GitHub
- **Shopping**: Product searches
- **Images**: Visual content

## Data Flow Schema

### **Complete Request Flow**
```
1. User Query → MCP Tool
2. Search Engine → Multiple Providers
3. URL Discovery → Result Ranking
4. Content Fetching → Hybrid Scraping
5. Content Processing → Text Extraction
6. Size Management → Smart Truncation
7. Response Assembly → JSON Output
```

### **Scraping Decision Tree**
```
URL Input
├── Domain Check (abc.com bypass)
├── Proxy Available?
│   ├── Yes → Try Proxy HTTP
│   │   ├── Success → Return Content
│   │   └── Fail → Try Direct HTTP
│   └── No → Try Direct HTTP
├── HTTP Success?
│   ├── Yes → Check Content Type
│   │   ├── Static → Return Content
│   │   └── Dynamic → Browser Fallback
│   └── No → Browser Fallback
└── Browser Rendering
    ├── With Proxy (if available)
    └── Without Proxy (fallback)
```

## Performance Characteristics

### **Speed Optimization**
- **HTTP First**: 2-5 seconds per URL
- **Browser Fallback**: 10-30 seconds per URL
- **Parallel Processing**: 3-5x faster for multiple URLs
- **Content Caching**: Eliminates duplicate fetches

### **Memory Management**
- **LRU Cache**: 10 items, 10-minute TTL
- **Garbage Collection**: Forced after operations
- **Size Limits**: Prevents memory bloat
- **Session Management**: Proper cleanup

### **Error Handling**
- **SSL Certificate Issues**: Automatic retry with relaxed SSL
- **Timeout Management**: Configurable timeouts
- **Provider Fallback**: Multiple search providers
- **Graceful Degradation**: Partial success handling

## Configuration Schema

### **Environment Variables**
```bash
# Search Providers (need at least one)
SERPAPI_KEY=your_serpapi_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CSE_ID=your_cse_id
TAVILY_API_KEY=your_tavily_key

# Proxy Configuration (optional)
PROXY_URL=http://proxy.example.com:8080
PROXY_USER=username
PROXY_PASS=password

# Server Settings
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO
TIMEOUT_SECONDS=30
```

### **Content Limits**
```python
MAX_CONTENT_PER_URL = 20000      # 20k chars per URL
MAX_TOTAL_RESPONSE_SIZE = 600000 # 600k chars total
TARGET_URLS_TO_FETCH = 3         # 3 URLs per search
```

## Response Schema

### **Search Response**
```json
{
  "query": "search terms",
  "success": true,
  "search_date": "2025-01-27",
  "results_count": 5,
  "results": [
    {
      "rank": 1,
      "title": "Page Title",
      "url": "https://example.com",
      "snippet": "Content snippet..."
    }
  ],
  "scraped_content": [
    {
      "url": "https://example.com",
      "title": "Page Title",
      "content": "Full extracted content...",
      "content_length": 15000,
      "estimated_tokens": 3750,
      "was_truncated": false,
      "success": true
    }
  ],
  "scraping_stats": {
    "successful_fetches": 3,
    "failed_fetches": 0,
    "total_chars": 45000,
    "estimated_tokens": 11250
  }
}
```

### **Single URL Response**
```json
{
  "url": "https://example.com",
  "success": true,
  "fetched_date": "2025-01-27",
  "content_length": 15000,
  "original_length": 25000,
  "was_truncated": true,
  "content": "Extracted content..."
}
```

## Key Features

### **Intelligence**
- **Content Detection**: Automatic JS framework detection
- **Strategy Selection**: Optimal method per URL
- **Proxy Management**: Smart proxy usage
- **Size Management**: Intelligent truncation

### **Reliability**
- **Multi-Provider**: Search engine redundancy
- **Fallback Chains**: HTTP → Proxy → Browser
- **Error Recovery**: SSL retry, provider switching
- **Timeout Handling**: Graceful failure management

### **Performance**
- **Parallel Processing**: Concurrent URL fetching
- **Content Caching**: Duplicate request elimination
- **Memory Efficiency**: Size limits and cleanup
- **Rate Limiting**: Respectful scraping

### **Flexibility**
- **Multiple Providers**: SerpAPI, Google CSE, Tavily
- **Proxy Support**: Corporate network compatibility
- **Category Support**: News, academic, technical, shopping
- **Configurable Limits**: Adjustable content sizes

This hybrid scraping system provides a robust, intelligent, and efficient solution for web content extraction with automatic strategy selection, comprehensive error handling, and optimal performance characteristics.

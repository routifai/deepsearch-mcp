# DeepSearch MCP Server

A production-ready Model Context Protocol (MCP) server that provides intelligent web search capabilities with AI-powered query analysis, automatic content extraction, and comprehensive deep research functionality.

## 🚀 Features

- **AI-Powered Query Analysis**: Automatically determines optimal search strategies
- **Multi-Provider Search**: Support for SerpAPI and Google Custom Search
- **Intelligent Content Extraction**: Advanced web scraping with site-specific optimizations
- **Deep Research Capabilities**: Comprehensive crawling with AI-powered link discovery
- **Memory-Safe Caching**: TTL-based caching with size limits to prevent memory leaks
- **Resource Pooling**: Browser instance pooling for efficient resource management
- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Temporal Context Awareness**: Automatic handling of current vs historical information

## 🏗️ Architecture

```
server.py (MCP Entry Point)
    ↓
search_orchestrator.py (Main Logic)
    ↓
├── query_analyzer.py (AI Analysis)
├── search_engine.py (Search Providers)
├── deep_search.py (Deep Research)
└── web_fetcher.py (Content Extraction)
    ↓
├── browser_pool.py (Resource Management)
├── cache.py (Memory-Safe Caching)
└── config.py (Configuration)
```

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd deepsearch-mcp
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Required API Keys**
- **OpenAI API Key** (required for query analysis and deep search)
- **SerpAPI Key** OR **Google Custom Search API + CSE ID** (for search)

## ⚙️ Configuration

All configuration is managed through environment variables. See `.env.example` for all available options.

### Key Settings:

- `MAX_CONCURRENT=3` - Maximum concurrent operations
- `MAX_CACHE_SIZE=1000` - Cache size limit (prevents memory leaks)
- `TIMEOUT_SECONDS=30` - Request timeout
- `BROWSER_POOL_SIZE=3` - Browser instance pool size
- `LOG_LEVEL=INFO` - Logging level (DEBUG, INFO, WARNING, ERROR)

## 🚀 Usage

### Start the Server
```bash
python server.py
```

### MCP Tools

#### `web_search`
Intelligent web search with automatic content fetching:
```python
result = await web_search(
    query="current president of France",
    category="general",  # auto, news, academic, technical, etc.
    num_results=5
)
```

#### `fetch_url`
Direct URL content extraction:
```python
content = await fetch_url(
    url="https://example.com/article",
    mode="partial"  # snippet, partial, complete
)
```

#### `deep_search`
Comprehensive research with AI-powered link discovery:
```python
report = await deep_search(
    query="climate change impact on agriculture",
    mode="intensive",  # standard, intensive, ultra
    max_pages=20
)
```

### HTTP Endpoints

- **MCP Endpoint**: `http://127.0.0.1:8000/mcp`
- **Health Check**: `http://127.0.0.1:8000/health`
- **Server Info**: `http://127.0.0.1:8000/`

## 🔍 Search Categories

- `general`: Standard web search
- `news`: News articles and current events
- `academic`: Research papers and scholarly content
- `technical`: Documentation and programming resources
- `shopping`: Product information and prices
- `images`: Image search
- `local`: Location-based results

## 🔬 Deep Search Modes

- `standard`: Basic deep research with moderate crawling
- `intensive`: Comprehensive analysis with extensive link discovery
- `ultra`: Maximum depth research with AI-powered content prioritization

## 📊 Monitoring

### Health Check Response:
```json
{
  "status": "healthy",
  "components": {
    "search_engine": { "provider": "serpapi" },
    "browser_pool": { "available_browsers": 3 },
    "query_analyzer": { "hit_rate": 0.85 }
  }
}
```

### Statistics Available:
- Cache hit rates
- Browser pool utilization
- Search success rates
- Processing times
- Memory usage
- Deep search performance metrics

## 🛠️ Development

### Run Tests:
```bash
pytest tests/
```

### Code Formatting:
```bash
black .
```

### Type Checking:
```bash
mypy .
```

## 🔒 Security

- API keys stored in environment variables only
- No hardcoded credentials
- Request timeouts prevent hang attacks
- Resource limits prevent DoS attacks
- Input validation on all endpoints

## 🐛 Troubleshooting

### Common Issues:

1. **"No search provider configured"**
   - Check that either `SERPAPI_KEY` or both `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set

2. **"Browser pool exhausted"**
   - Increase `BROWSER_POOL_SIZE` or `MAX_CONCURRENT` settings

3. **"Query analysis failed"**
   - Verify `OPENAI_API_KEY` is valid and has sufficient credits

4. **High memory usage**
   - Reduce `MAX_CACHE_SIZE` or `CACHE_TTL` settings

5. **Deep search timeout**
   - Increase `TIMEOUT_SECONDS` or reduce `max_pages` parameter

### Debug Mode:
```bash
LOG_LEVEL=DEBUG python server.py
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue with:
- Error messages
- Configuration (without API keys)
- Steps to reproduce
- Expected vs actual behavior
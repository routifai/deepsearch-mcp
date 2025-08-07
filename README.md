# DeepSearch MCP

Advanced web crawling and content extraction using MCP (Model Context Protocol).

## Features

- **Intelligent Deep Crawling**: Uses Crawl4AI with built-in link extraction
- **Multi-Strategy Search**: Combines direct, enhanced, and academic search
- **Concurrent Processing**: Efficient batch processing with configurable concurrency
- **Smart Link Scoring**: Advanced relevance scoring for intelligent crawling
- **Comprehensive Statistics**: Detailed metrics and performance tracking

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://your-custom-endpoint.com/v1  # Optional: for custom endpoints
OPENAI_MODEL=gpt-4o-mini
LLM_CLIENT_TYPE=default  # or "custom" for custom endpoints

# Search API Configuration
SERPAPI_KEY=your-serpapi-key
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-google-cse-id

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# Performance Settings
MAX_CONCURRENT=3
TIMEOUT_SECONDS=30
```

### Centralized LLM Configuration

The system uses centralized LLM configuration that's shared across all components:

- **Default OpenAI**: Uses standard OpenAI API
- **Custom Endpoints**: Supports custom OpenAI-compatible endpoints
- **Automatic Detection**: Automatically detects and uses the appropriate configuration

#### Using Custom Endpoints

To use a custom OpenAI-compatible endpoint:

1. Set environment variables:
```bash
export OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
export LLM_CLIENT_TYPE=custom
export OPENAI_API_KEY=your-api-key
```

2. All components will automatically use the custom endpoint:
   - Client chat interface
   - Query analysis
   - Deep search operations

## Usage

### Start the Server

```bash
python server.py
```

### Use the Client

```bash
python client.py
```

### Deep Search Examples

```python
from tools.deep_search import deep_search_tool

# Standard search
result = await deep_search_tool.execute_deep_search("Python tutorials")

# Intensive search
result = await deep_search_tool.execute_deep_search("AI research papers", mode="intensive")

# Ultra search
result = await deep_search_tool.execute_deep_search("Machine learning trends", mode="ultra")
```

## Architecture

- **`configurations/`**: Centralized configuration management
- **`tools/`**: Core crawling and search tools
- **`agents/`**: LLM integration and query analysis
- **`client.py`**: Interactive MCP client
- **`server.py`**: FastMCP server implementation

## Performance

- **Concurrent Crawling**: Configurable concurrency (3-8 crawlers)
- **Intelligent Caching**: URL and analysis caching for efficiency
- **Smart Filtering**: Automatic filtering of low-quality links
- **Depth Control**: Configurable crawl depth (2-4 levels)

## License

MIT License
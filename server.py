#!/usr/bin/env python3
"""
Fixed MCP Server Implementation
Production-ready server with correct FastMCP usage and proper error handling.
"""

import asyncio
import logging
import signal
import sys
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from configurations.config import config, validate_startup_config
from tools.search_orchestrator import SearchOrchestrator
from tools.web_fetcher import WebFetcher
from tools.deep_search import deep_search_tool

from configurations.browser_pool import browser_pool, initialize_browser_pool, cleanup_browser_pool
from configurations.exceptions import handle_error, ConfigurationError

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPSearchServer:
    """Main MCP server class with lifecycle management"""
    
    def __init__(self):
        # Validate configuration with helpful messages
        if not validate_startup_config():
            raise ConfigurationError(
                "Invalid configuration. Please check the setup guide above."
            )
        
        self.orchestrator = SearchOrchestrator()
        self.web_fetcher = WebFetcher()
        self.app = FastMCP("intelligent-search-server")
        self._setup_tools()
        self._setup_routes()
        
    def _setup_tools(self):
        """Setup MCP tools"""
        
        @self.app.tool()
        async def web_search(query: str, category: str = "auto", num_results: int = 5) -> str:
            """
            Intelligent web search with automatic content fetching.
            
            Args:
                query: The search query to execute
                category: Search category (auto, news, academic, technical, general, shopping, images, local)
                num_results: Number of results to return (1-10)
                
            Returns:
                Extracted content from relevant web sources
            """
            
            try:
                logger.info(f"Web search request: '{query}' (category: {category})")
                
                # Use orchestrator for complete search workflow
                result = await self.orchestrator.search_and_fetch(query)
                
                if "error" in result:
                    return result["error"]
                
                content = result.get("content", "No content available")
                
                # Log statistics
                stats = result.get("stats", {})
                logger.info(f"Search completed: {stats.get('successful_fetches', 0)} sources fetched "
                           f"in {result.get('processing_time', 0):.2f}s")
                
                return content
                
            except Exception as e:
                error_msg = handle_error(e, "web_search")
                logger.error(f"Web search failed: {error_msg}")
                return error_msg
        
        @self.app.tool()
        async def fetch_url(url: str, mode: str = "partial", max_tokens: int = 2000) -> str:
            """
            Fetch and extract content from a specific URL.
            
            Args:
                url: The URL to fetch content from
                mode: "snippet" (basic), "partial" (moderate), "complete" (full content)
                max_tokens: Maximum tokens (for compatibility, actual limit is in config)
                
            Returns:
                Extracted content from the URL
            """
            
            try:
                logger.info(f"URL fetch request: '{url}' (mode: {mode})")
                
                content = await self.web_fetcher.fetch_url(url, mode)
                
                logger.info(f"URL fetch completed: {len(content)} characters")
                return content
                
            except Exception as e:
                error_msg = handle_error(e, "fetch_url")
                logger.error(f"URL fetch failed: {error_msg}")
                return error_msg
        
        @self.app.tool()
        async def deep_search(query: str, mode: str = "intensive", max_pages: int = None) -> str:
            # Get current temporal context
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')
            knowledge_cutoff = config.get_knowledge_cutoff().strftime('%Y-%m-%d')
            
            f"""
            Deep Search Intelligence System
            
            Performs comprehensive web crawling with AI-powered link discovery and content analysis.
            Designed for thorough research and information gathering across multiple sources.
            use this tool when user asks for deep research or information gathering.
            
            TEMPORAL CONTEXT: Current date is {current_date}. Knowledge cutoff is {knowledge_cutoff}.
            For queries about recent events, conflicts, or developments since {knowledge_cutoff}, 
            this tool will automatically search for and prioritize current information.
            
            Args:
                query: the search query to execute
                mode: Intelligence level ("standard", "intensive", "ultra")
                max_pages: Override maximum pages to crawl (optional)
                
            Returns:
                Comprehensive research report with detailed analysis and source citations
            """
            
            try:
                logger.info(f"🚀 DEEP SEARCH INITIATED: '{query}' (mode: {mode})")
                
                # Execute the bragging rights deep search
                result = await deep_search_tool.execute_deep_search(query, mode)
                
                if not result.get("success", False):
                    return f"❌ Deep search failed: {result.get('error', 'Unknown error')}"
                
                # Log impressive stats
                stats = result.get("stats", {})
                logger.info(f"🎉 DEEP SEARCH COMPLETE: {stats.get('pages_crawled', 0)} pages, "
                        f"{stats.get('domains_hit', 0)} domains, "
                        f"{result.get('processing_time', 0):.1f}s "
                        f"({stats.get('crawl_speed', 0):.1f} pages/sec)")
                
                return result["summary"]
                
            except Exception as e:
                error_msg = handle_error(e, "deep_search")
                logger.error(f"Deep search failed: {error_msg}")
                return error_msg
    
    def _setup_routes(self):
        """Setup custom HTTP routes"""
        
        @self.app.custom_route("/health", methods=["GET"])
        async def health_check(request):
            """Health check endpoint with detailed status"""
            from starlette.responses import JSONResponse
            from datetime import datetime
            
            try:
                # Get component status
                stats = await self.orchestrator.get_stats()
                pool_stats = browser_pool.get_stats()
                
                return JSONResponse({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "3.0",
                    "components": {
                        "search_engine": stats["search_engine"],
                        "browser_pool": pool_stats,
                        "query_analyzer": stats["query_analyzer"]
                    },
                    "config": config.get_status_info()
                })
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse({
                    "status": "degraded",
                    "error": str(e)
                }, status_code=503)
        
        @self.app.custom_route("/", methods=["GET"])
        async def root(request):
            """Root endpoint with server information"""
            from starlette.responses import JSONResponse
            
            return JSONResponse({
                "name": "Intelligent Search MCP Server",
                "version": "3.0",
                "description": "Production-ready MCP server with intelligent search capabilities",
                "endpoints": {
                    "mcp": "/mcp",
                    "health": "/health"
                },
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Intelligent web search with automatic content fetching"
                    },
                    {
                        "name": "fetch_url", 
                        "description": "Direct URL content fetching"
                    }
                ],
                "features": [
                    "AI-powered query analysis",
                    "Multi-provider search support (SerpAPI, Google CSE)",
                    "Intelligent content extraction",
                    "Memory-safe caching",
                    "Resource pooling",
                    "Production-ready error handling"
                ],
                "config": config.get_status_info()
            })
    
    async def startup(self):
        """Server startup tasks"""
        logger.info("🚀 Starting Intelligent Search MCP Server v3.0")
        
        # Get provider info
        provider_info = config.get_status_info()
        primary_provider = provider_info["search_providers"]["primary"]
        available_providers = provider_info["search_providers"]["available"]
        
        logger.info(f"🔧 Search: {', '.join(available_providers)} (primary: {primary_provider})")
        logger.info(f"🌐 Server will start on: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        
        # Initialize browser pool
        await initialize_browser_pool()
        logger.info("✅ Browser pool initialized")
        
        logger.info("🎯 MCP Tools: web_search, fetch_url")
        logger.info("📊 Health check: /health")
        logger.info("🔄 Ready for requests")
    
    async def shutdown(self):
        """Server shutdown tasks"""
        logger.info("🛑 Shutting down server...")
        
        # Cleanup browser pool
        await cleanup_browser_pool()
        logger.info("✅ Browser pool cleaned up")
        
        logger.info("👋 Server stopped")
    
    def run(self):
        """Run the server with proper FastMCP usage"""
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, starting graceful shutdown...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Initialize in sync context
            asyncio.run(self.startup())
            
            # Run server with correct FastMCP API
            # FastMCP.run() only accepts transport parameter
            self.app.run(transport="streamable-http")
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Cleanup in sync context
            asyncio.run(self.shutdown())

def main():
    """Main entry point"""
    try:
        server = MCPSearchServer()
        server.run()
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        print("\n💡 Please check your .env file and ensure you have:")
        print("   - SERPAPI_KEY (or GOOGLE_API_KEY + GOOGLE_CSE_ID)")
        print("   - OPENAI_API_KEY")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
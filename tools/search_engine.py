#!/usr/bin/env python3
"""
Modular Search Engine
Adapts to available providers - works perfectly with just SerpAPI or multiple providers.
"""

import asyncio
import requests
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Import from your structure
from configurations.config import config, SearchProvider
from configurations.exceptions import SearchError, SearchProviderError, SearchTimeoutError

logger = logging.getLogger(__name__)

class SearchCategory(Enum):
    NEWS = "news"
    ACADEMIC = "academic" 
    TECHNICAL = "technical"
    GENERAL = "general"
    SHOPPING = "shopping"
    IMAGES = "images"
    LOCAL = "local"

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int = 0
    source: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ModularSearchEngine:
    """Modular search engine that adapts to available providers"""
    
    def __init__(self):
        self.timeout = config.TIMEOUT_SECONDS
        self.available_providers = config.get_available_search_providers()
        self.primary_provider = config.get_primary_search_provider()
        
        # Log what we have available
        if self.primary_provider == SearchProvider.NONE:
            logger.warning("⚠️ No search providers configured!")
        else:
            provider_list = [p.value for p in self.available_providers]
            logger.info(f"🔍 Search engine initialized: {', '.join(provider_list)} (primary: {self.primary_provider.value})")
    
    def is_available(self) -> bool:
        """Check if any search provider is available"""
        return self.primary_provider != SearchProvider.NONE
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get detailed provider status"""
        return {
            "available_providers": [p.value for p in self.available_providers],
            "primary_provider": self.primary_provider.value,
            "serpapi_available": SearchProvider.SERPAPI in self.available_providers,
            "google_cse_available": SearchProvider.GOOGLE_CSE in self.available_providers,
            "total_providers": len(self.available_providers)
        }
    
    async def search(self, query: str, category: SearchCategory = SearchCategory.GENERAL, 
                    num_results: int = 5, preferred_provider: Optional[str] = None) -> List[SearchResult]:
        """
        Main search method with automatic provider selection and fallback
        
        Args:
            query: Search query
            category: Search category
            num_results: Number of results to return
            preferred_provider: Optional provider preference for this search
        """
        
        if not self.is_available():
            raise SearchProviderError("No search providers configured. Please set SERPAPI_KEY or GOOGLE_API_KEY+GOOGLE_CSE_ID")
        
        # Determine which provider to use
        provider_to_use = self._select_provider(preferred_provider)
        
        logger.info(f"🔍 Searching with {provider_to_use.value}: '{query[:50]}...' ({category.value})")
        
        try:
            # Run search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            if provider_to_use == SearchProvider.SERPAPI:
                results = await loop.run_in_executor(
                    None, self._serpapi_search, query, category, num_results
                )
            elif provider_to_use == SearchProvider.GOOGLE_CSE:
                results = await loop.run_in_executor(
                    None, self._google_cse_search, query, category, num_results
                )
            else:
                raise SearchProviderError(f"Provider {provider_to_use.value} not available")
            
            logger.info(f"✅ Search completed: {len(results)} results from {provider_to_use.value}")
            return results
            
        except Exception as e:
            # Try fallback if we have multiple providers
            if len(self.available_providers) > 1:
                return await self._try_fallback_provider(query, category, num_results, provider_to_use, e)
            else:
                # No fallback available
                logger.error(f"Search failed with only provider {provider_to_use.value}: {e}")
                if "timeout" in str(e).lower():
                    raise SearchTimeoutError(f"Search timed out for query: {query}")
                else:
                    raise SearchProviderError(f"Search failed: {str(e)}")
    
    def _select_provider(self, preferred_provider: Optional[str] = None) -> SearchProvider:
        """Select which provider to use for this search"""
        
        # If user specified a preference for this search, try to honor it
        if preferred_provider:
            if preferred_provider.lower() == "serpapi" and SearchProvider.SERPAPI in self.available_providers:
                return SearchProvider.SERPAPI
            elif preferred_provider.lower() == "google_cse" and SearchProvider.GOOGLE_CSE in self.available_providers:
                return SearchProvider.GOOGLE_CSE
        
        # Fall back to primary provider
        return self.primary_provider
    
    async def _try_fallback_provider(self, query: str, category: SearchCategory, 
                                   num_results: int, failed_provider: SearchProvider, 
                                   original_error: Exception) -> List[SearchResult]:
        """Try fallback provider if primary fails"""
        
        # Find alternative provider
        fallback_provider = None
        for provider in self.available_providers:
            if provider != failed_provider:
                fallback_provider = provider
                break
        
        if not fallback_provider:
            raise SearchProviderError(f"No fallback available. Original error: {str(original_error)}")
        
        logger.warning(f"🔄 Primary provider {failed_provider.value} failed, trying fallback {fallback_provider.value}")
        
        try:
            loop = asyncio.get_event_loop()
            
            if fallback_provider == SearchProvider.SERPAPI:
                results = await loop.run_in_executor(
                    None, self._serpapi_search, query, category, num_results
                )
            elif fallback_provider == SearchProvider.GOOGLE_CSE:
                results = await loop.run_in_executor(
                    None, self._google_cse_search, query, category, num_results
                )
            else:
                raise SearchProviderError(f"Unknown fallback provider: {fallback_provider.value}")
            
            logger.info(f"✅ Fallback successful: {len(results)} results from {fallback_provider.value}")
            return results
            
        except Exception as fallback_error:
            logger.error(f"Both providers failed. Primary: {original_error}, Fallback: {fallback_error}")
            raise SearchProviderError(f"All providers failed. Primary error: {str(original_error)}")
    
    def _serpapi_search(self, query: str, category: SearchCategory, num_results: int) -> List[SearchResult]:
        """Search using SerpAPI"""
        
        if not config.SERPAPI_KEY:
            raise SearchProviderError("SerpAPI key not configured")
        
        params = {
            "api_key": config.SERPAPI_KEY,
            "q": query,
            "num": min(num_results, 10),
            "engine": "google",
            "gl": "us",
            "hl": "en"
        }
        
        # Category-specific parameters
        if category == SearchCategory.NEWS:
            params["tbm"] = "nws"
            params["tbs"] = "qdr:w"  # Past week
        elif category == SearchCategory.SHOPPING:
            params["tbm"] = "shop"
        elif category == SearchCategory.IMAGES:
            params["tbm"] = "isch"
        elif category == SearchCategory.LOCAL:
            params["tbm"] = "lcl"
        elif category == SearchCategory.ACADEMIC:
            params["q"] = f"{query} site:scholar.google.com OR site:arxiv.org"
        elif category == SearchCategory.TECHNICAL:
            params["q"] = f"{query} site:stackoverflow.com OR site:github.com"
        
        try:
            response = requests.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_serpapi_results(data, category)
            
        except requests.exceptions.Timeout:
            raise SearchTimeoutError("SerpAPI request timed out")
        except requests.exceptions.RequestException as e:
            raise SearchProviderError(f"SerpAPI error: {str(e)}")
    
    def _google_cse_search(self, query: str, category: SearchCategory, num_results: int) -> List[SearchResult]:
        """Search using Google Custom Search Engine"""
        
        if not (config.GOOGLE_API_KEY and config.GOOGLE_CSE_ID):
            raise SearchProviderError("Google CSE not configured")
        
        # Modify query based on category
        modified_query = query
        if category == SearchCategory.NEWS:
            modified_query = f"{query} site:reuters.com OR site:bbc.com OR site:cnn.com"
        elif category == SearchCategory.ACADEMIC:
            modified_query = f"{query} site:scholar.google.com OR site:arxiv.org"
        elif category == SearchCategory.TECHNICAL:
            modified_query = f"{query} site:stackoverflow.com OR site:github.com"
        
        params = {
            "key": config.GOOGLE_API_KEY,
            "cx": config.GOOGLE_CSE_ID,
            "q": modified_query,
            "num": min(num_results, 10),
            "safe": "medium"
        }
        
        if category == SearchCategory.IMAGES:
            params["searchType"] = "image"
        
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1", 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown Google CSE error")
                raise SearchProviderError(f"Google CSE error: {error_msg}")
            
            return self._parse_google_cse_results(data)
            
        except requests.exceptions.Timeout:
            raise SearchTimeoutError("Google CSE request timed out")
        except requests.exceptions.RequestException as e:
            raise SearchProviderError(f"Google CSE error: {str(e)}")
    
    def _parse_serpapi_results(self, data: dict, category: SearchCategory) -> List[SearchResult]:
        """Parse SerpAPI response"""
        results = []
        
        # Handle different result types
        if category == SearchCategory.NEWS and "news_results" in data:
            items = data["news_results"]
            source_prefix = "News"
        elif category == SearchCategory.SHOPPING and "shopping_results" in data:
            items = data["shopping_results"]
            source_prefix = "Shopping"
        elif "organic_results" in data:
            items = data["organic_results"]
            source_prefix = "Search"
        else:
            return results
        
        for i, item in enumerate(items):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                rank=i + 1,
                source=f"SerpAPI {source_prefix}",
                metadata={
                    "category": category.value,
                    "provider": "serpapi",
                    "date": item.get("date", ""),
                    "price": item.get("price", "")
                }
            ))
        
        return results
    
    def _parse_google_cse_results(self, data: dict) -> List[SearchResult]:
        """Parse Google CSE response"""
        results = []
        items = data.get("items", [])
        
        for i, item in enumerate(items):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                rank=i + 1,
                source="Google CSE",
                metadata={
                    "provider": "google_cse",
                    "formatted_url": item.get("formattedUrl", ""),
                    "display_link": item.get("displayLink", "")
                }
            ))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive search engine status"""
        return {
            "available": self.is_available(),
            "providers": self.get_provider_status(),
            "timeout": self.timeout,
            "supported_categories": [cat.value for cat in SearchCategory],
            "features": {
                "fallback_support": len(self.available_providers) > 1,
                "category_optimization": True,
                "async_support": True
            }
        }

# Global search engine instance
search_engine = ModularSearchEngine()
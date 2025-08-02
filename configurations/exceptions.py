#!/usr/bin/env python3
"""
Standardized Exception Handling
Provides consistent error handling across the entire application.
"""

class SearchServerError(Exception):
    """Base exception for all search server errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details or ""
        super().__init__(self.message)

class ConfigurationError(SearchServerError):
    """Raised when configuration is invalid or missing"""
    pass

class SearchError(SearchServerError):
    """Base exception for search-related errors"""
    pass

class SearchProviderError(SearchError):
    """Error with external search provider (SerpAPI, Google CSE)"""
    pass

class SearchTimeoutError(SearchError):
    """Search operation timed out"""
    pass

class FetchError(SearchServerError):
    """Base exception for content fetching errors"""
    pass

class FetchTimeoutError(FetchError):
    """Content fetching timed out"""
    pass

class ContentExtractionError(FetchError):
    """Failed to extract meaningful content from URL"""
    pass

class AnalysisError(SearchServerError):
    """Error during query analysis"""
    pass

class LLMError(AnalysisError):
    """Error communicating with LLM provider"""
    pass

class ResourceExhaustionError(SearchServerError):
    """System resources exhausted (memory, connections, etc.)"""
    pass

class CrawlError(SearchServerError):
    """Error during deep crawling operations"""
    pass

def handle_error(error: Exception, context: str = "") -> str:
    """
    Standardized error handling that returns user-friendly error messages
    """
    context_prefix = f"[{context}] " if context else ""
    
    if isinstance(error, SearchTimeoutError):
        return f"⏱️ {context_prefix}Search timed out. Please try again with a simpler query."
    
    elif isinstance(error, SearchProviderError):
        return f"🔍 {context_prefix}Search service unavailable. Please try again later."
    
    elif isinstance(error, FetchTimeoutError):
        return f"🌐 {context_prefix}Content fetching timed out. Some URLs may be unavailable."
    
    elif isinstance(error, ContentExtractionError):
        return f"📄 {context_prefix}Could not extract content from the webpage."
    
    elif isinstance(error, LLMError):
        return f"🧠 {context_prefix}Query analysis failed. Please try rephrasing your query."
    
    elif isinstance(error, ResourceExhaustionError):
        return f"⚠️ {context_prefix}System overloaded. Please try again in a moment."
    
    elif isinstance(error, ConfigurationError):
        return f"⚙️ {context_prefix}Server configuration error. Please contact administrator."
    
    else:
        return f"❌ {context_prefix}Unexpected error: {str(error)}"
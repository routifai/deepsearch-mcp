#!/usr/bin/env python3
"""
Modular Configuration Management
Supports different provider configurations - works with SerpAPI only or multiple providers.
"""

import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum

class SearchProvider(Enum):
    SERPAPI = "serpapi"
    GOOGLE_CSE = "google_cse"
    TAVILY = "tavily"
    NONE = "none"

@dataclass
class Config:
    """Modular configuration that adapts to available providers"""
    
    # API Keys (optional - system detects what's available)
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LLM Configuration
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_CLIENT_TYPE: str = os.getenv("LLM_CLIENT_TYPE", "default")  # "default" or "custom"
    
    # Performance Settings
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "20"))
    MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "3"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    PAGE_TIMEOUT: int = int(os.getenv("PAGE_TIMEOUT", "10"))
    
    # Memory Management
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Crawling Settings
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "2"))
    DELAY_SECONDS: float = float(os.getenv("DELAY_SECONDS", "0.5"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    MAX_LINKS_PER_PAGE: int = int(os.getenv("MAX_LINKS_PER_PAGE", "5"))
    
    # Browser Pool Settings
    BROWSER_POOL_SIZE: int = int(os.getenv("BROWSER_POOL_SIZE", "3"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Knowledge Cutoff
    KNOWLEDGE_CUTOFF: str = os.getenv("KNOWLEDGE_CUTOFF", "2025-01-31")
    
    # Server Settings
    SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # Provider Preference (optional - system auto-detects if not set)
    PREFERRED_SEARCH_PROVIDER: str = os.getenv("PREFERRED_SEARCH_PROVIDER", "auto")
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation"""
        self._available_providers = self._detect_providers()
        self._primary_provider = self._select_primary_provider()
        self._validation_status = self._validate_config()
    
    def _detect_providers(self) -> List[SearchProvider]:
        """Detect available search providers based on API keys"""
        providers = []
        
        if self.SERPAPI_KEY:
            providers.append(SearchProvider.SERPAPI)
        
        if self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID:
            providers.append(SearchProvider.GOOGLE_CSE)
        
        if self.TAVILY_API_KEY:
            providers.append(SearchProvider.TAVILY)
        
        return providers
    
    def _select_primary_provider(self) -> SearchProvider:
        """Select the primary search provider to use"""
        if not self._available_providers:
            return SearchProvider.NONE
        
        # If user specified a preference, use it if available
        if self.PREFERRED_SEARCH_PROVIDER.lower() == "serpapi" and SearchProvider.SERPAPI in self._available_providers:
            return SearchProvider.SERPAPI
        elif self.PREFERRED_SEARCH_PROVIDER.lower() == "google_cse" and SearchProvider.GOOGLE_CSE in self._available_providers:
            return SearchProvider.GOOGLE_CSE
        elif self.PREFERRED_SEARCH_PROVIDER.lower() == "tavily" and SearchProvider.TAVILY in self._available_providers:
            return SearchProvider.TAVILY
        
        # Auto-select: prefer Tavily if available, then SerpAPI, otherwise use first available
        if SearchProvider.TAVILY in self._available_providers:
            return SearchProvider.TAVILY
        elif SearchProvider.SERPAPI in self._available_providers:
            return SearchProvider.SERPAPI
        else:
            return self._available_providers[0]
    
    def _validate_config(self) -> dict:
        """Single validation method that returns comprehensive status"""
        status = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "search": {},
            "llm": {}
        }
        
        # Validate search configuration
        if not self._available_providers:
            status["valid"] = False
            status["errors"].append("No search provider configured. Please set SERPAPI_KEY, TAVILY_API_KEY, or (GOOGLE_API_KEY + GOOGLE_CSE_ID)")
            
            # Provide specific guidance based on what's missing
            if not self.SERPAPI_KEY and not self.TAVILY_API_KEY and not (self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID):
                status["warnings"].append("No API keys found. You need at least one search provider configured.")
            elif self.GOOGLE_API_KEY and not self.GOOGLE_CSE_ID:
                status["warnings"].append("Google API key found but missing GOOGLE_CSE_ID")
            elif self.GOOGLE_CSE_ID and not self.GOOGLE_API_KEY:
                status["warnings"].append("Google CSE ID found but missing GOOGLE_API_KEY")
        else:
            provider_names = [p.value for p in self._available_providers]
            status["search"] = {
                "available": provider_names,
                "primary": self._primary_provider.value,
                "count": len(self._available_providers)
            }
            
            # Add helpful info about provider selection
            if len(self._available_providers) > 1:
                status["warnings"].append(f"Multiple providers available: {', '.join(provider_names)}. Using {self._primary_provider.value} as primary.")
        
        # Validate LLM configuration
        if not self.OPENAI_API_KEY:
            status["valid"] = False
            status["errors"].append("OpenAI API key required for query analysis (OPENAI_API_KEY)")
        else:
            status["llm"] = {
                "configured": True,
                "client_type": self.LLM_CLIENT_TYPE,
                "model": self.OPENAI_MODEL
            }
            
            # Check custom LLM configuration
            if self.LLM_CLIENT_TYPE == "custom" and not self.OPENAI_BASE_URL:
                status["valid"] = False
                status["errors"].append("Custom LLM client requires OPENAI_BASE_URL")
            elif self.LLM_CLIENT_TYPE == "custom":
                status["warnings"].append(f"Using custom LLM client with base URL: {self.OPENAI_BASE_URL}")
        
        return status
    
    # Simplified public interface
    def get_available_search_providers(self) -> List[SearchProvider]:
        """Get list of available search providers"""
        return self._available_providers
    
    def get_primary_search_provider(self) -> SearchProvider:
        """Get the primary search provider"""
        return self._primary_provider
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return self._validation_status["valid"]
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self._validation_status["errors"]
    
    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings"""
        return self._validation_status.get("warnings", [])
    
    def get_knowledge_cutoff(self) -> datetime:
        """Get knowledge cutoff as datetime object"""
        return datetime.fromisoformat(self.KNOWLEDGE_CUTOFF)
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration for client initialization"""
        return {
            "api_key": self.OPENAI_API_KEY,
            "base_url": self.OPENAI_BASE_URL if self.LLM_CLIENT_TYPE == "custom" else None,
            "model": self.OPENAI_MODEL,
            "client_type": self.LLM_CLIENT_TYPE
        }
    
    def get_status_info(self) -> dict:
        """Get configuration status for health checks"""
        return {
            "search_providers": {
                "available": [p.value for p in self._available_providers],
                "primary": self._primary_provider.value,
                "serpapi_configured": bool(self.SERPAPI_KEY),
                "google_cse_configured": bool(self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID),
                "tavily_configured": bool(self.TAVILY_API_KEY)
            },
            "llm": {
                "configured": bool(self.OPENAI_API_KEY),
                "client_type": self.LLM_CLIENT_TYPE,
                "base_url": self.OPENAI_BASE_URL if self.LLM_CLIENT_TYPE == "custom" else "default",
                "model": self.OPENAI_MODEL,
                "knowledge_cutoff": self.KNOWLEDGE_CUTOFF
            },
            "performance": {
                "max_concurrent": self.MAX_CONCURRENT,
                "timeout_seconds": self.TIMEOUT_SECONDS,
                "cache_size": self.MAX_CACHE_SIZE
            },
            "validation": {
                "valid": self._validation_status["valid"],
                "errors": self._validation_status["errors"],
                "warnings": self._validation_status.get("warnings", [])
            }
        }

# Global config instance
config = Config()

def validate_startup_config() -> bool:
    """Validate configuration at startup with helpful messages"""
    print("🔧 Validating Configuration...")
    
    status = config._validation_status
    
    # Show warnings if any
    if status.get("warnings"):
        print("⚠️ Configuration Warnings:")
        for warning in status["warnings"]:
            print(f"   {warning}")
        print()
    
    if config.is_valid():
        print("✅ Configuration Valid:")
        print(f"   Search: {len(status['search'].get('available', []))} provider(s) available")
        print(f"   LLM: {status['llm'].get('model', 'Not configured')}")
        print("🎉 All systems ready!")
        return True
    else:
        print("❌ Configuration Error:")
        for error in config.get_validation_errors():
            print(f"   {error}")
        
        print("\n💡 Quick Setup Guide:")
        print("   1. For SerpAPI: Set SERPAPI_KEY in your .env file")
        print("   2. For Tavily: Set TAVILY_API_KEY in your .env file")
        print("   3. For Google CSE: Set both GOOGLE_API_KEY and GOOGLE_CSE_ID")
        print("   4. For LLM: Set OPENAI_API_KEY")
        print("   5. You only need ONE search provider to get started!")
        return False
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
    NONE = "none"

@dataclass
class Config:
    """Modular configuration that adapts to available providers"""
    
    # API Keys (optional - system detects what's available)
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
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
    
    def get_available_search_providers(self) -> List[SearchProvider]:
        """Get list of available search providers based on API keys"""
        providers = []
        
        if self.SERPAPI_KEY:
            providers.append(SearchProvider.SERPAPI)
        
        if self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID:
            providers.append(SearchProvider.GOOGLE_CSE)
        
        return providers
    
    def get_primary_search_provider(self) -> SearchProvider:
        """Get the primary search provider to use"""
        available = self.get_available_search_providers()
        
        if not available:
            return SearchProvider.NONE
        
        # If user specified a preference, use it if available
        if self.PREFERRED_SEARCH_PROVIDER.lower() == "serpapi" and SearchProvider.SERPAPI in available:
            return SearchProvider.SERPAPI
        elif self.PREFERRED_SEARCH_PROVIDER.lower() == "google_cse" and SearchProvider.GOOGLE_CSE in available:
            return SearchProvider.GOOGLE_CSE
        
        # Auto-select: prefer SerpAPI if available, otherwise use first available
        if SearchProvider.SERPAPI in available:
            return SearchProvider.SERPAPI
        else:
            return available[0]
    
    def validate_search_config(self) -> tuple[bool, str]:
        """Validate search configuration and return status"""
        available = self.get_available_search_providers()
        
        if not available:
            return False, "No search provider configured. Please set SERPAPI_KEY or (GOOGLE_API_KEY + GOOGLE_CSE_ID)"
        
        primary = self.get_primary_search_provider()
        provider_names = [p.value for p in available]
        
        return True, f"Search configured with {len(available)} provider(s): {', '.join(provider_names)}. Using: {primary.value}"
    
    def validate_llm_config(self) -> tuple[bool, str]:
        """Validate LLM configuration"""
        if not self.OPENAI_API_KEY:
            return False, "OpenAI API key required for query analysis (OPENAI_API_KEY)"
        
        return True, "LLM configuration valid"
    
    def validate(self) -> bool:
        """Validate overall configuration"""
        search_valid, search_msg = self.validate_search_config()
        llm_valid, llm_msg = self.validate_llm_config()
        
        if not search_valid or not llm_valid:
            print(f"❌ Configuration Error:")
            if not search_valid:
                print(f"   Search: {search_msg}")
            if not llm_valid:
                print(f"   LLM: {llm_msg}")
            return False
        
        print(f"✅ Configuration Valid:")
        print(f"   Search: {search_msg}")
        print(f"   LLM: {llm_msg}")
        return True
    
    def get_knowledge_cutoff(self) -> datetime:
        """Get knowledge cutoff as datetime object"""
        return datetime.fromisoformat(self.KNOWLEDGE_CUTOFF)
    
    def get_status_info(self) -> dict:
        """Get configuration status for health checks"""
        available_providers = self.get_available_search_providers()
        primary_provider = self.get_primary_search_provider()
        
        return {
            "search_providers": {
                "available": [p.value for p in available_providers],
                "primary": primary_provider.value,
                "serpapi_configured": bool(self.SERPAPI_KEY),
                "google_cse_configured": bool(self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID)
            },
            "llm": {
                "configured": bool(self.OPENAI_API_KEY),
                "knowledge_cutoff": self.KNOWLEDGE_CUTOFF
            },
            "performance": {
                "max_concurrent": self.MAX_CONCURRENT,
                "timeout_seconds": self.TIMEOUT_SECONDS,
                "cache_size": self.MAX_CACHE_SIZE
            }
        }

# Global config instance
config = Config()

def validate_startup_config() -> bool:
    """Validate configuration at startup with helpful messages"""
    print("🔧 Validating Configuration...")
    
    if config.validate():
        print("🎉 All systems ready!")
        return True
    else:
        print("\n💡 Quick Setup Guide:")
        print("   1. For SerpAPI: Set SERPAPI_KEY in your .env file")
        print("   2. For Google CSE: Set both GOOGLE_API_KEY and GOOGLE_CSE_ID")
        print("   3. For LLM: Set OPENAI_API_KEY")
        print("   4. You only need ONE search provider to get started!")
        return False
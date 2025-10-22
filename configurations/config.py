#!/usr/bin/env python3
"""
Simple Configuration Management
Just the essentials for the MCP search server.
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Simple configuration for MCP search server"""
    
    # API Keys
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Server Settings
    SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Timeout
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    
    
    def get_available_providers(self) -> List[str]:
        """Get list of available search providers"""
        providers = []
        
        if self.SERPAPI_KEY:
            providers.append("serpapi")
        
        if self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID:
            providers.append("google_cse")
        
        if self.TAVILY_API_KEY:
            providers.append("tavily")
        
        return providers
    
    def get_primary_provider(self) -> str:
        """Get the primary search provider"""
        providers = self.get_available_providers()
        if not providers:
            return "none"
        
        # Simple preference: tavily > serpapi > google_cse
        if "tavily" in providers:
            return "tavily"
        elif "serpapi" in providers:
            return "serpapi"
        else:
            return providers[0]
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.get_available_providers()) > 0
    
    
    def get_status_info(self) -> dict:
        """Get configuration status for health checks"""
        return {
            "search_providers": {
                "available": self.get_available_providers(),
                "primary": self.get_primary_provider(),
                "serpapi_configured": bool(self.SERPAPI_KEY),
                "google_cse_configured": bool(self.GOOGLE_API_KEY and self.GOOGLE_CSE_ID),
                "tavily_configured": bool(self.TAVILY_API_KEY)
            },
            "server": {
                "host": self.SERVER_HOST,
                "port": self.SERVER_PORT
            }
        }

# Global config instance
config = Config()

def validate_startup_config() -> bool:
    """Validate configuration at startup"""
    # Only print validation messages in HTTP mode to avoid MCP JSON parsing errors
    import sys
    is_stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "--stdio"
    
    if not is_stdio_mode:
        print("Validating Configuration...")
    
    if config.is_valid():
        providers = config.get_available_providers()
        primary = config.get_primary_provider()
        
        if not is_stdio_mode:
            print(f"Configuration Valid:")
            print(f"   Search providers: {', '.join(providers)} (primary: {primary})")
            print(f"   Server: {config.SERVER_HOST}:{config.SERVER_PORT}")
            print("Ready to start!")
        return True
    else:
        if not is_stdio_mode:
            print("Configuration Error: No search providers configured")
            print("\nQuick Setup Guide:")
            print("   1. Set SERPAPI_KEY in your .env file, OR")
            print("   2. Set TAVILY_API_KEY in your .env file, OR") 
            print("   3. Set both GOOGLE_API_KEY and GOOGLE_CSE_ID")
            print("   4. You only need ONE search provider!")
        return False
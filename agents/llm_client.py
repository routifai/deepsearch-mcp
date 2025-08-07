#!/usr/bin/env python3
"""
Centralized LLM Client
Provides a unified interface for LLM operations with proper error handling and configuration.
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import openai

logger = logging.getLogger(__name__)

class LLMClientInterface(ABC):
    """Abstract interface for LLM clients"""
    
    @abstractmethod
    def get_sync_client(self):
        """Get synchronous client"""
        pass
    
    @abstractmethod
    def get_async_client(self):
        """Get asynchronous client"""
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Get the model name"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is properly configured"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection"""
        pass

class OpenAIClient(LLMClientInterface):
    """OpenAI client implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        # Get API key from environment if not provided
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize clients
        self._sync_client = None
        self._async_client = None
        
        logger.info(f"OpenAI client initialized with model: {self._model}")
    
    def get_sync_client(self):
        """Get synchronous OpenAI client"""
        if self._sync_client is None:
            try:
                self._sync_client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=30
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI sync client: {e}")
                raise
        
        return self._sync_client
    
    def get_async_client(self):
        """Get asynchronous OpenAI client"""
        if self._async_client is None:
            try:
                self._async_client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=30
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI async client: {e}")
                raise
        
        return self._async_client
    
    @property
    def model(self) -> str:
        """Get the model name"""
        return self._model
    
    def is_available(self) -> bool:
        """Check if OpenAI client is properly configured"""
        try:
            return bool(self.api_key and self.get_sync_client())
        except Exception:
            return False
    
    def test_connection(self) -> bool:
        """Test the connection with a minimal request"""
        try:
            client = self.get_sync_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )
            logger.debug("OpenAI client test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI client test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "openai",
            "model": self._model,
            "max_tokens": 4096 if "gpt-4o-mini" in self._model else 8192,
            "supports_functions": True,
            "supports_streaming": True
        }

# Global client instance management
_global_client: Optional[LLMClientInterface] = None

def get_llm_client() -> LLMClientInterface:
    """Get the global LLM client instance using centralized configuration"""
    global _global_client
    
    if _global_client is None:
        # Import config here to avoid circular imports
        from configurations.config import config
        llm_config = config.get_llm_config()
        
        if llm_config["client_type"] == "custom" and llm_config["base_url"]:
            _global_client = create_custom_client(
                base_url=llm_config["base_url"],
                model=llm_config["model"],
                api_key=llm_config["api_key"]
            )
        else:
            _global_client = create_default_client(model=llm_config["model"])
    
    return _global_client

def create_default_client(model: str = "gpt-4o-mini") -> OpenAIClient:
    """Create default OpenAI client"""
    from configurations.config import config
    return OpenAIClient(api_key=config.OPENAI_API_KEY, model=model)

def create_custom_client(base_url: str, model: str = "gpt-4o-mini", api_key: str = None) -> OpenAIClient:
    """Create custom OpenAI client with custom base URL"""
    from configurations.config import config
    api_key = api_key or config.OPENAI_API_KEY
    client = OpenAIClient(api_key=api_key, model=model)
    
    # Set custom base URL for both sync and async clients
    if client._sync_client:
        client._sync_client.base_url = base_url
    if client._async_client:
        client._async_client.base_url = base_url
    
    logger.info(f"Created custom OpenAI client with base URL: {base_url}")
    return client

def set_global_client(client: LLMClientInterface):
    """Set the global LLM client instance"""
    global _global_client
    _global_client = client
    logger.info(f"Global LLM client set to: {type(client).__name__}")

def reset_global_client():
    """Reset the global LLM client"""
    global _global_client
    _global_client = None
    logger.info("Global LLM client reset")
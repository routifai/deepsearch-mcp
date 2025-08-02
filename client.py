#!/usr/bin/env python3
"""
MCP Client with Streamable HTTP Transport
Interactive client using OpenAI to intelligently interact with FastMCP tools.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

# Import our centralized LLM client (using your structure)
from agents.llm_client import get_llm_client, create_default_client, create_custom_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIMCPClient:
    """Interactive MCP client using OpenAI for intelligent tool usage"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8000/mcp",
                 llm_client_type: str = "default",
                 custom_base_url: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        self.server_url = server_url
        
        # Initialize LLM client based on type
        if llm_client_type == "default":
            self.llm_client = create_default_client(model)
        elif llm_client_type == "custom":
            if not custom_base_url:
                raise ValueError("custom_base_url is required for custom client type")
            self.llm_client = create_custom_client(custom_base_url, model)
        else:
            raise ValueError("llm_client_type must be 'default' or 'custom'")
        
        self.openai_client = self.llm_client.get_async_client()
        self.model = self.llm_client.model
        
        logger.info(f"MCP Client initialized with model: {self.model}")
    
    async def chat_with_tools(self, user_message: str) -> str:
        """Chat using OpenAI with MCP tools"""
        try:
            # Connect to MCP server for this chat session
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize connection
                    await session.initialize()
                    
                    # Get available tools
                    tools_response = await session.list_tools()
                    
                    # Convert MCP tools to OpenAI format
                    openai_tools = []
                    for tool in tools_response.tools:
                        # FastMCP tools have better schemas by default
                        openai_function = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema or {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                        }
                        openai_tools.append(openai_function)
                    
                    logger.info(f"📋 Loaded {len(openai_tools)} tools for OpenAI")
                    
                    # Create messages for OpenAI
                    messages = [
                        {
                            "role": "system",
                            "content": """You are a helpful AI assistant with access to web search and content fetching tools. 
                            
                            CRITICAL RULE: NEVER modify, rewrite, or change user queries when calling tools. Use the exact query the user provided.
                            
                            Available tools:
                            - web_search: Search the web for current information
                            - fetch_url: Fetch and extract content from specific URLs
                            
                            Use these tools when users:
                            - Ask for current information or recent events
                            - Want to search for specific topics
                            - Ask you to fetch content from URLs
                            - Need information that might be time-sensitive
                            - Request comprehensive research or analysis
                            
                            IMPORTANT: When calling web_search:
                            - Use the EXACT query the user provided
                            - Do NOT add years, dates, or modify the query
                            - Do NOT rewrite or "improve" the user's query
                            - Pass the user's original words as-is
                            
                            Be helpful and use the appropriate tools when needed, but preserve user intent exactly."""
                        },
                        {
                            "role": "user", 
                            "content": user_message
                        }
                    ]
                    
                    # First call to OpenAI
                    response = await self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        max_tokens=1000
                    )
                    
                    response_message = response.choices[0].message
                    messages.append(response_message)
                    
                    # Check if tools were called
                    if response_message.tool_calls:
                        print(f"\n🔧 LLM decided to use tools: {[tc.function.name for tc in response_message.tool_calls]}")
                        
                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            
                            print(f"📞 Calling FastMCP tool '{tool_name}' with args: {tool_args}")
                            
                            # Call the MCP tool
                            try:
                                result = await session.call_tool(tool_name, tool_args)
                                
                                if result.content and len(result.content) > 0:
                                    content = result.content[0]
                                    tool_result = content.text if hasattr(content, 'text') else str(content)
                                else:
                                    tool_result = "No response from MCP tool"
                                
                                print(f"✅ Tool result: {tool_result[:200]}...")
                                
                                # Add tool result to conversation
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                                
                            except McpError as e:
                                logger.error(f"MCP Error: {e}")
                                tool_result = f"MCP Error: {e}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                        
                        # Get final response from OpenAI
                        final_response = await self.openai_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=1000
                        )
                        
                        final_message = final_response.choices[0].message.content
                        print(f"🤖 Final LLM response: {final_message}")
                        
                        return final_message
                    else:
                        print(f"\n💬 LLM responded directly (no tools used): {response_message.content}")
                        return response_message.content
                        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            logger.exception("Chat error details:")
            return f"Error: {e}"
    
    async def test_connection(self) -> bool:
        """Test connection to both LLM and MCP server"""
        try:
            # Test LLM connection
            if not self.llm_client.is_available():
                logger.error("LLM client not available")
                return False
            
            # Test MCP server connection
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    logger.info(f"MCP server connection successful, {len(tools.tools)} tools available")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

async def main():
    """Main interactive chat function"""
    print("🚀 Intelligent Search MCP Client v3.0")
    print("=" * 50)
    
    # Test LLM client connection first
    try:
        llm_client = get_llm_client()
        if not llm_client.test_connection():
            print("❌ Failed to connect to LLM. Check your API key and configuration.")
            return
        
        print("✅ LLM connection successful!")
        
        # Create MCP client
        client = OpenAIMCPClient()
        
        # Test MCP server connection
        if not await client.test_connection():
            print("❌ Failed to connect to MCP server. Make sure the server is running.")
            return
        
        print("✅ MCP server connection successful!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    print("\n🤖 Starting interactive chat...")
    print("💡 The LLM will automatically use MCP tools when appropriate.")
    print("🛑 Type 'quit', 'exit', or 'bye' to exit.")
    print("\n💬 Try these examples:")
    print("   - 'What are the latest developments in AI?' → Should trigger web search")
    print("   - 'Fetch content from https://example.com' → Should trigger URL fetching")
    print("   - 'Search for Python tutorials' → Should trigger web search")
    print("   - 'Current weather in New York' → Should trigger web search")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            # Get LLM response
            print("\n🔄 Processing...")
            response = await client.chat_with_tools(user_input)
            
            print(f"\n🤖 Assistant: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Main loop error:")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
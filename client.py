"""
FastMCP Client using OpenAI GPT-4o-mini
Connects to FastMCP server via Streamable HTTP
"""

import asyncio
import json
import os
from typing import Optional

from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FastMCPOpenAIClient:
    def __init__(self, openai_api_key: str, server_url: str = "http://localhost:8000/mcp"):
        """
        Initialize the FastMCP client with OpenAI
        
        Args:
            openai_api_key: Your OpenAI API key
            server_url: URL of your FastMCP server (default: http://localhost:8000/mcp)
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.server_url = server_url
        self.mcp_client: Optional[Client] = None
        self.tools = []
        
    async def connect(self):
        """Connect to the FastMCP server"""
        print(f"Connecting to FastMCP server at {self.server_url}...")
        
        try:
            # Create FastMCP client - auto-detects Streamable HTTP transport
            self.mcp_client = Client(self.server_url)
            
            # Connect to server
            await self.mcp_client.__aenter__()
            
            # List available tools
            tools_response = await self.mcp_client.list_tools()
            # FastMCP Client returns a list directly, not an object with .tools
            self.tools = tools_response if isinstance(tools_response, list) else tools_response.tools
            
            print(f"✅ Connected! Found {len(self.tools)} tools")
            
            # Display available tools
            if self.tools:
                print("\n📦 Available tools:")
                for tool in self.tools:
                    print(f"   • {tool.name}: {tool.description}")
            
            return self.tools
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to FastMCP server at {self.server_url}\n"
                f"Error: {e}\n\n"
                f"Make sure your server is running:\n"
                f"   fastmcp run server.py --transport http --port 8000\n"
                f"   OR in your server code: mcp.run(transport='http', port=8000)"
            )
    
    async def disconnect(self):
        """Disconnect from the FastMCP server"""
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            print("\n👋 Disconnected from server")
    
    def _format_tools_for_openai(self):
        """Convert FastMCP tools to OpenAI function calling format"""
        openai_tools = []
        for tool in self.tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a FastMCP tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Tool result
        """
        if not self.mcp_client:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        result = await self.mcp_client.call_tool(tool_name, arguments)
        return result
    
    async def chat(self, user_message: str, model: str = "gpt-4o-mini", conversation_history: list = None):
        """
        Send a message to OpenAI and handle tool calls via FastMCP
        
        Args:
            user_message: The user's message
            model: OpenAI model to use (default: gpt-4o-mini)
            conversation_history: Optional existing conversation history
            
        Returns:
            The final response from OpenAI
        """
        if not self.mcp_client:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        # Initialize conversation history
        messages = conversation_history or []
        messages.append({"role": "user", "content": user_message})
        
        # Get OpenAI function definitions
        openai_tools = self._format_tools_for_openai()
        
        # Initial OpenAI completion
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None
        )
        
        # Handle tool calls in a loop
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while response.choices[0].finish_reason == "tool_calls" and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            tool_calls = assistant_message.tool_calls
            
            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"\nCalling tool: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, indent=2)}")
                
                try:
                    # Call the FastMCP tool
                    result = await self.call_tool(tool_name, tool_args)
                    
                    # Extract text from result
                    if hasattr(result, 'content') and result.content:
                        tool_result = '\n'.join([
                            item.text if hasattr(item, 'text') else str(item)
                            for item in result.content
                        ])
                    else:
                        tool_result = str(result)
                    
                    print(f"   ✅ Result: {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")
                    
                except Exception as e:
                    tool_result = f"Error calling tool: {str(e)}"
                    print(f"   ❌ Error: {e}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Get next response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None
            )
        
        # Return final response
        final_response = response.choices[0].message.content
        return final_response, messages
    
    async def interactive_chat(self, model: str = "gpt-4o-mini"):
        """
        Start an interactive chat session
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        print("\n" + "="*70)
        print(f"💬 Interactive Chat with {model}")
        print("   Type your messages (or 'quit' to exit)")
        print("="*70 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                # Check for exit
                if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Send message and get response
                print(f"\n🤖 {model}: ", end="", flush=True)
                response, conversation_history = await self.chat(
                    user_input, 
                    model=model, 
                    conversation_history=conversation_history
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


async def main():
    """Main function to run the FastMCP + OpenAI client"""
    
    # Get configuration from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Validate API key
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("\n💡 Create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print("   MCP_SERVER_URL=http://localhost:8000/mcp  # optional")
        print("   OPENAI_MODEL=gpt-4o-mini  # optional")
        return
    
    # Create client
    client = FastMCPOpenAIClient(
        openai_api_key=OPENAI_API_KEY,
        server_url=SERVER_URL
    )
    
    try:
        # Connect to FastMCP server
        await client.connect()
        
        # Example: Single query
        print("\n" + "="*70)
        print("🧪 Testing with a single query")
        print("="*70)
        response, _ = await client.chat(
            "What tools are available on this server?",
            model=MODEL
        )
        print(f"\n🤖 Response: {response}")
        
        # Start interactive chat
        await client.interactive_chat(model=MODEL)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("🚀 FastMCP Client with OpenAI GPT-4o-mini\n")
    asyncio.run(main())
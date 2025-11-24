"""Test script to verify MCP server connection"""
import asyncio
import json
from mcp_use import MCPClient

async def test_connection():
    # Load configuration
    with open('airbnb_mcp.json', 'r') as f:
        config = json.load(f)
    
    print("Creating MCP client...")
    client = MCPClient.from_dict(config)
    
    print("Getting server names...")
    server_names = client.get_server_names()
    print(f"Available servers: {server_names}")
    
    if server_names:
        server_name = server_names[0]
        print(f"\nAttempting to create session for '{server_name}'...")
        
        try:
            # Try with a shorter timeout
            session = await asyncio.wait_for(
                client.create_session(server_name, auto_initialize=True),
                timeout=15
            )
            print(f"✅ Session created successfully!")
            
            # Try to list tools
            print("\nListing available tools...")
            tools = await asyncio.wait_for(
                session.list_tools(),
                timeout=10
            )
            print(f"Available tools: {len(tools)} tools found")
            for tool in tools[:5]:  # Show first 5
                print(f"  - {tool.name}: {tool.description[:60]}...")
            
            # Clean up
            await client.close_session(server_name)
            print("\n✅ Test completed successfully!")
            
        except asyncio.TimeoutError:
            print("❌ Timeout waiting for MCP server response")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())

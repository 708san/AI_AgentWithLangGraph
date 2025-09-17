import os
from langchain_mcp import MCPClient

mcp_endpoints = {
    "pcf": "hogehoge"
}

mcp_clients = {}

for name, url in mcp_endpoints.items():
    if url:
        mcp_clients[name] = MCPClient(url=url)
        print(f"MCP client for '{name}' initialized.")
    else:
        print(f"MCP server URL for '{name}' is not set.")
"""Round-trip sanity check for the GoodFoods MCP server.

Starts mcp_server/server.py as a subprocess, connects a real MCP
ClientSession to it over stdio, calls list_tools() and call_tool("ping"),
and prints both results. This is the thing to run before any booking
logic gets added — it proves the MCP plumbing itself works.

Usage:
    python tests/roundtrip_check.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.mcp_client import mcp_session


async def main() -> None:
    async with mcp_session() as session:
        tools = await session.list_tools()
        print("list_tools() ->")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")

        result = await session.call_tool("ping", {})
        print("\ncall_tool('ping') ->")
        for block in result.content:
            print(f"  {block.text}" if hasattr(block, "text") else f"  {block}")


if __name__ == "__main__":
    asyncio.run(main())

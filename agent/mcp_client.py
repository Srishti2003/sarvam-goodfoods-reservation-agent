"""Reusable stdio MCP client for talking to mcp_server/server.py.

Foundation only: wraps server startup + session handshake so other code
(the orchestrator, tests) doesn't repeat the stdio plumbing. No tool-specific
logic lives here.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"


def default_server_params() -> StdioServerParameters:
    """Params to launch mcp_server/server.py as a subprocess over stdio."""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )


@asynccontextmanager
async def mcp_session(server_params: StdioServerParameters | None = None):
    """Start the MCP server as a subprocess and yield a live ClientSession."""
    params = server_params or default_server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

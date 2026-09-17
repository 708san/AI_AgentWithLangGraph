"""Backward-compatible import location for the TogoMCP client."""

from .togomcp_client import TogoMCPClient, TogoMCPError

__all__ = ["TogoMCPClient", "TogoMCPError"]

"""
Benchmark scenario definitions.

Each scenario is a combination of server type, protocol, tool, concurrency level,
and duration. The runner iterates through all scenarios and collects metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ServerType(Enum):
    GRADIO = "gradio"
    FASTMCP = "fastmcp"


class Protocol(Enum):
    HTTP_API = "http_api"       # Direct REST API call
    MCP_SSE = "mcp_sse"         # MCP over SSE transport
    MCP_STREAMABLE = "mcp_streamable"  # MCP over Streamable HTTP


class ToolName(Enum):
    ECHO = "echo"
    FIBONACCI = "fibonacci"
    JSON_TRANSFORM = "json_transform"
    ASYNC_SLEEP = "async_sleep"
    PAYLOAD_ECHO = "payload_echo"


@dataclass
class Scenario:
    """A single benchmark scenario."""
    server: ServerType
    protocol: Protocol
    tool: ToolName
    virtual_users: int
    duration_seconds: int = 60
    warmup_seconds: int = 5
    concurrency_limit: int | None = 1  # Only relevant for Gradio
    tool_params: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        cl = "unlimited" if self.concurrency_limit is None else str(self.concurrency_limit)
        return (
            f"{self.server.value}__{self.protocol.value}__{self.tool.value}"
            f"__vu{self.virtual_users}__cl{cl}"
        )

    @property
    def display_name(self) -> str:
        cl = "inf" if self.concurrency_limit is None else str(self.concurrency_limit)
        return (
            f"{self.server.value.title()} | {self.protocol.value} | "
            f"{self.tool.value} | VU={self.virtual_users} | CL={cl}"
        )


def build_scenario_matrix(
    servers: list[str] | None = None,
    protocols: list[str] | None = None,
    tools: list[str] | None = None,
    vu_levels: list[int] | None = None,
    concurrency_limits: list[int | None] | None = None,
    duration: int = 60,
    warmup: int = 5,
    tool_params: dict[str, dict] | None = None,
) -> list[Scenario]:
    """Build the full scenario matrix from parameter sweeps.

    Returns a list of Scenario objects to execute.
    """
    from servers.config import (
        DEFAULT_VU_LEVELS,
        DEFAULT_GRADIO_CONCURRENCY_LIMITS,
        TOOL_PARAMS,
    )

    _servers = [ServerType(s) for s in servers] if servers else list(ServerType)
    _tools = [ToolName(t) for t in tools] if tools else list(ToolName)
    _vu_levels = vu_levels or DEFAULT_VU_LEVELS
    _concurrency_limits = concurrency_limits or DEFAULT_GRADIO_CONCURRENCY_LIMITS
    _tool_params = tool_params or TOOL_PARAMS

    # Determine protocols per server
    _protocol_map = {
        ServerType.GRADIO: [Protocol.HTTP_API, Protocol.MCP_STREAMABLE],
        ServerType.FASTMCP: [Protocol.HTTP_API, Protocol.MCP_STREAMABLE],
    }
    if protocols:
        _all_protocols = [Protocol(p) for p in protocols]
        _protocol_map = {s: _all_protocols for s in _servers}

    scenarios = []
    for server in _servers:
        for protocol in _protocol_map.get(server, []):
            for tool in _tools:
                params = _tool_params.get(tool.value, {})
                for vu in _vu_levels:
                    if server == ServerType.GRADIO:
                        # Sweep concurrency limits for Gradio
                        for cl in _concurrency_limits:
                            scenarios.append(
                                Scenario(
                                    server=server,
                                    protocol=protocol,
                                    tool=tool,
                                    virtual_users=vu,
                                    duration_seconds=duration,
                                    warmup_seconds=warmup,
                                    concurrency_limit=cl,
                                    tool_params=params,
                                )
                            )
                    else:
                        # FastMCP doesn't have a concurrency_limit knob
                        scenarios.append(
                            Scenario(
                                server=server,
                                protocol=protocol,
                                tool=tool,
                                virtual_users=vu,
                                duration_seconds=duration,
                                warmup_seconds=warmup,
                                concurrency_limit=None,
                                tool_params=params,
                            )
                        )

    return scenarios


def build_quick_scenarios(
    duration: int = 15,
    warmup: int = 3,
) -> list[Scenario]:
    """Build a minimal scenario set for smoke testing."""
    from servers.config import TOOL_PARAMS

    scenarios = []
    for server in [ServerType.GRADIO, ServerType.FASTMCP]:
        for protocol in [Protocol.HTTP_API]:
            for tool in [ToolName.ECHO, ToolName.FIBONACCI]:
                for vu in [1, 10]:
                    cl = 1 if server == ServerType.GRADIO else None
                    scenarios.append(
                        Scenario(
                            server=server,
                            protocol=protocol,
                            tool=tool,
                            virtual_users=vu,
                            duration_seconds=duration,
                            warmup_seconds=warmup,
                            concurrency_limit=cl,
                            tool_params=TOOL_PARAMS.get(tool.value, {}),
                        )
                    )
    return scenarios

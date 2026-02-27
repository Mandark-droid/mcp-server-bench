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
    queue_enabled: bool = True  # Gradio queue mode; irrelevant for FastMCP

    @property
    def scenario_id(self) -> str:
        cl = "unlimited" if self.concurrency_limit is None else str(self.concurrency_limit)
        sid = (
            f"{self.server.value}__{self.protocol.value}__{self.tool.value}"
            f"__vu{self.virtual_users}__cl{cl}"
        )
        if not self.queue_enabled:
            sid += "__noq"
        return sid

    @property
    def display_name(self) -> str:
        cl = "inf" if self.concurrency_limit is None else str(self.concurrency_limit)
        q = " | Q=off" if not self.queue_enabled else ""
        return (
            f"{self.server.value.title()} | {self.protocol.value} | "
            f"{self.tool.value} | VU={self.virtual_users} | CL={cl}{q}"
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
    queue_modes: list[bool] | None = None,
) -> list[Scenario]:
    """Build the full scenario matrix from parameter sweeps.

    Args:
        queue_modes: List of queue_enabled values for Gradio scenarios.
            Default [True, False] benchmarks both modes.
            FastMCP scenarios always use queue_enabled=True (irrelevant).

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
    _queue_modes = queue_modes if queue_modes is not None else [True, False]

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
                        # Sweep concurrency limits and queue modes for Gradio
                        for cl in _concurrency_limits:
                            for q in _queue_modes:
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
                                        queue_enabled=q,
                                    )
                                )
                    else:
                        # FastMCP doesn't have queue or concurrency_limit knobs
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
    """Build a minimal scenario set for smoke testing.

    Includes one queue=False variant for Gradio echo at VU=1 to
    spot-check the queue overhead difference.
    """
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

    # Add one queue=False Gradio variant for quick comparison
    scenarios.append(
        Scenario(
            server=ServerType.GRADIO,
            protocol=Protocol.HTTP_API,
            tool=ToolName.ECHO,
            virtual_users=1,
            duration_seconds=duration,
            warmup_seconds=warmup,
            concurrency_limit=1,
            tool_params=TOOL_PARAMS.get(ToolName.ECHO.value, {}),
            queue_enabled=False,
        )
    )
    return scenarios

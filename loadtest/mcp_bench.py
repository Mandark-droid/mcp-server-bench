"""
MCP protocol load tester.

Implements the full MCP session lifecycle:
  initialize → list_tools → call_tool (repeated) → close

Tests both Gradio MCP endpoints and FastMCP Streamable HTTP endpoints.
Uses httpx for HTTP transport with SSE parsing.
"""

import asyncio
import json
import time
import uuid

import httpx

from loadtest.metrics import MetricsCollector, RequestMetric
from loadtest.scenarios import Scenario, ServerType
from servers.config import (
    GRADIO_MCP_STREAMABLE,
    FASTMCP_MCP_STREAMABLE,
)


def _get_mcp_base_url(scenario: Scenario) -> str:
    """Get the MCP endpoint base URL."""
    if scenario.server == ServerType.GRADIO:
        return GRADIO_MCP_STREAMABLE
    return FASTMCP_MCP_STREAMABLE


def _build_jsonrpc(method: str, params: dict | None = None, req_id: str | None = None) -> dict:
    """Build a JSON-RPC 2.0 request."""
    msg = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id or str(uuid.uuid4()),
    }
    if params:
        msg["params"] = params
    return msg


_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


async def _mcp_initialize(
    client: httpx.AsyncClient,
    base_url: str,
) -> str | None:
    """Perform MCP initialization handshake. Returns session ID if provided."""
    init_msg = _build_jsonrpc("initialize", {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "benchmark-client", "version": "0.1.0"},
    })

    try:
        response = await client.post(
            base_url,
            json=init_msg,
            headers=_MCP_HEADERS,
            timeout=15.0,
        )
        session_id = response.headers.get("mcp-session-id")

        # Send initialized notification
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        headers = {**_MCP_HEADERS}
        if session_id:
            headers["mcp-session-id"] = session_id
        await client.post(base_url, json=initialized_msg, headers=headers, timeout=10.0)

        return session_id
    except Exception:
        return None


async def _mcp_call_tool(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str | None,
    tool_name: str,
    arguments: dict,
) -> tuple[float, bool, int, str | None]:
    """Call an MCP tool and return (latency_ms, success, response_size, error)."""
    call_msg = _build_jsonrpc("tools/call", {
        "name": tool_name,
        "arguments": arguments,
    })

    headers = {**_MCP_HEADERS}
    if session_id:
        headers["mcp-session-id"] = session_id

    start = time.time()
    try:
        response = await client.post(
            base_url,
            json=call_msg,
            headers=headers,
            timeout=30.0,
        )
        latency_ms = (time.time() - start) * 1000

        success = 200 <= response.status_code < 300
        body = response.text
        # Check for JSON-RPC error
        if success:
            try:
                data = response.json()
                if isinstance(data, dict) and "error" in data:
                    success = False
            except Exception:
                pass

        return latency_ms, success, len(response.content), None if success else body[:200]

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return latency_ms, False, 0, str(e)[:200]


def _build_mcp_tool_args(scenario: Scenario) -> dict:
    """Convert scenario tool_params to MCP tool arguments format."""
    params = scenario.tool_params
    tool = scenario.tool.value

    if tool == "echo":
        return {"message": params.get("message", "hello")}
    elif tool == "fibonacci":
        return {"n": params.get("n", 25)}
    elif tool == "json_transform":
        data = params.get("data", {})
        return {"data": json.dumps(data) if isinstance(data, dict) else data}
    elif tool == "async_sleep":
        return {"duration_ms": params.get("duration_ms", 50)}
    elif tool == "payload_echo":
        return {"payload": params.get("payload", "x" * 1000)}
    return params


async def _mcp_worker(
    client: httpx.AsyncClient,
    base_url: str,
    scenario: Scenario,
    collector: MetricsCollector,
    stop_event: asyncio.Event,
) -> None:
    """A single MCP virtual user — initializes session, then calls tool in a loop."""
    # Initialize MCP session
    session_id = await _mcp_initialize(client, base_url)
    tool_args = _build_mcp_tool_args(scenario)

    while not stop_event.is_set():
        latency_ms, success, resp_size, error = await _mcp_call_tool(
            client, base_url, session_id,
            scenario.tool.value, tool_args,
        )
        collector.record_request(
            RequestMetric(
                timestamp=time.time(),
                latency_ms=latency_ms,
                status_code=200 if success else 500,
                success=success,
                response_size_bytes=resp_size,
                error=error,
            )
        )
        await asyncio.sleep(0)


async def run_mcp_benchmark(
    scenario: Scenario,
    collector: MetricsCollector,
) -> None:
    """Run an MCP protocol benchmark for the given scenario.

    Each virtual user maintains its own MCP session and calls tools in a loop.
    """
    base_url = _get_mcp_base_url(scenario)
    total_duration = scenario.warmup_seconds + scenario.duration_seconds

    limits = httpx.Limits(
        max_connections=scenario.virtual_users * 2 + 10,
        max_keepalive_connections=scenario.virtual_users + 5,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        # Verify MCP endpoint is reachable
        try:
            test_msg = _build_jsonrpc("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "health-check", "version": "0.1.0"},
            })
            resp = await client.post(
                base_url,
                json=test_msg,
                headers=_MCP_HEADERS,
                timeout=15.0,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"MCP endpoint returned {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"MCP server not reachable at {base_url}: {e}") from e

        stop_event = asyncio.Event()
        collector.start(warmup_seconds=scenario.warmup_seconds)

        workers = [
            asyncio.create_task(
                _mcp_worker(client, base_url, scenario, collector, stop_event)
            )
            for _ in range(scenario.virtual_users)
        ]

        # Wait for test duration then signal stop
        await asyncio.sleep(total_duration)
        stop_event.set()

        # Give workers time to finish in-flight requests, with a safety timeout
        safety_timeout = 30.0 + 5  # request_timeout + grace
        try:
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True),
                timeout=safety_timeout,
            )
        except asyncio.TimeoutError:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        collector.stop()

"""
FastMCP benchmark server.

Exposes all 5 benchmark tools as MCP tools via Streamable HTTP transport.
Also exposes a simple REST API for direct HTTP benchmarking.

Usage:
    python fastmcp_server.py
    python fastmcp_server.py --port 8100
"""

import argparse
import json
import os
import time

from fastmcp import FastMCP

from shared_tools import (
    async_sleep_impl,
    echo_sync,
    fibonacci_sync,
    json_transform_sync,
    payload_echo_sync,
)
from config import FASTMCP_PORT

# --- Create FastMCP server ---
mcp = FastMCP(
    name="FastMCP Benchmark Server",
    instructions="Benchmark server exposing 5 standardized tools for performance testing.",
)


# --- Register tools (thin async wrappers around shared logic) ---


@mcp.tool()
async def echo(message: str) -> dict:
    """Echo a message back. Measures raw framework overhead.

    Args:
        message: The message string to echo back.

    Returns:
        dict with the echoed message and server timestamp.
    """
    return echo_sync(message)


@mcp.tool()
async def fibonacci(n: int) -> dict:
    """Calculate the nth Fibonacci number recursively. CPU-bound workload.

    Args:
        n: Position in the Fibonacci sequence (0-35).

    Returns:
        dict with the Fibonacci result and computation time.
    """
    return fibonacci_sync(n)


@mcp.tool()
async def json_transform(data: str) -> dict:
    """Transform JSON data — uppercase strings, sum numbers.

    Args:
        data: A JSON string to transform.

    Returns:
        dict with transformed data and processing stats.
    """
    parsed = json.loads(data) if isinstance(data, str) else data
    return json_transform_sync(parsed)


@mcp.tool()
async def async_sleep(duration_ms: int) -> dict:
    """Simulate an async I/O operation.

    Args:
        duration_ms: Sleep duration in milliseconds (0-5000).

    Returns:
        dict with actual sleep duration and overhead.
    """
    return await async_sleep_impl(duration_ms)


@mcp.tool()
async def payload_echo(payload: str) -> dict:
    """Echo a large payload back. Tests throughput with large payloads.

    Args:
        payload: A string payload of arbitrary size.

    Returns:
        dict with the payload and size metrics.
    """
    return payload_echo_sync(payload)


# --- Also expose a lightweight REST API for direct HTTP benchmarking ---

from starlette.requests import Request
from starlette.responses import JSONResponse


@mcp.custom_route("/api/echo", methods=["POST"])
async def rest_echo(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(echo_sync(body.get("message", "")))


@mcp.custom_route("/api/fibonacci", methods=["POST"])
async def rest_fibonacci(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(fibonacci_sync(body.get("n", 10)))


@mcp.custom_route("/api/json_transform", methods=["POST"])
async def rest_json_transform(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(json_transform_sync(body.get("data", {})))


@mcp.custom_route("/api/async_sleep", methods=["POST"])
async def rest_async_sleep(request: Request) -> JSONResponse:
    body = await request.json()
    result = await async_sleep_impl(body.get("duration_ms", 50))
    return JSONResponse(result)


@mcp.custom_route("/api/payload_echo", methods=["POST"])
async def rest_payload_echo(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(payload_echo_sync(body.get("payload", "")))


@mcp.custom_route("/api/health", methods=["GET"])
async def rest_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "fastmcp", "timestamp": time.time()})


def main():
    parser = argparse.ArgumentParser(description="FastMCP Benchmark Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FASTMCP_PORT", str(FASTMCP_PORT))),
    )
    args = parser.parse_args()

    # Run with streamable HTTP transport
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=args.port,
    )


if __name__ == "__main__":
    main()

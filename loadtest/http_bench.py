"""
HTTP/REST API load tester.

Sends concurrent requests to REST API endpoints using async httpx.
Supports both Gradio API format (queue-based SSE) and FastMCP REST endpoints.
"""

import asyncio
import json
import time

import httpx

from loadtest.metrics import MetricsCollector, RequestMetric
from loadtest.scenarios import Protocol, Scenario, ServerType
from servers.config import (
    GRADIO_API_BASE,
    FASTMCP_BASE,
)


def _build_url(scenario: Scenario) -> str:
    """Build the request URL based on server type and tool."""
    if scenario.server == ServerType.GRADIO:
        # Gradio 6.x queue API: /gradio_api/call/{api_name}
        return f"{GRADIO_API_BASE}/gradio_api/call/{scenario.tool.value}"
    else:
        # FastMCP REST endpoint format: /api/{tool_name}
        return f"{FASTMCP_BASE}/api/{scenario.tool.value}"


def _build_payload(scenario: Scenario) -> dict:
    """Build the request payload based on server type."""
    params = scenario.tool_params

    if scenario.server == ServerType.GRADIO:
        # Gradio expects {"data": [arg1, arg2, ...]}
        if scenario.tool.value == "echo":
            return {"data": [params.get("message", "hello")]}
        elif scenario.tool.value == "fibonacci":
            return {"data": [params.get("n", 25)]}
        elif scenario.tool.value == "json_transform":
            return {"data": [json.dumps(params.get("data", {}))]}
        elif scenario.tool.value == "async_sleep":
            return {"data": [params.get("duration_ms", 50)]}
        elif scenario.tool.value == "payload_echo":
            return {"data": [params.get("payload", "x" * 1000)]}
        return {"data": list(params.values())}
    else:
        # FastMCP REST expects the params directly
        return params


async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    collector: MetricsCollector,
    is_gradio: bool = False,
) -> None:
    """Execute a single HTTP request and record metrics."""
    start = time.time()
    try:
        # Step 1: POST the request
        response = await client.post(
            url,
            json=payload,
            timeout=30.0,
        )

        if is_gradio and 200 <= response.status_code < 300:
            # Step 2: For Gradio, fetch the result via SSE
            event_data = response.json()
            event_id = event_data.get("event_id")
            if event_id:
                result_url = f"{url}/{event_id}"
                result_resp = await client.get(result_url, timeout=30.0)
                latency_ms = (time.time() - start) * 1000
                # Check if the SSE response contains a successful completion
                body = result_resp.text
                success = "event: complete" in body
                collector.record_request(
                    RequestMetric(
                        timestamp=start,
                        latency_ms=latency_ms,
                        status_code=result_resp.status_code if success else 500,
                        success=success,
                        response_size_bytes=len(result_resp.content),
                        error=None if success else body[:200],
                    )
                )
                return

        latency_ms = (time.time() - start) * 1000
        collector.record_request(
            RequestMetric(
                timestamp=start,
                latency_ms=latency_ms,
                status_code=response.status_code,
                success=200 <= response.status_code < 300,
                response_size_bytes=len(response.content),
                error=None if response.status_code < 400 else response.text[:200],
            )
        )
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        collector.record_request(
            RequestMetric(
                timestamp=start,
                latency_ms=latency_ms,
                status_code=0,
                success=False,
                response_size_bytes=0,
                error=str(e)[:200],
            )
        )


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    collector: MetricsCollector,
    stop_event: asyncio.Event,
    is_gradio: bool = False,
) -> None:
    """A single virtual user -- sends requests in a loop until stopped."""
    while not stop_event.is_set():
        await _single_request(client, url, payload, collector, is_gradio)
        # Tiny yield to prevent event loop starvation
        await asyncio.sleep(0)


async def run_http_benchmark(
    scenario: Scenario,
    collector: MetricsCollector,
) -> None:
    """Run an HTTP API benchmark for the given scenario.

    Spawns `virtual_users` concurrent workers that hammer the endpoint
    for `duration_seconds` (plus warmup).
    """
    url = _build_url(scenario)
    payload = _build_payload(scenario)
    total_duration = scenario.warmup_seconds + scenario.duration_seconds
    is_gradio = scenario.server == ServerType.GRADIO

    # Connection pool sized to virtual users
    limits = httpx.Limits(
        max_connections=scenario.virtual_users + 10,
        max_keepalive_connections=scenario.virtual_users + 5,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        # Verify server is reachable
        try:
            health_url = (
                f"{GRADIO_API_BASE}/"
                if is_gradio
                else f"{FASTMCP_BASE}/api/health"
            )
            await client.get(health_url, timeout=10.0)
        except Exception as e:
            raise RuntimeError(
                f"Server not reachable at {health_url}: {e}"
            ) from e

        stop_event = asyncio.Event()
        collector.start(warmup_seconds=scenario.warmup_seconds)

        # Spawn virtual users
        workers = [
            asyncio.create_task(
                _worker(client, url, payload, collector, stop_event, is_gradio)
            )
            for _ in range(scenario.virtual_users)
        ]

        # Wait for test duration then signal stop
        await asyncio.sleep(total_duration)
        stop_event.set()

        # Give workers time to finish in-flight requests
        await asyncio.sleep(1.0)
        for w in workers:
            w.cancel()

        # Suppress cancellation errors
        await asyncio.gather(*workers, return_exceptions=True)
        collector.stop()

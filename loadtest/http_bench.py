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
import servers.config as cfg


def _build_url(scenario: Scenario) -> str:
    """Build the request URL based on server type and tool."""
    if scenario.server == ServerType.GRADIO:
        # Gradio 6.x queue API: /gradio_api/call/{api_name}
        return f"{cfg.GRADIO_API_BASE}/gradio_api/call/{scenario.tool.value}"
    else:
        # FastMCP REST endpoint format: /api/{tool_name}
        return f"{cfg.FASTMCP_BASE}/api/{scenario.tool.value}"


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
    request_timeout: float = 30.0,
) -> None:
    """Execute a single HTTP request and record metrics."""
    start = time.time()
    try:
        # Step 1: POST the request
        response = await client.post(
            url,
            json=payload,
            timeout=request_timeout,
        )

        if is_gradio and 200 <= response.status_code < 300:
            # Step 2: For Gradio, fetch the result via SSE
            event_data = response.json()
            event_id = event_data.get("event_id")
            if event_id:
                result_url = f"{url}/{event_id}"
                result_resp = await client.get(result_url, timeout=request_timeout)
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
    deadline: float,
    is_gradio: bool = False,
    request_timeout: float = 30.0,
) -> None:
    """A single virtual user -- sends requests in a loop until deadline."""
    while time.time() < deadline:
        await _single_request(client, url, payload, collector, is_gradio, request_timeout)
        # Yield to allow other tasks (metrics collection, etc.) to run
        await asyncio.sleep(0.001)


async def run_http_benchmark(
    scenario: Scenario,
    collector: MetricsCollector,
    request_timeout: float = 30.0,
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
                f"{cfg.GRADIO_API_BASE}/"
                if is_gradio
                else f"{cfg.FASTMCP_BASE}/api/health"
            )
            await client.get(health_url, timeout=10.0)
        except Exception as e:
            raise RuntimeError(
                f"Server not reachable at {health_url}: {e}"
            ) from e

        collector.start(warmup_seconds=scenario.warmup_seconds)
        deadline = time.time() + total_duration

        # Spawn virtual users — each worker checks the deadline itself
        workers = [
            asyncio.create_task(
                _worker(client, url, payload, collector, deadline, is_gradio, request_timeout)
            )
            for _ in range(scenario.virtual_users)
        ]

        # Wait for all workers to finish (they self-terminate at deadline)
        # Add a safety timeout so we never hang forever
        safety_timeout = total_duration + request_timeout + 10
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

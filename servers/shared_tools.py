"""
Shared tool implementations for the benchmark suite.

CRITICAL: Both Gradio and FastMCP servers import these functions directly.
This guarantees byte-identical computation — the benchmark measures ONLY
framework overhead, not implementation differences.
"""

import asyncio
import json
import time


def echo_sync(message: str) -> dict:
    """Echo a message back. Measures raw framework overhead.

    Args:
        message: The message string to echo back.

    Returns:
        dict with the echoed message and server timestamp.
    """
    return {
        "message": message,
        "timestamp_ns": time.time_ns(),
    }


def fibonacci_sync(n: int) -> dict:
    """Calculate the nth Fibonacci number recursively. CPU-bound workload.

    Args:
        n: Position in the Fibonacci sequence (0-35).

    Returns:
        dict with the Fibonacci result and computation time.
    """
    n = min(max(n, 0), 35)  # Clamp to safe range

    def _fib(k: int) -> int:
        if k <= 1:
            return k
        return _fib(k - 1) + _fib(k - 2)

    start = time.perf_counter_ns()
    result = _fib(n)
    elapsed_ns = time.perf_counter_ns() - start

    return {
        "n": n,
        "result": result,
        "compute_time_ns": elapsed_ns,
    }


def json_transform_sync(data: dict) -> dict:
    """Transform JSON data — uppercase all string values, sum all numeric values.

    Tests JSON parsing, traversal, and serialization overhead.

    Args:
        data: A JSON-serializable dictionary to transform.

    Returns:
        dict with transformed data and processing stats.
    """
    string_count = 0
    number_count = 0
    number_sum = 0.0

    def _transform(obj):
        nonlocal string_count, number_count, number_sum
        if isinstance(obj, dict):
            return {k: _transform(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_transform(item) for item in obj]
        elif isinstance(obj, str):
            string_count += 1
            return obj.upper()
        elif isinstance(obj, (int, float)):
            number_count += 1
            number_sum += obj
            return obj
        return obj

    start = time.perf_counter_ns()
    transformed = _transform(data)
    elapsed_ns = time.perf_counter_ns() - start

    # Round-trip through JSON to measure serialization
    json_bytes = json.dumps(transformed).encode("utf-8")

    return {
        "transformed": transformed,
        "stats": {
            "strings_transformed": string_count,
            "numbers_found": number_count,
            "number_sum": number_sum,
            "json_size_bytes": len(json_bytes),
            "processing_time_ns": elapsed_ns,
        },
    }


async def async_sleep_impl(duration_ms: int) -> dict:
    """Simulate an async I/O operation (e.g., database query, API call).

    Tests the framework's ability to handle concurrent async operations
    without blocking the event loop.

    Args:
        duration_ms: How long to sleep in milliseconds (0-5000).

    Returns:
        dict with actual sleep duration and scheduling overhead.
    """
    duration_ms = min(max(duration_ms, 0), 5000)
    target_s = duration_ms / 1000.0

    start = time.perf_counter()
    await asyncio.sleep(target_s)
    actual_s = time.perf_counter() - start

    return {
        "requested_ms": duration_ms,
        "actual_ms": round(actual_s * 1000, 3),
        "overhead_ms": round((actual_s - target_s) * 1000, 3),
    }


def payload_echo_sync(payload: str) -> dict:
    """Echo a large payload back. Tests throughput with varying payload sizes.

    Args:
        payload: A string payload of arbitrary size.

    Returns:
        dict with the payload and size metrics.
    """
    payload_bytes = len(payload.encode("utf-8"))
    return {
        "payload": payload,
        "size_bytes": payload_bytes,
        "timestamp_ns": time.time_ns(),
    }

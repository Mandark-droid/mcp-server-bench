"""
Gradio benchmark server.

Exposes all 5 benchmark tools as both Gradio API endpoints and MCP tools.
Concurrency limit is configurable via CLI argument or environment variable.

Usage:
    python gradio_server.py                          # default concurrency_limit=1
    python gradio_server.py --concurrency-limit 10   # custom limit
    python gradio_server.py --concurrency-limit none  # unlimited
    GRADIO_CONCURRENCY_LIMIT=5 python gradio_server.py
"""

import argparse
import os
import sys

import gradio as gr

from shared_tools import (
    async_sleep_impl,
    echo_sync,
    fibonacci_sync,
    json_transform_sync,
    payload_echo_sync,
)
from config import GRADIO_PORT


def parse_args() -> tuple[int | None, bool]:
    """Parse CLI args for concurrency limit and queue mode.

    Returns:
        (concurrency_limit, no_queue) tuple.
    """
    parser = argparse.ArgumentParser(description="Gradio Benchmark Server")
    parser.add_argument(
        "--concurrency-limit",
        type=str,
        default=os.environ.get("GRADIO_CONCURRENCY_LIMIT", "1"),
        help="Concurrency limit per event (integer or 'none' for unlimited)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_PORT", str(GRADIO_PORT))),
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        default=False,
        help="Disable Gradio queue (skip progress notifications overhead)",
    )
    args = parser.parse_args()

    if args.port:
        global _server_port
        _server_port = args.port

    val = args.concurrency_limit.strip().lower()
    concurrency_limit = None if val == "none" else int(val)
    return concurrency_limit, args.no_queue


_server_port = GRADIO_PORT


# --- Tool wrappers (thin wrappers to match Gradio's sync/async expectations) ---


def echo(message: str) -> dict:
    """Echo a message back. Measures raw framework overhead.

    Args:
        message: The message string to echo back.

    Returns:
        dict with the echoed message and server timestamp.
    """
    return echo_sync(message)


def fibonacci(n: int) -> dict:
    """Calculate the nth Fibonacci number recursively. CPU-bound workload.

    Args:
        n: Position in the Fibonacci sequence (0-35).

    Returns:
        dict with the Fibonacci result and computation time.
    """
    return fibonacci_sync(n)


def json_transform(data: str) -> dict:
    """Transform JSON data — uppercase strings, sum numbers.

    Args:
        data: A JSON string to transform.

    Returns:
        dict with transformed data and processing stats.
    """
    import json as _json

    parsed = _json.loads(data) if isinstance(data, str) else data
    return json_transform_sync(parsed)


async def async_sleep(duration_ms: int) -> dict:
    """Simulate an async I/O operation.

    Args:
        duration_ms: Sleep duration in milliseconds (0-5000).

    Returns:
        dict with actual sleep duration and overhead.
    """
    return await async_sleep_impl(duration_ms)


def payload_echo(payload: str) -> dict:
    """Echo a large payload back. Tests throughput with large payloads.

    Args:
        payload: A string payload of arbitrary size.

    Returns:
        dict with the payload and size metrics.
    """
    return payload_echo_sync(payload)


def build_app(concurrency_limit: int | None, no_queue: bool = False) -> gr.Blocks:
    """Build the Gradio app with all benchmark tools."""

    queue_label = "off" if no_queue else "on"
    with gr.Blocks(title="Gradio Benchmark Server") as demo:
        gr.Markdown("# 🔬 Gradio Benchmark Server")
        gr.Markdown(
            f"Concurrency limit: **{concurrency_limit if concurrency_limit is not None else 'unlimited'}** | "
            f"Queue: **{queue_label}**"
        )

        # --- Echo ---
        with gr.Row():
            with gr.Column():
                echo_input = gr.Textbox(label="Message", value="hello benchmark")
            with gr.Column():
                echo_output = gr.JSON(label="Echo Result")
        echo_btn = gr.Button("Echo")
        echo_btn.click(
            echo,
            inputs=[echo_input],
            outputs=[echo_output],
            api_name="echo",
            concurrency_limit=concurrency_limit,
            **({} if not no_queue else {"queue": False}),
        )

        # --- Fibonacci ---
        with gr.Row():
            with gr.Column():
                fib_input = gr.Number(label="N", value=25, precision=0)
            with gr.Column():
                fib_output = gr.JSON(label="Fibonacci Result")
        fib_btn = gr.Button("Fibonacci")
        fib_btn.click(
            fibonacci,
            inputs=[fib_input],
            outputs=[fib_output],
            api_name="fibonacci",
            concurrency_limit=concurrency_limit,
            **({} if not no_queue else {"queue": False}),
        )

        # --- JSON Transform ---
        with gr.Row():
            with gr.Column():
                json_input = gr.Textbox(
                    label="JSON Data",
                    value='{"users": [{"name": "alice", "age": 30}]}',
                    lines=3,
                )
            with gr.Column():
                json_output = gr.JSON(label="Transform Result")
        json_btn = gr.Button("JSON Transform")
        json_btn.click(
            json_transform,
            inputs=[json_input],
            outputs=[json_output],
            api_name="json_transform",
            concurrency_limit=concurrency_limit,
            **({} if not no_queue else {"queue": False}),
        )

        # --- Async Sleep ---
        with gr.Row():
            with gr.Column():
                sleep_input = gr.Number(label="Duration (ms)", value=50, precision=0)
            with gr.Column():
                sleep_output = gr.JSON(label="Sleep Result")
        sleep_btn = gr.Button("Async Sleep")
        sleep_btn.click(
            async_sleep,
            inputs=[sleep_input],
            outputs=[sleep_output],
            api_name="async_sleep",
            concurrency_limit=concurrency_limit,
            **({} if not no_queue else {"queue": False}),
        )

        # --- Payload Echo ---
        with gr.Row():
            with gr.Column():
                payload_input = gr.Textbox(
                    label="Payload", value="x" * 1000, lines=2
                )
            with gr.Column():
                payload_output = gr.JSON(label="Payload Result")
        payload_btn = gr.Button("Payload Echo")
        payload_btn.click(
            payload_echo,
            inputs=[payload_input],
            outputs=[payload_output],
            api_name="payload_echo",
            concurrency_limit=concurrency_limit,
            **({} if not no_queue else {"queue": False}),
        )

    return demo


def main():
    concurrency_limit, no_queue = parse_args()
    demo = build_app(concurrency_limit, no_queue=no_queue)
    if not no_queue:
        demo.queue(default_concurrency_limit=concurrency_limit or 40)
    demo.launch(
        server_name="0.0.0.0",
        server_port=_server_port,
        mcp_server=True,
        quiet=True,
    )


if __name__ == "__main__":
    main()

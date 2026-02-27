"""
Server configuration constants shared across the benchmark suite.
"""

# --- Server Ports ---
GRADIO_PORT = 7860
FASTMCP_PORT = 8100

# --- Server URLs ---
GRADIO_API_BASE = f"http://127.0.0.1:{GRADIO_PORT}"
GRADIO_MCP_SSE = f"{GRADIO_API_BASE}/gradio_api/mcp/sse"
GRADIO_MCP_STREAMABLE = f"{GRADIO_API_BASE}/gradio_api/mcp/"
GRADIO_API_PREDICT = f"{GRADIO_API_BASE}/api/{{api_name}}"

FASTMCP_BASE = f"http://127.0.0.1:{FASTMCP_PORT}"
FASTMCP_MCP_SSE = f"{FASTMCP_BASE}/sse"
FASTMCP_MCP_STREAMABLE = f"{FASTMCP_BASE}/mcp/"

# --- Benchmark Defaults ---
DEFAULT_WARMUP_SECONDS = 5
DEFAULT_TEST_DURATION_SECONDS = 60
DEFAULT_VU_LEVELS = [1, 10, 25, 50]
DEFAULT_GRADIO_CONCURRENCY_LIMITS = [1, 5, 10, None]

# --- Tool Parameters for Benchmarking ---
TOOL_PARAMS = {
    "echo": {"message": "hello benchmark"},
    "fibonacci": {"n": 25},
    "json_transform": {
        "data": {
            "users": [
                {"name": "alice", "age": 30, "city": "new york"},
                {"name": "bob", "age": 25, "city": "san francisco"},
                {"name": "charlie", "age": 35, "city": "london"},
                {"name": "diana", "age": 28, "city": "tokyo"},
                {"name": "eve", "age": 32, "city": "paris"},
            ]
        }
    },
    "async_sleep": {"duration_ms": 50},
    "payload_echo": {"payload": "x" * 10_000},  # 10KB payload
}

# --- System Metrics Sampling ---
SYSTEM_METRICS_INTERVAL_SECONDS = 1.0

# --- Results ---
RESULTS_DIR = "results"
HF_DATASET_REPO = "kshitijthakkar/mcp-server-bench"

# 🔬 mcp-server-bench

**A structured, reproducible benchmark comparing MCP server implementations (Gradio, FastMCP, and more) under identical workloads.**

> Part of the [TraceMind](https://huggingface.co/kshitijthakkar) AI Observability Ecosystem

---

## Why This Exists

There are **zero published benchmarks** for Gradio's performance as an API/MCP server. The community relies on vague claims ("scales to thousands") with no empirical data. Meanwhile, FastMCP (the reference Python MCP implementation) has only been benchmarked in one study (TM Dev Lab, Feb 2026) showing ~292 RPS on a single worker.

This suite provides apples-to-apples comparison by:
- Implementing **identical tools** with **identical logic** on both Gradio and FastMCP
- Testing both as **REST API servers** and as **MCP servers (SSE/Streamable HTTP)**
- Sweeping across **concurrency configurations** to find scaling characteristics
- Collecting **latency percentiles, throughput, error rates, memory, and CPU** metrics
- Generating **automated comparison reports** with statistical analysis

---

## Architecture

```
mcp-server-bench/
│
├── servers/                    # Server implementations
│   ├── shared_tools.py         # Shared tool logic (imported by both servers)
│   ├── gradio_server.py        # Gradio server (API + MCP mode)
│   ├── fastmcp_server.py       # FastMCP server (Streamable HTTP)
│   └── config.py               # Server configuration constants
│
├── loadtest/                   # Load testing framework
│   ├── runner.py               # Orchestrator — launches servers, runs tests, collects results
│   ├── http_bench.py           # HTTP/REST API load tester (async httpx)
│   ├── mcp_bench.py            # MCP protocol load tester (SSE client)
│   ├── scenarios.py            # Test scenario definitions
│   └── metrics.py              # Metrics collection and aggregation
│
├── analysis/                   # Result analysis
│   ├── analyzer.py             # Statistical analysis and comparison
│   ├── report_generator.py     # Markdown/HTML report generator
│   └── plots.py                # Visualization (matplotlib/plotly)
│
├── results/                    # Output directory (gitignored except samples)
│   └── .gitkeep
│
├── pyproject.toml              # Project config and dependencies
├── run_benchmark.py            # CLI entry point
└── README.md
```

---

## Benchmark Tools (Identical on Both Servers)

| Tool | Category | What It Tests | Parameters |
|------|----------|---------------|------------|
| `echo` | Baseline | Raw framework overhead, serialization | `message: str` |
| `fibonacci` | CPU-bound | Compute under load, GIL contention | `n: int (0-35)` |
| `json_transform` | Data processing | JSON parse/serialize, string ops | `data: dict` |
| `async_sleep` | I/O-bound | Async concurrency, event loop efficiency | `duration_ms: int` |
| `payload_echo` | Throughput | Large payload handling | `payload: str (variable size)` |

---

## Test Matrix

The benchmark sweeps across these dimensions:

| Dimension | Values |
|-----------|--------|
| **Server** | Gradio API, Gradio MCP, FastMCP API, FastMCP MCP |
| **Concurrency (VUs)** | 1, 10, 25, 50, 100 |
| **Gradio concurrency_limit** | 1 (default), 5, 10, None |
| **Tool** | echo, fibonacci, json_transform, async_sleep, payload_echo |
| **Duration** | 60s per scenario (configurable) |

Total scenarios: ~200 combinations (automated)

---

## Metrics Collected

**Per-request:**
- Response time (ms)
- HTTP status code
- Payload size (bytes)

**Aggregated per scenario:**
- Throughput (requests/second)
- Latency: p50, p75, p90, p95, p99, max
- Error rate (%)
- Success count / Total count

**System-level (sampled every 1s):**
- CPU usage (%)
- Memory RSS (MB)
- Open file descriptors
- Thread count

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/kshitijthakkar/mcp-server-bench
cd mcp-server-bench
pip install -e ".[dev]"

# Run full benchmark suite
python run_benchmark.py --full

# Run quick smoke test (1 tool, 2 concurrency levels, 15s each)
python run_benchmark.py --quick

# Run specific comparison
python run_benchmark.py --servers gradio,fastmcp --tools echo,fibonacci --vus 10,50

# Generate report from existing results
python run_benchmark.py --analyze results/2026-02-27_run_001/
```

---

## Output

The benchmark produces:
1. **Raw JSON results** — per-request timings for full reproducibility
2. **Summary CSV** — one row per scenario with aggregated metrics
3. **Comparison report** (Markdown + HTML) — side-by-side analysis with charts
4. **HuggingFace Dataset** — auto-push to `kshitijthakkar/mcp-server-bench`

---

## Key Design Decisions

1. **No external load test tools (no k6/locust)** — Pure Python async httpx client gives us precise control over MCP protocol semantics and avoids measuring tool overhead.

2. **Shared tool logic** — Both servers import from `shared_tools.py` ensuring byte-identical computation. The benchmark measures only framework overhead.

3. **Process isolation** — Each server runs in a subprocess. The load tester runs in the main process. No resource contention between server and client.

4. **Warmup phase** — 5s warmup before measurement begins. Accounts for JIT-like effects in Python (import caching, connection pool warmup).

5. **MCP testing uses proper protocol** — Not just HTTP POSTs to the SSE endpoint. We implement the full MCP session lifecycle: initialize → list_tools → call_tool → close.

---

## Contributing

PRs welcome. Key areas:
- Additional server implementations (e.g., raw FastAPI, LitServe, Go MCP SDK, Node.js MCP SDK)
- Docker-based isolation for stricter resource control
- CI/CD integration for regression tracking
- HuggingFace Spaces hosted benchmark runner

---

## License

MIT

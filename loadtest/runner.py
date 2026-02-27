"""
Benchmark orchestrator.

Manages the full lifecycle:
  1. Launch server subprocess
  2. Wait for server readiness
  3. Start system metrics collection
  4. Run load test (HTTP or MCP)
  5. Stop system metrics collection
  6. Aggregate results
  7. Shutdown server
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from loadtest.http_bench import run_http_benchmark
from loadtest.mcp_bench import run_mcp_benchmark
from loadtest.metrics import MetricsCollector, ScenarioResult, SystemSample
from loadtest.scenarios import Protocol, Scenario, ServerType
import servers.config as cfg

console = Console()


class ServerProcess:
    """Manages a server subprocess."""

    def __init__(self, server_type: ServerType, concurrency_limit: int | None = None):
        self.server_type = server_type
        self.concurrency_limit = concurrency_limit
        self.process: subprocess.Popen | None = None
        self.pid: int | None = None

    def start(self) -> None:
        """Start the server subprocess."""
        servers_dir = Path(__file__).parent.parent / "servers"

        if self.server_type == ServerType.GRADIO:
            cl_arg = "none" if self.concurrency_limit is None else str(self.concurrency_limit)
            cmd = [
                sys.executable,
                str(servers_dir / "gradio_server.py"),
                "--concurrency-limit", cl_arg,
                "--port", str(cfg.GRADIO_PORT),
            ]
        else:
            cmd = [
                sys.executable,
                str(servers_dir / "fastmcp_server.py"),
                "--port", str(cfg.FASTMCP_PORT),
            ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(servers_dir),
            env=env,
        )
        self.pid = self.process.pid

    async def wait_ready(self, timeout: float = 30.0) -> bool:
        """Wait for the server to be ready to accept requests."""
        if self.server_type == ServerType.GRADIO:
            url = f"{cfg.GRADIO_API_BASE}/"
        else:
            url = f"{cfg.FASTMCP_BASE}/api/health"

        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code < 500:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        return False

    def stop(self) -> None:
        """Stop the server subprocess gracefully."""
        if self.process:
            try:
                # Try graceful shutdown first
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

    def get_psutil_process(self) -> psutil.Process | None:
        """Get the psutil Process object for system metrics."""
        if self.pid:
            try:
                return psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return None
        return None


async def _collect_system_metrics(
    server_proc: ServerProcess,
    collector: MetricsCollector,
    stop_event: asyncio.Event,
) -> None:
    """Collect system metrics at regular intervals."""
    proc = server_proc.get_psutil_process()
    if not proc:
        return

    while not stop_event.is_set():
        try:
            with proc.oneshot():
                cpu = proc.cpu_percent()
                mem = proc.memory_info().rss / (1024 * 1024)  # MB
                threads = proc.num_threads()
                try:
                    fds = proc.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    fds = 0

            collector.record_system_sample(
                SystemSample(
                    timestamp=time.time(),
                    cpu_percent=cpu,
                    memory_rss_mb=mem,
                    thread_count=threads,
                    open_fds=fds,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        await asyncio.sleep(cfg.SYSTEM_METRICS_INTERVAL_SECONDS)


async def run_scenario(
    scenario: Scenario,
    request_timeout: float = 30.0,
    server_timeout: float = 30.0,
) -> ScenarioResult:
    """Execute a single benchmark scenario end-to-end."""
    console.print(f"\n[bold cyan]> {scenario.display_name}[/]")

    # 1. Start server
    server = ServerProcess(scenario.server, scenario.concurrency_limit)
    server.start()
    console.print(f"  Server PID: {server.pid}")

    try:
        # 2. Wait for readiness
        ready = await server.wait_ready(timeout=server_timeout)
        if not ready:
            console.print("[red]  X Server failed to start[/]")
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_display=scenario.display_name,
                server=scenario.server.value,
                protocol=scenario.protocol.value,
                tool=scenario.tool.value,
                virtual_users=scenario.virtual_users,
                concurrency_limit=scenario.concurrency_limit,
                duration_seconds=0,
                errors=["Server failed to start within timeout"],
            )
        console.print("  [green]OK Server ready[/]")

        # 3. Set up metrics collection
        collector = MetricsCollector()
        sys_stop = asyncio.Event()
        sys_task = asyncio.create_task(
            _collect_system_metrics(server, collector, sys_stop)
        )

        # 4. Run the appropriate benchmark
        console.print(
            f"  Running: {scenario.virtual_users} VUs x "
            f"{scenario.duration_seconds}s (+{scenario.warmup_seconds}s warmup)"
        )

        if scenario.protocol == Protocol.HTTP_API:
            await run_http_benchmark(
                scenario, collector, request_timeout=request_timeout,
            )
        else:
            await run_mcp_benchmark(scenario, collector)

        # 5. Stop system metrics collection
        sys_stop.set()
        await sys_task

        # 6. Aggregate results
        result = collector.aggregate(scenario)
        console.print(
            f"  [green]OK[/] {result.total_requests} requests | "
            f"{result.throughput_rps:.1f} RPS | "
            f"p50={result.latency_p50:.1f}ms | "
            f"p99={result.latency_p99:.1f}ms | "
            f"err={result.error_rate*100:.1f}%"
        )
        return result

    finally:
        # 7. Shutdown server
        server.stop()
        console.print("  Server stopped")
        # Small gap between scenarios to let ports release
        await asyncio.sleep(2.0)


def save_results(
    results: list[ScenarioResult],
    run_dir: Path,
    quiet: bool = False,
) -> None:
    """Save results to JSON and CSV files.

    Args:
        quiet: If True, skip console output (used for incremental saves).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    import pandas as pd
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    csv_path = run_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    # Detailed JSON (includes raw latencies)
    json_path = run_dir / "detailed_results.json"
    detailed = []
    for r in results:
        d = r.to_dict()
        d["raw_latencies_sample"] = r.raw_latencies[:1000]  # Cap at 1000 for file size
        d["errors_sample"] = r.errors[:50]
        detailed.append(d)

    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "scenarios": detailed,
            },
            f,
            indent=2,
            default=str,
        )

    if not quiet:
        console.print(f"\n[green]Summary saved to {csv_path}[/]")
        console.print(f"[green]Detailed results saved to {json_path}[/]")
        print_summary_table(results)


def print_summary_table(results: list[ScenarioResult]) -> None:
    """Print a rich summary table to the console."""
    table = Table(
        title="Benchmark Results Summary",
        show_lines=True,
    )
    table.add_column("Server", style="cyan")
    table.add_column("Protocol", style="blue")
    table.add_column("Tool", style="green")
    table.add_column("VUs", justify="right")
    table.add_column("CL", justify="right")
    table.add_column("RPS", justify="right", style="bold")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("Err%", justify="right")

    for r in results:
        cl_str = "inf" if r.concurrency_limit is None else str(r.concurrency_limit)
        err_style = "red" if r.error_rate > 0.01 else "green"
        table.add_row(
            r.server,
            r.protocol,
            r.tool,
            str(r.virtual_users),
            cl_str,
            f"{r.throughput_rps:.1f}",
            f"{r.latency_p50:.1f}",
            f"{r.latency_p95:.1f}",
            f"{r.latency_p99:.1f}",
            f"[{err_style}]{r.error_rate*100:.1f}%[/]",
        )

    console.print(table)


async def run_all_scenarios(
    scenarios: list[Scenario],
    run_dir: Path,
    request_timeout: float = 30.0,
    server_timeout: float = 30.0,
) -> list[ScenarioResult]:
    """Run all scenarios sequentially and collect results.

    Results are saved to disk after each scenario so that completed
    work is never lost if the run is interrupted.
    """
    results = []
    total = len(scenarios)

    console.print(f"\n[bold]Running {total} benchmark scenarios[/]\n")

    # Group by server type to minimize server restarts
    sorted_scenarios = sorted(
        scenarios,
        key=lambda s: (s.server.value, s.concurrency_limit or 999, s.protocol.value),
    )

    for i, scenario in enumerate(sorted_scenarios, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Scenario {i}/{total}[/]")

        result = await run_scenario(
            scenario,
            request_timeout=request_timeout,
            server_timeout=server_timeout,
        )
        results.append(result)

        # Save after every scenario so results survive interruption
        save_results(results, run_dir, quiet=True)

    return results

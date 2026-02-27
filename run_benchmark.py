#!/usr/bin/env python3
"""
mcp-server-bench -- CLI Entry Point

Usage:
    python run_benchmark.py --full
    python run_benchmark.py --quick
    python run_benchmark.py --servers gradio,fastmcp --tools echo,fibonacci --vus 10,50
    python run_benchmark.py --analyze results/2026-02-27_run_001/
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
# --- Presets ---
@click.option("--full", is_flag=True, help="Run full benchmark matrix (~200 scenarios)")
@click.option("--quick", is_flag=True, help="Quick smoke test (~8 scenarios, 15s each)")
# --- Scenario selection ---
@click.option(
    "--servers",
    type=str,
    default=None,
    help="Comma-separated server list: gradio,fastmcp",
)
@click.option(
    "--tools",
    type=str,
    default=None,
    help="Comma-separated tool list: echo,fibonacci,json_transform,async_sleep,payload_echo",
)
@click.option(
    "--vus",
    type=str,
    default=None,
    help="Comma-separated VU levels: 1,10,50",
)
@click.option(
    "--cls",
    type=str,
    default=None,
    help="Comma-separated Gradio concurrency_limits: 1,5,10,none",
)
@click.option(
    "--protocols",
    type=str,
    default=None,
    help="Comma-separated protocols: http_api,mcp_streamable",
)
@click.option(
    "--queue-modes",
    type=str,
    default=None,
    help="Comma-separated Gradio queue modes: true,false (default: both)",
)
# --- Load generation ---
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Test duration per scenario in seconds (default: 60, quick: 15)",
)
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Warmup period per scenario in seconds (default: 5, quick: 3)",
)
@click.option(
    "--request-timeout",
    type=float,
    default=None,
    help="Per-request HTTP timeout in seconds (default: 30)",
)
@click.option(
    "--server-timeout",
    type=float,
    default=None,
    help="Server startup readiness timeout in seconds (default: 30)",
)
# --- Server ports ---
@click.option(
    "--gradio-port",
    type=int,
    default=None,
    help="Port for Gradio server (default: 7860)",
)
@click.option(
    "--fastmcp-port",
    type=int,
    default=None,
    help="Port for FastMCP server (default: 8100)",
)
# --- Metrics & output ---
@click.option(
    "--metrics-interval",
    type=float,
    default=None,
    help="System metrics sampling interval in seconds (default: 1.0)",
)
@click.option(
    "--results-dir",
    type=str,
    default=None,
    help="Base directory for results (default: results/)",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output subdirectory name (default: auto-generated timestamp)",
)
# --- Post-run ---
@click.option(
    "--analyze",
    type=str,
    default=None,
    help="Path to existing results directory to analyze (skip running benchmarks)",
)
@click.option(
    "--push-hf",
    is_flag=True,
    help="Push results to HuggingFace dataset after completion",
)
def main(
    full, quick, servers, tools, vus, cls, protocols, queue_modes,
    duration, warmup, request_timeout, server_timeout,
    gradio_port, fastmcp_port,
    metrics_interval, results_dir, output,
    analyze, push_hf,
):
    """mcp-server-bench -- MCP Server Benchmark Suite"""

    console.print("\n[bold magenta]mcp-server-bench[/]\n")

    # --- Apply port overrides before anything imports config ---
    import servers.config as cfg

    if gradio_port is not None:
        cfg.GRADIO_PORT = gradio_port
        cfg.GRADIO_API_BASE = f"http://127.0.0.1:{gradio_port}"
        cfg.GRADIO_MCP_SSE = f"{cfg.GRADIO_API_BASE}/gradio_api/mcp/sse"
        cfg.GRADIO_MCP_STREAMABLE = f"{cfg.GRADIO_API_BASE}/gradio_api/mcp/"
        cfg.GRADIO_API_PREDICT = f"{cfg.GRADIO_API_BASE}/api/{{api_name}}"
    if fastmcp_port is not None:
        cfg.FASTMCP_PORT = fastmcp_port
        cfg.FASTMCP_BASE = f"http://127.0.0.1:{fastmcp_port}"
        cfg.FASTMCP_MCP_SSE = f"{cfg.FASTMCP_BASE}/sse"
        cfg.FASTMCP_MCP_STREAMABLE = f"{cfg.FASTMCP_BASE}/mcp/"
    if metrics_interval is not None:
        cfg.SYSTEM_METRICS_INTERVAL_SECONDS = metrics_interval
    if results_dir is not None:
        cfg.RESULTS_DIR = results_dir

    # --- Analyze existing results ---
    if analyze:
        _analyze_results(Path(analyze))
        return

    # --- Resolve defaults based on preset ---
    if quick:
        duration = duration if duration is not None else 15
        warmup = warmup if warmup is not None else 3
    else:
        duration = duration if duration is not None else 60
        warmup = warmup if warmup is not None else 5

    request_timeout = request_timeout if request_timeout is not None else 30.0
    server_timeout = server_timeout if server_timeout is not None else 30.0

    # --- Build scenario matrix ---
    from loadtest.scenarios import build_scenario_matrix, build_quick_scenarios

    if quick:
        scenarios = build_quick_scenarios(
            duration=duration,
            warmup=warmup,
        )
    elif full:
        scenarios = build_scenario_matrix(
            duration=duration,
            warmup=warmup,
        )
    else:
        _servers = servers.split(",") if servers else None
        _tools = tools.split(",") if tools else ["echo", "fibonacci"]
        _vus = [int(v) for v in vus.split(",")] if vus else [1, 10, 50]
        _protocols = protocols.split(",") if protocols else None
        _cls = None
        if cls:
            _cls = [None if c.strip().lower() == "none" else int(c) for c in cls.split(",")]
        _queue_modes = None
        if queue_modes:
            _queue_modes = [v.strip().lower() == "true" for v in queue_modes.split(",")]

        scenarios = build_scenario_matrix(
            servers=_servers,
            tools=_tools,
            vu_levels=_vus,
            concurrency_limits=_cls,
            protocols=_protocols,
            duration=duration,
            warmup=warmup,
            queue_modes=_queue_modes,
        )

    console.print(f"[bold]Scenarios to run: {len(scenarios)}[/]")

    # Estimate total time
    total_time_est = sum(s.duration_seconds + s.warmup_seconds + 3 for s in scenarios)
    console.print(
        f"[dim]Estimated time: {total_time_est // 60}m {total_time_est % 60}s[/]"
    )
    console.print(
        f"[dim]Duration: {duration}s | Warmup: {warmup}s | "
        f"Request timeout: {request_timeout}s | Server timeout: {server_timeout}s[/]"
    )

    # --- Prepare output directory ---
    base_dir = cfg.RESULTS_DIR
    if output:
        run_dir = Path(base_dir) / output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{timestamp}"

    console.print(f"[dim]Results will be saved to: {run_dir}[/]\n")

    # --- Run benchmarks ---
    from loadtest.runner import run_all_scenarios, save_results

    results = asyncio.run(run_all_scenarios(
        scenarios,
        run_dir=run_dir,
        request_timeout=request_timeout,
        server_timeout=server_timeout,
    ))
    # Final save with console output (incremental saves are quiet)
    save_results(results, run_dir)

    # --- Generate analysis ---
    _analyze_results(run_dir)

    # --- Push to HuggingFace ---
    if push_hf:
        _push_to_hf(run_dir)

    console.print("\n[bold green]Benchmark complete![/]\n")


def _analyze_results(run_dir: Path) -> None:
    """Run analysis on completed benchmark results."""
    from analysis.analyzer import load_results, compare_servers, generate_markdown_report
    from analysis.plots import generate_all_plots

    console.print(f"\n[bold]Analyzing results from {run_dir}[/]\n")

    try:
        df = load_results(run_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/]")
        return

    # Generate comparison
    comparisons = compare_servers(df)

    # Generate report
    report_path = run_dir / "REPORT.md"
    report = generate_markdown_report(df, comparisons, report_path)
    console.print(f"[green]Report generated: {report_path}[/]")

    # Generate plots
    plots_dir = run_dir / "plots"
    plot_paths = generate_all_plots(df, plots_dir)
    console.print(f"[green]Generated {len(plot_paths)} charts in {plots_dir}[/]")

    # Print winners summary
    winners = comparisons.get("winners", {})
    if winners:
        console.print("\n[bold]Winners by Tool:[/]")
        for tool, w in winners.items():
            winner = w["throughput_winner"].title()
            marker = "[yellow]*[/]" if winner == "Gradio" else "[green]*[/]"
            console.print(
                f"  {marker} {tool}: [bold]{winner}[/] "
                f"({w['speedup']}x faster, "
                f"{w['gradio_best_rps']} vs {w['fastmcp_best_rps']} RPS)"
            )


def _push_to_hf(run_dir: Path) -> None:
    """Push results to HuggingFace datasets with full report and charts."""
    from servers.config import HF_DATASET_REPO

    try:
        from datasets import Dataset
        from huggingface_hub import HfApi, CommitOperationAdd
        import pandas as pd

        # --- Phase 1: Push CSV data (creates parquet files) ---
        csv_path = run_dir / "summary.csv"
        if not csv_path.exists():
            console.print("[yellow]No summary.csv found, skipping HF push[/]")
            return

        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)
        ds.push_to_hub(HF_DATASET_REPO, private=False)
        console.print(f"[green]Pushed dataset to HuggingFace: {HF_DATASET_REPO}[/]")

        # --- Phase 2: Upload report, plots, and detailed results ---
        operations = []

        # Build README.md from REPORT.md
        report_path = run_dir / "REPORT.md"
        if report_path.exists():
            report_content = report_path.read_text(encoding="utf-8")

            yaml_front_matter = (
                "---\n"
                "language:\n"
                "- en\n"
                "license: mit\n"
                "tags:\n"
                "- mcp\n"
                "- benchmark\n"
                "- gradio\n"
                "- fastmcp\n"
                "- api-performance\n"
                "pretty_name: MCP Server Benchmark Results\n"
                "size_categories:\n"
                "- n<1K\n"
                "---\n\n"
            )

            # Build charts section to insert before ## Methodology
            plots_dir = run_dir / "plots"
            chart_names = [
                ("throughput_comparison.png", "Throughput Comparison"),
                ("latency_comparison.png", "Latency Comparison"),
                ("gradio_cl_scaling.png", "Gradio Concurrency Limit Scaling"),
                ("protocol_overhead.png", "Protocol Overhead"),
            ]
            chart_lines = ["## Benchmark Charts\n"]
            for fname, alt in chart_names:
                if (plots_dir / fname).exists():
                    chart_lines.append(f"![{alt}](plots/{fname})\n")
            chart_section = "\n".join(chart_lines) + "\n"

            # Insert charts before ## Methodology
            if "## Methodology" in report_content:
                report_content = report_content.replace(
                    "## Methodology",
                    chart_section + "## Methodology",
                )
            else:
                # Append at end if Methodology section not found
                report_content += "\n" + chart_section

            footer = (
                "\n---\n\n"
                "*Benchmarks generated by "
                "[mcp-server-bench](https://github.com/Mandark-droid/mcp-server-bench)*\n"
            )
            readme_content = yaml_front_matter + report_content + footer

            operations.append(CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=readme_content.encode("utf-8"),
            ))
        else:
            console.print("[yellow]No REPORT.md found, skipping dataset card[/]")

        # Add plot images
        plots_dir = run_dir / "plots"
        if plots_dir.exists():
            for png in sorted(plots_dir.glob("*.png")):
                operations.append(CommitOperationAdd(
                    path_in_repo=f"plots/{png.name}",
                    path_or_fileobj=str(png),
                ))

        # Add detailed results JSON
        details_path = run_dir / "detailed_results.json"
        if details_path.exists():
            operations.append(CommitOperationAdd(
                path_in_repo="detailed_results.json",
                path_or_fileobj=str(details_path),
            ))

        if operations:
            api = HfApi()
            api.create_commit(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                operations=operations,
                commit_message="Update benchmark report, charts, and detailed results",
            )
            console.print(
                f"[green]Uploaded report + {len(operations) - 1} assets "
                f"to {HF_DATASET_REPO}[/]"
            )

    except Exception as e:
        console.print(f"[yellow]Failed to push to HF: {e}[/]")


if __name__ == "__main__":
    main()

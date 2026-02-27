#!/usr/bin/env python3
"""
mcp-server-bench — CLI Entry Point

Usage:
    python run_benchmark.py --full                   # Full sweep (~200 scenarios)
    python run_benchmark.py --quick                  # Quick smoke test (~8 scenarios)
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
@click.option("--full", is_flag=True, help="Run full benchmark matrix (~200 scenarios)")
@click.option("--quick", is_flag=True, help="Quick smoke test (~8 scenarios, 15s each)")
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
    "--duration",
    type=int,
    default=60,
    help="Test duration per scenario in seconds (default: 60)",
)
@click.option(
    "--analyze",
    type=str,
    default=None,
    help="Path to existing results directory to analyze (skip running benchmarks)",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output directory name (default: auto-generated timestamp)",
)
@click.option(
    "--push-hf",
    is_flag=True,
    help="Push results to HuggingFace dataset after completion",
)
def main(
    full, quick, servers, tools, vus, cls, protocols,
    duration, analyze, output, push_hf,
):
    """🔬 mcp-server-bench — MCP Server Benchmark Suite"""

    console.print("\n[bold magenta]🔬 mcp-server-bench[/]\n")

    # --- Analyze existing results ---
    if analyze:
        _analyze_results(Path(analyze))
        return

    # --- Build scenario matrix ---
    from loadtest.scenarios import build_scenario_matrix, build_quick_scenarios

    if quick:
        scenarios = build_quick_scenarios()
        duration = 15
    elif full:
        scenarios = build_scenario_matrix(duration=duration)
    else:
        # Custom configuration
        _servers = servers.split(",") if servers else None
        _tools = tools.split(",") if tools else ["echo", "fibonacci"]
        _vus = [int(v) for v in vus.split(",")] if vus else [1, 10, 50]
        _protocols = protocols.split(",") if protocols else None
        _cls = None
        if cls:
            _cls = [None if c.strip().lower() == "none" else int(c) for c in cls.split(",")]

        scenarios = build_scenario_matrix(
            servers=_servers,
            tools=_tools,
            vu_levels=_vus,
            concurrency_limits=_cls,
            protocols=_protocols,
            duration=duration,
        )

    console.print(f"[bold]Scenarios to run: {len(scenarios)}[/]")

    # Estimate total time
    total_time_est = sum(s.duration_seconds + s.warmup_seconds + 3 for s in scenarios)
    console.print(
        f"[dim]Estimated time: {total_time_est // 60}m {total_time_est % 60}s[/]"
    )

    # --- Prepare output directory ---
    from servers.config import RESULTS_DIR

    if output:
        run_dir = Path(RESULTS_DIR) / output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path(RESULTS_DIR) / f"run_{timestamp}"

    console.print(f"[dim]Results will be saved to: {run_dir}[/]\n")

    # --- Run benchmarks ---
    from loadtest.runner import run_all_scenarios, save_results

    results = asyncio.run(run_all_scenarios(scenarios))
    save_results(results, run_dir)

    # --- Generate analysis ---
    _analyze_results(run_dir)

    # --- Push to HuggingFace ---
    if push_hf:
        _push_to_hf(run_dir)

    console.print("\n[bold green]✅ Benchmark complete![/]\n")


def _analyze_results(run_dir: Path) -> None:
    """Run analysis on completed benchmark results."""
    from analysis.analyzer import load_results, compare_servers, generate_markdown_report
    from analysis.plots import generate_all_plots

    console.print(f"\n[bold]📊 Analyzing results from {run_dir}[/]\n")

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
        console.print("\n[bold]🏆 Winners by Tool:[/]")
        for tool, w in winners.items():
            winner = w["throughput_winner"].title()
            emoji = "🟠" if winner == "Gradio" else "🟢"
            console.print(
                f"  {emoji} {tool}: [bold]{winner}[/] "
                f"({w['speedup']}x faster, "
                f"{w['gradio_best_rps']} vs {w['fastmcp_best_rps']} RPS)"
            )


def _push_to_hf(run_dir: Path) -> None:
    """Push results to HuggingFace datasets."""
    from servers.config import HF_DATASET_REPO

    try:
        from datasets import Dataset
        import pandas as pd

        df = pd.read_csv(run_dir / "summary.csv")
        ds = Dataset.from_pandas(df)
        ds.push_to_hub(HF_DATASET_REPO, private=False)
        console.print(f"[green]Pushed to HuggingFace: {HF_DATASET_REPO}[/]")
    except Exception as e:
        console.print(f"[yellow]Failed to push to HF: {e}[/]")


if __name__ == "__main__":
    main()

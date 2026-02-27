"""
Statistical analysis and comparison of benchmark results.

Produces side-by-side comparisons between Gradio and FastMCP
across all dimensions: throughput, latency, resource usage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(run_dir: str | Path) -> pd.DataFrame:
    """Load benchmark results from a run directory."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    json_path = run_dir / "detailed_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return pd.DataFrame(data["scenarios"])

    raise FileNotFoundError(f"No results found in {run_dir}")


def compare_servers(df: pd.DataFrame) -> dict:
    """Generate head-to-head comparison between Gradio and FastMCP.

    Returns a dict with comparison metrics organized by dimension.
    """
    comparisons = {}

    # --- Throughput comparison by tool and VU level ---
    throughput = (
        df.groupby(["server", "protocol", "tool", "virtual_users"])
        .agg(
            rps_mean=("throughput_rps", "mean"),
            rps_std=("throughput_rps", "std"),
        )
        .reset_index()
    )
    comparisons["throughput"] = throughput.to_dict("records")

    # --- Latency comparison ---
    latency = (
        df.groupby(["server", "protocol", "tool", "virtual_users"])
        .agg(
            p50_mean=("latency_p50_ms", "mean"),
            p95_mean=("latency_p95_ms", "mean"),
            p99_mean=("latency_p99_ms", "mean"),
            mean_mean=("latency_mean_ms", "mean"),
        )
        .reset_index()
    )
    comparisons["latency"] = latency.to_dict("records")

    # --- Gradio concurrency_limit scaling ---
    gradio_df = df[df["server"] == "gradio"]
    if not gradio_df.empty:
        cl_scaling = (
            gradio_df.groupby(["concurrency_limit", "tool", "virtual_users"])
            .agg(
                rps_mean=("throughput_rps", "mean"),
                p50_mean=("latency_p50_ms", "mean"),
                p99_mean=("latency_p99_ms", "mean"),
            )
            .reset_index()
        )
        comparisons["gradio_cl_scaling"] = cl_scaling.to_dict("records")

    # --- Protocol comparison (HTTP API vs MCP) ---
    protocol_cmp = (
        df.groupby(["server", "protocol", "tool"])
        .agg(
            rps_mean=("throughput_rps", "mean"),
            p50_mean=("latency_p50_ms", "mean"),
            overhead_p50=("latency_p50_ms", "mean"),
        )
        .reset_index()
    )
    comparisons["protocol_overhead"] = protocol_cmp.to_dict("records")

    # --- Resource usage ---
    if "avg_cpu_pct" in df.columns:
        resources = (
            df.groupby(["server", "virtual_users"])
            .agg(
                avg_cpu=("avg_cpu_pct", "mean"),
                avg_mem=("avg_memory_mb", "mean"),
                peak_mem=("peak_memory_mb", "max"),
            )
            .reset_index()
        )
        comparisons["resources"] = resources.to_dict("records")

    # --- Summary winners ---
    winners = _compute_winners(df)
    comparisons["winners"] = winners

    return comparisons


def _compute_winners(df: pd.DataFrame) -> dict:
    """Determine which server wins in each category."""
    winners = {}

    # Group by matching scenarios (same tool, VU, protocol)
    # Compare Gradio (best CL config) vs FastMCP
    tools = df["tool"].unique()

    for tool in tools:
        tool_df = df[df["tool"] == tool]

        # Best Gradio config (highest RPS across CL values)
        gradio_best = (
            tool_df[tool_df["server"] == "gradio"]
            .sort_values("throughput_rps", ascending=False)
            .head(1)
        )
        fastmcp_best = (
            tool_df[tool_df["server"] == "fastmcp"]
            .sort_values("throughput_rps", ascending=False)
            .head(1)
        )

        if not gradio_best.empty and not fastmcp_best.empty:
            g_rps = gradio_best["throughput_rps"].iloc[0]
            f_rps = fastmcp_best["throughput_rps"].iloc[0]

            winners[tool] = {
                "throughput_winner": "fastmcp" if f_rps > g_rps else "gradio",
                "gradio_best_rps": round(g_rps, 1),
                "fastmcp_best_rps": round(f_rps, 1),
                "speedup": round(max(g_rps, f_rps) / max(min(g_rps, f_rps), 0.1), 2),
                "gradio_best_cl": (
                    gradio_best["concurrency_limit"].iloc[0]
                    if "concurrency_limit" in gradio_best.columns
                    else None
                ),
            }

    return winners


def generate_markdown_report(
    df: pd.DataFrame,
    comparisons: dict,
    output_path: Path,
) -> str:
    """Generate a comprehensive Markdown comparison report."""
    lines = []
    lines.append("# 🔬 Gradio vs FastMCP Benchmark Report\n")
    lines.append(f"**Generated:** {pd.Timestamp.now().isoformat()}\n")
    lines.append(f"**Total scenarios:** {len(df)}\n")

    # --- Executive Summary ---
    lines.append("## Executive Summary\n")
    winners = comparisons.get("winners", {})
    if winners:
        for tool, w in winners.items():
            winner = w["throughput_winner"].title()
            lines.append(
                f"- **{tool}**: {winner} wins "
                f"({w['gradio_best_rps']} vs {w['fastmcp_best_rps']} RPS, "
                f"{w['speedup']}x difference)"
            )
            if w.get("gradio_best_cl") is not None:
                cl = "unlimited" if w["gradio_best_cl"] is None else w["gradio_best_cl"]
                lines.append(f"  - Gradio best config: concurrency_limit={cl}")
        lines.append("")

    # --- Throughput Comparison Table ---
    lines.append("## Throughput (Requests/Second)\n")
    pivot = df.pivot_table(
        values="throughput_rps",
        index=["tool", "virtual_users"],
        columns=["server", "protocol"],
        aggfunc="max",
    )
    lines.append(pivot.to_markdown())
    lines.append("")

    # --- Latency Comparison ---
    lines.append("## Latency p50 (ms)\n")
    pivot_lat = df.pivot_table(
        values="latency_p50_ms",
        index=["tool", "virtual_users"],
        columns=["server", "protocol"],
        aggfunc="min",  # Best (lowest) latency
    )
    lines.append(pivot_lat.to_markdown())
    lines.append("")

    # --- Gradio CL Scaling ---
    if "gradio_cl_scaling" in comparisons:
        lines.append("## Gradio concurrency_limit Scaling\n")
        lines.append(
            "How does Gradio's throughput change as concurrency_limit increases?\n"
        )
        gradio_only = df[df["server"] == "gradio"]
        if not gradio_only.empty:
            cl_pivot = gradio_only.pivot_table(
                values="throughput_rps",
                index=["tool", "virtual_users"],
                columns="concurrency_limit",
                aggfunc="mean",
            )
            lines.append(cl_pivot.to_markdown())
        lines.append("")

    # --- Protocol Overhead ---
    lines.append("## Protocol Overhead: HTTP API vs MCP\n")
    lines.append(
        "Comparing latency of the same tool called via REST API vs MCP protocol:\n"
    )
    proto_pivot = df.pivot_table(
        values="latency_p50_ms",
        index=["server", "tool"],
        columns="protocol",
        aggfunc="mean",
    )
    lines.append(proto_pivot.to_markdown())
    lines.append("")

    # --- Error Rates ---
    lines.append("## Error Rates\n")
    errors = df[df["error_rate_pct"] > 0]
    if errors.empty:
        lines.append("All scenarios completed with 0% error rate. ✅\n")
    else:
        lines.append(errors[["scenario_id", "error_rate_pct"]].to_markdown(index=False))
    lines.append("")

    # --- Resource Usage ---
    if "avg_cpu_pct" in df.columns:
        lines.append("## Resource Usage\n")
        res_pivot = df.pivot_table(
            values=["avg_cpu_pct", "avg_memory_mb", "peak_memory_mb"],
            index="server",
            aggfunc=["mean", "max"],
        )
        lines.append(res_pivot.to_markdown())
        lines.append("")

    # --- Methodology ---
    lines.append("## Methodology\n")
    lines.append(
        "- Both servers use identical tool implementations (imported from shared_tools.py)\n"
        "- Each scenario runs in an isolated server subprocess\n"
        "- Warmup period excluded from measurements\n"
        "- Load generated by async httpx workers (not external tools)\n"
        "- MCP tests use full protocol lifecycle (initialize → call_tool)\n"
        "- System metrics sampled every 1s via psutil\n"
    )

    report = "\n".join(lines)
    output_path.write_text(report)
    return report

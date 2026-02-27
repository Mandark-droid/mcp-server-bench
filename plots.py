"""
Visualization module for benchmark results.

Generates charts comparing Gradio vs FastMCP performance
across throughput, latency, and resource dimensions.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# --- Style Configuration ---
COLORS = {
    "gradio": "#FF6B35",      # Gradio orange
    "fastmcp": "#4ECDC4",     # Teal
    "http_api": "#2196F3",    # Blue
    "mcp_streamable": "#9C27B0",  # Purple
}

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "figure.dpi": 150,
    "font.size": 10,
})


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart: Throughput (RPS) by server, tool, and VU level."""
    tools = df["tool"].unique()
    fig, axes = plt.subplots(1, len(tools), figsize=(6 * len(tools), 5), squeeze=False)
    fig.suptitle("Throughput Comparison (Requests/Second)", fontsize=14, fontweight="bold")

    for idx, tool in enumerate(tools):
        ax = axes[0][idx]
        tool_df = df[df["tool"] == tool]

        # Get best config per server per VU level
        for server in ["gradio", "fastmcp"]:
            s_df = tool_df[tool_df["server"] == server]
            if s_df.empty:
                continue
            best = s_df.groupby("virtual_users")["throughput_rps"].max().reset_index()
            ax.plot(
                best["virtual_users"],
                best["throughput_rps"],
                marker="o",
                label=server.title(),
                color=COLORS[server],
                linewidth=2,
                markersize=6,
            )

        ax.set_title(tool, fontsize=11)
        ax.set_xlabel("Virtual Users")
        ax.set_ylabel("RPS")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    path = output_dir / "throughput_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latency_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: Latency percentiles (p50, p95, p99) by server."""
    servers = df["server"].unique()
    tools = df["tool"].unique()
    vu_target = df["virtual_users"].median()  # Pick median VU level

    # Filter to closest VU level
    target_df = df[df["virtual_users"] == df["virtual_users"].unique()[len(df["virtual_users"].unique())//2]]
    if target_df.empty:
        target_df = df

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Latency Percentiles @ VU={int(vu_target)}", fontsize=14, fontweight="bold")

    x = np.arange(len(tools))
    width = 0.12
    percentiles = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]
    pct_labels = ["p50", "p95", "p99"]

    offset = 0
    for server in servers:
        s_df = target_df[target_df["server"] == server]
        if s_df.empty:
            continue
        for j, (pct, label) in enumerate(zip(percentiles, pct_labels)):
            values = []
            for tool in tools:
                t_df = s_df[s_df["tool"] == tool]
                values.append(t_df[pct].min() if not t_df.empty else 0)

            bars = ax.bar(
                x + offset * width,
                values,
                width,
                label=f"{server.title()} {label}",
                color=COLORS.get(server, "#888"),
                alpha=0.5 + 0.2 * j,
            )
            offset += 1

    ax.set_xticks(x + width * (offset - 1) / 2)
    ax.set_xticklabels(tools)
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = output_dir / "latency_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gradio_cl_scaling(df: pd.DataFrame, output_dir: Path) -> Path:
    """Line chart: How Gradio throughput scales with concurrency_limit."""
    gradio_df = df[df["server"] == "gradio"]
    if gradio_df.empty:
        return output_dir / "gradio_cl_scaling.png"

    tools = gradio_df["tool"].unique()
    fig, axes = plt.subplots(1, len(tools), figsize=(6 * len(tools), 5), squeeze=False)
    fig.suptitle("Gradio: concurrency_limit Scaling", fontsize=14, fontweight="bold")

    for idx, tool in enumerate(tools):
        ax = axes[0][idx]
        tool_df = gradio_df[gradio_df["tool"] == tool]

        for vu in sorted(tool_df["virtual_users"].unique()):
            vu_df = tool_df[tool_df["virtual_users"] == vu].sort_values("concurrency_limit")
            cl_values = vu_df["concurrency_limit"].fillna(999).values
            cl_labels = ["∞" if v >= 999 else str(int(v)) for v in cl_values]

            ax.plot(
                range(len(cl_values)),
                vu_df["throughput_rps"].values,
                marker="s",
                label=f"VU={vu}",
                linewidth=2,
            )

        ax.set_title(tool)
        ax.set_xlabel("concurrency_limit")
        ax.set_ylabel("RPS")
        # Set x labels
        cl_unique = sorted(tool_df["concurrency_limit"].fillna(999).unique())
        ax.set_xticks(range(len(cl_unique)))
        ax.set_xticklabels(["∞" if v >= 999 else str(int(v)) for v in cl_unique])
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    path = output_dir / "gradio_cl_scaling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_protocol_overhead(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart: HTTP API vs MCP protocol latency overhead."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Protocol Overhead: HTTP API vs MCP", fontsize=14, fontweight="bold")

    protocols = df["protocol"].unique()
    tools = df["tool"].unique()
    servers = df["server"].unique()

    x = np.arange(len(tools))
    width = 0.2
    offset = 0

    for server in servers:
        for protocol in protocols:
            sp_df = df[(df["server"] == server) & (df["protocol"] == protocol)]
            if sp_df.empty:
                continue
            values = []
            for tool in tools:
                t_df = sp_df[sp_df["tool"] == tool]
                values.append(t_df["latency_p50_ms"].mean() if not t_df.empty else 0)

            color = COLORS.get(server, "#888")
            alpha = 0.6 if "mcp" in protocol else 1.0
            ax.bar(
                x + offset * width,
                values,
                width,
                label=f"{server.title()} ({protocol})",
                color=color,
                alpha=alpha,
            )
            offset += 1

    ax.set_xticks(x + width * (offset - 1) / 2)
    ax.set_xticklabels(tools)
    ax.set_ylabel("p50 Latency (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = output_dir / "protocol_overhead.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate all benchmark visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    plots.append(plot_throughput_comparison(df, output_dir))
    plots.append(plot_latency_comparison(df, output_dir))
    plots.append(plot_gradio_cl_scaling(df, output_dir))
    plots.append(plot_protocol_overhead(df, output_dir))

    return plots

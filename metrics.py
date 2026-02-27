"""
Metrics collection and aggregation.

Collects per-request timings and system-level metrics,
then aggregates into summary statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RequestMetric:
    """A single request measurement."""
    timestamp: float           # Unix timestamp when request was initiated
    latency_ms: float          # End-to-end latency in milliseconds
    status_code: int           # HTTP status code (200, 500, etc.)
    success: bool              # Whether the request was successful
    response_size_bytes: int   # Size of the response body
    error: str | None = None   # Error message if failed


@dataclass
class SystemSample:
    """A single system metrics sample."""
    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    thread_count: int
    open_fds: int


@dataclass
class ScenarioResult:
    """Aggregated results for a single benchmark scenario."""
    scenario_id: str
    scenario_display: str
    server: str
    protocol: str
    tool: str
    virtual_users: int
    concurrency_limit: int | None
    duration_seconds: float

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    throughput_rps: float = 0.0

    # Latency percentiles (ms)
    latency_min: float = 0.0
    latency_p50: float = 0.0
    latency_p75: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_max: float = 0.0
    latency_mean: float = 0.0
    latency_stddev: float = 0.0

    # System metrics (averages over test duration)
    avg_cpu_percent: float = 0.0
    avg_memory_rss_mb: float = 0.0
    peak_memory_rss_mb: float = 0.0
    avg_thread_count: float = 0.0

    # Raw data (not serialized to summary CSV)
    raw_latencies: list[float] = field(default_factory=list, repr=False)
    errors: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary for CSV/DataFrame output."""
        return {
            "scenario_id": self.scenario_id,
            "server": self.server,
            "protocol": self.protocol,
            "tool": self.tool,
            "virtual_users": self.virtual_users,
            "concurrency_limit": self.concurrency_limit,
            "duration_s": round(self.duration_seconds, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate_pct": round(self.error_rate * 100, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_min_ms": round(self.latency_min, 3),
            "latency_p50_ms": round(self.latency_p50, 3),
            "latency_p75_ms": round(self.latency_p75, 3),
            "latency_p90_ms": round(self.latency_p90, 3),
            "latency_p95_ms": round(self.latency_p95, 3),
            "latency_p99_ms": round(self.latency_p99, 3),
            "latency_max_ms": round(self.latency_max, 3),
            "latency_mean_ms": round(self.latency_mean, 3),
            "latency_stddev_ms": round(self.latency_stddev, 3),
            "avg_cpu_pct": round(self.avg_cpu_percent, 2),
            "avg_memory_mb": round(self.avg_memory_rss_mb, 2),
            "peak_memory_mb": round(self.peak_memory_rss_mb, 2),
        }


class MetricsCollector:
    """Collects request metrics during a benchmark run and computes aggregates."""

    def __init__(self):
        self.requests: list[RequestMetric] = []
        self.system_samples: list[SystemSample] = []
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._warmup_end: float | None = None

    def start(self, warmup_seconds: float = 0):
        self._start_time = time.time()
        self._warmup_end = self._start_time + warmup_seconds

    def stop(self):
        self._end_time = time.time()

    def record_request(self, metric: RequestMetric):
        """Record a single request metric."""
        self.requests.append(metric)

    def record_system_sample(self, sample: SystemSample):
        """Record a system metrics sample."""
        self.system_samples.append(sample)

    def aggregate(self, scenario) -> ScenarioResult:
        """Compute aggregated statistics from collected metrics."""
        # Filter out warmup period
        measurement_requests = [
            r for r in self.requests
            if self._warmup_end is None or r.timestamp >= self._warmup_end
        ]

        if not measurement_requests:
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_display=scenario.display_name,
                server=scenario.server.value,
                protocol=scenario.protocol.value,
                tool=scenario.tool.value,
                virtual_users=scenario.virtual_users,
                concurrency_limit=scenario.concurrency_limit,
                duration_seconds=0,
            )

        latencies = [r.latency_ms for r in measurement_requests]
        successes = [r for r in measurement_requests if r.success]
        failures = [r for r in measurement_requests if not r.success]
        lat_array = np.array(latencies)

        # Calculate measurement window duration
        first_ts = min(r.timestamp for r in measurement_requests)
        last_ts = max(r.timestamp for r in measurement_requests)
        duration = last_ts - first_ts if last_ts > first_ts else 1.0

        # System metrics (only during measurement window)
        sys_samples = [
            s for s in self.system_samples
            if self._warmup_end is None or s.timestamp >= self._warmup_end
        ]

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_display=scenario.display_name,
            server=scenario.server.value,
            protocol=scenario.protocol.value,
            tool=scenario.tool.value,
            virtual_users=scenario.virtual_users,
            concurrency_limit=scenario.concurrency_limit,
            duration_seconds=duration,
            total_requests=len(measurement_requests),
            successful_requests=len(successes),
            failed_requests=len(failures),
            error_rate=len(failures) / len(measurement_requests) if measurement_requests else 0,
            throughput_rps=len(measurement_requests) / duration,
            latency_min=float(np.min(lat_array)),
            latency_p50=float(np.percentile(lat_array, 50)),
            latency_p75=float(np.percentile(lat_array, 75)),
            latency_p90=float(np.percentile(lat_array, 90)),
            latency_p95=float(np.percentile(lat_array, 95)),
            latency_p99=float(np.percentile(lat_array, 99)),
            latency_max=float(np.max(lat_array)),
            latency_mean=float(np.mean(lat_array)),
            latency_stddev=float(np.std(lat_array)),
            raw_latencies=latencies,
            errors=[r.error for r in failures if r.error],
        )

        if sys_samples:
            result.avg_cpu_percent = np.mean([s.cpu_percent for s in sys_samples])
            result.avg_memory_rss_mb = np.mean([s.memory_rss_mb for s in sys_samples])
            result.peak_memory_rss_mb = max(s.memory_rss_mb for s in sys_samples)
            result.avg_thread_count = np.mean([s.thread_count for s in sys_samples])

        return result

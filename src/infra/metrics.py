"""Metrics integration for Prometheus and CloudWatch."""

from __future__ import annotations

import time
from dataclasses import dataclass

from prometheus_client import Counter, Histogram


_VALIDATION_COUNT = Counter(
    "rag_validation_requests_total",
    "Total number of validation requests",
    labelnames=("status",),
)

_VALIDATION_LATENCY = Histogram(
    "rag_validation_latency_seconds",
    "Validation latency in seconds",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)


@dataclass(slots=True)
class MetricsEvent:
    status: str
    latency_seconds: float


def record_validation_metrics(status: str, latency_seconds: float) -> MetricsEvent:
    """Records validation counters and latency histograms."""
    safe_status = status if status in {"passed", "failed", "error"} else "error"
    bounded_latency = max(0.0, float(latency_seconds))

    _VALIDATION_COUNT.labels(status=safe_status).inc()
    _VALIDATION_LATENCY.observe(bounded_latency)

    return MetricsEvent(status=safe_status, latency_seconds=bounded_latency)


class ValidationTimer:
    """Context manager for measuring and recording validation latency."""

    def __init__(self, status: str = "passed"):
        self.status = status
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else self.status
        record_validation_metrics(status=status, latency_seconds=time.perf_counter() - self._start)
        return False

"""Benchmarks that live outside this repository.

`spec.json` and `store-schema.json` are the contract between the validator and the dashboard.
A benchmark in another repository needs a third: `api.py` says what a benchmark distribution
must expose, and nothing here imports one.
"""

from __future__ import annotations

from .api import (
    BENCHMARK_API_VERSION,
    COMMAND_METHODS,
    PURE_METHODS,
    Benchmark,
    validate_plugin,
)

__all__ = [
    "BENCHMARK_API_VERSION",
    "COMMAND_METHODS",
    "PURE_METHODS",
    "Benchmark",
    "validate_plugin",
]

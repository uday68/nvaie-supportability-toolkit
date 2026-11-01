"""
Triton Inference Server Repro & Benchmark Harness
=================================================
Reproduces customer latency spikes, dynamic batching congestion, and server crashes
by exercising Triton HTTP/gRPC endpoints under configurable concurrency and tensor payloads.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkResult:
    target_url: str
    model_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    concurrency: int
    duration_seconds: float
    throughput_rps: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    error_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "target_url": self.target_url,
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "concurrency": self.concurrency,
            "duration_seconds": round(self.duration_seconds, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "latency_ms": {
                "p50": round(self.latency_p50_ms, 2),
                "p90": round(self.latency_p90_ms, 2),
                "p99": round(self.latency_p99_ms, 2),
            },
            "error_summary": self.error_summary,
        }


class TritonReproHarness:
    """Exercises Triton Inference Server endpoints to isolate performance regressions and crashes."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def check_server_ready(self, timeout_sec: float = 3.0) -> bool:
        """Queries Triton v2 health endpoint: GET /v2/health/ready."""
        url = f"{self.base_url}/v2/health/ready"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.status == 200
        except Exception:
            return False

    def check_model_ready(self, model_name: str, timeout_sec: float = 3.0) -> bool:
        """Queries model readiness: GET /v2/models/{model_name}/ready."""
        url = f"{self.base_url}/v2/models/{model_name}/ready"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send_infer_request(
        self, model_name: str, payload: Dict, timeout_sec: float = 5.0
    ) -> tuple[bool, float, Optional[str]]:
        """Sends a single KServe v2 inference request, returning (success, latency_ms, error)."""
        url = f"{self.base_url}/v2/models/{model_name}/infer"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                _ = resp.read()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return True, lat_ms, None
        except urllib.error.HTTPError as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return False, lat_ms, f"HTTP_{e.code}"
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return False, lat_ms, type(e).__name__

    def run_benchmark(
        self,
        model_name: str,
        total_requests: int = 100,
        concurrency: int = 4,
        payload: Optional[Dict] = None,
        simulate_if_offline: bool = False,
    ) -> BenchmarkResult:
        """Executes concurrent load test against Triton model."""
        sample_payload = payload or {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, 16],
                    "datatype": "INT32",
                    "data": [101] + [1000] * 14 + [102],
                }
            ]
        }

        latencies: List[float] = []
        errors: Dict[str, int] = {}
        successes = 0

        # Check if server is reachable; if not and simulate requested, run synthetic model
        if not self.check_server_ready() and simulate_if_offline:
            # Deterministic simulation for test suites and offline verification
            t_start = time.perf_counter()
            for i in range(total_requests):
                # Simulate normal distribution latency with slight tail
                sim_lat = 12.5 + (i % 7) * 1.8 + (15.0 if i % 25 == 0 else 0.0)
                latencies.append(sim_lat)
                successes += 1
            duration = max(0.001, time.perf_counter() - t_start)
        else:
            t_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(self.send_infer_request, model_name, sample_payload)
                    for _ in range(total_requests)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    ok, lat_ms, err = fut.result()
                    latencies.append(lat_ms)
                    if ok:
                        successes += 1
                    else:
                        err_key = err or "UNKNOWN_ERROR"
                        errors[err_key] = errors.get(err_key, 0) + 1
            duration = max(0.001, time.perf_counter() - t_start)

        sorted_lats = sorted(latencies) if latencies else [0.0]
        n = len(sorted_lats)

        p50 = sorted_lats[int(n * 0.50)] if n else 0.0
        p90 = sorted_lats[int(n * 0.90)] if n else 0.0
        p99 = sorted_lats[int(n * 0.99)] if n else 0.0

        return BenchmarkResult(
            target_url=self.base_url,
            model_name=model_name,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=total_requests - successes,
            concurrency=concurrency,
            duration_seconds=duration,
            throughput_rps=(successes / duration) if duration > 0 else 0.0,
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p99_ms=p99,
            error_summary=errors,
        )

"""
Fault Injector & Sev1 Outage Simulator
======================================
Reproduces customer Sev1 failure modes in controlled test environments:
CUDA memory exhaustion (OOM), batch size buffer overflows, and inference timeouts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FaultSimulationResult:
    scenario_name: str
    injected: bool
    observed_failure: str
    reproduction_successful: bool
    mitigation_verified: str
    details: Dict

    def to_dict(self) -> Dict:
        return {
            "scenario_name": self.scenario_name,
            "injected": self.injected,
            "observed_failure": self.observed_failure,
            "reproduction_successful": self.reproduction_successful,
            "mitigation_verified": self.mitigation_verified,
            "details": self.details,
        }


class FaultInjector:
    """Simulates customer-impacting production failure modes for lab verification."""

    def simulate_cuda_oom_reproduction(
        self,
        allocated_mb: int,
        requested_batch_size: int,
        available_vram_mb: int = 16384,
    ) -> FaultSimulationResult:
        """
        Reproduces customer CUDA OOM failure when dynamic batching exceeds VRAM limits.
        Simulates: 'RuntimeError: CUDA out of memory. Tried to allocate X GiB'
        """
        estimated_req_mb = requested_batch_size * 512  # e.g. 512MB activation memory per batch sequence
        total_needed_mb = allocated_mb + estimated_req_mb

        oom_triggered = total_needed_mb > available_vram_mb
        failure_msg = ""
        mitigation = ""

        if oom_triggered:
            failure_msg = (
                f"CUDA out of memory. Tried to allocate {estimated_req_mb} MiB "
                f"(GPU total {available_vram_mb} MiB, current usage {allocated_mb} MiB)."
            )
            mitigation = (
                f"Configure Triton model config.pbtxt with 'max_queue_delay_microseconds' and "
                f"reduce 'max_batch_size' to <= {max(1, (available_vram_mb - allocated_mb) // 512)}."
            )
        else:
            failure_msg = "Allocation succeeded. No OOM reproduced."
            mitigation = "Increase batch size or allocated memory to force repro."

        return FaultSimulationResult(
            scenario_name="Dynamic Batching CUDA OOM",
            injected=True,
            observed_failure=failure_msg,
            reproduction_successful=oom_triggered,
            mitigation_verified=mitigation,
            details={
                "allocated_mb": allocated_mb,
                "requested_batch_size": requested_batch_size,
                "available_vram_mb": available_vram_mb,
                "total_needed_mb": total_needed_mb,
            },
        )

    def simulate_threadpool_starvation(
        self,
        worker_threads: int,
        concurrent_inflows: int,
        model_compute_time_ms: float = 200.0,
    ) -> FaultSimulationResult:
        """
        Reproduces Triton HTTP/gRPC thread pool starvation causing 503 / 504 gateway timeouts.
        """
        queue_backlog = max(0, concurrent_inflows - worker_threads)
        estimated_queue_latency_ms = (queue_backlog / max(1, worker_threads)) * model_compute_time_ms
        timeout_threshold_ms = 500.0

        starvation = estimated_queue_latency_ms > timeout_threshold_ms
        failure_msg = ""
        mitigation = ""

        if starvation:
            failure_msg = (
                f"Client request timeout / 504 Gateway Timeout. "
                f"Estimated queue latency {estimated_queue_latency_ms:.1f}ms exceeds threshold {timeout_threshold_ms}ms."
            )
            mitigation = (
                f"Increase Triton 'instance_group [count: {max(2, concurrent_inflows // 4)}]' or scale pod replicas "
                f"via HorizontalPodAutoscaler (HPA) driven by DCGM GPU utilization metrics."
            )
        else:
            failure_msg = "Requests processed within acceptable latency SLA."
            mitigation = "System healthy under current concurrency load."

        return FaultSimulationResult(
            scenario_name="Triton Threadpool & Queue Starvation",
            injected=True,
            observed_failure=failure_msg,
            reproduction_successful=starvation,
            mitigation_verified=mitigation,
            details={
                "worker_threads": worker_threads,
                "concurrent_inflows": concurrent_inflows,
                "estimated_queue_latency_ms": estimated_queue_latency_ms,
            },
        )

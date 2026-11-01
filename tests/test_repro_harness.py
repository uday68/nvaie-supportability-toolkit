"""Unit tests for Triton Benchmark Harness and Fault Injector."""

from nvaie_toolkit.repro_harness.fault_injector import FaultInjector
from nvaie_toolkit.repro_harness.triton_bench import TritonReproHarness


def test_triton_bench_simulation():
    harness = TritonReproHarness(base_url="http://mock-unreachable-triton:8000")
    result = harness.run_benchmark(
        model_name="transformer_encoder",
        total_requests=40,
        concurrency=4,
        simulate_if_offline=True,
    )

    assert result.total_requests == 40
    assert result.successful_requests == 40
    assert result.failed_requests == 0
    assert result.latency_p50_ms > 0
    assert result.latency_p99_ms >= result.latency_p50_ms
    assert result.throughput_rps > 0


def test_fault_injector_cuda_oom():
    injector = FaultInjector()

    # Case 1: Over-allocation triggers simulated OOM
    result = injector.simulate_cuda_oom_reproduction(
        allocated_mb=14000,
        requested_batch_size=8,  # 8 * 512 = 4096 MB -> 18096 MB > 16384 MB
        available_vram_mb=16384,
    )

    assert result.reproduction_successful is True
    assert "CUDA out of memory" in result.observed_failure
    assert "reduce 'max_batch_size'" in result.mitigation_verified

    # Case 2: Fits in VRAM
    ok_result = injector.simulate_cuda_oom_reproduction(
        allocated_mb=4000,
        requested_batch_size=2,
        available_vram_mb=16384,
    )
    assert ok_result.reproduction_successful is False


def test_fault_injector_threadpool_starvation():
    injector = FaultInjector()
    result = injector.simulate_threadpool_starvation(
        worker_threads=2,
        concurrent_inflows=20,
        model_compute_time_ms=150.0,
    )

    assert result.reproduction_successful is True
    assert "504 Gateway Timeout" in result.observed_failure
    assert "instance_group" in result.mitigation_verified

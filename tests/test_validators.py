"""Unit tests for NVAIE Preflight and Network Fabric Validators."""

import tempfile
from pathlib import Path
from nvaie_toolkit.core.gpu_inspector import GPUClusterReport, GPUInstance
from nvaie_toolkit.core.system_inspector import SystemInspector, SystemReport
from nvaie_toolkit.validators.network_fabric import NetworkFabricValidator
from nvaie_toolkit.validators.preflight import CheckStatus, PreflightValidator


def test_preflight_validator_passes_on_compliant_system():
    # Mock system report
    mock_sys = SystemInspector()
    mock_sys.inspect = lambda: SystemReport(
        os_name="Linux",
        os_version="5.15.0",
        kernel_release="5.15.0-101-generic",
        architecture="x86_64",
        cgroups_version="v2",
        iommu_enabled=True,
        hugepages_total=1024,
        hugepages_free=512,
        hugepages_size_kb=2048,
        numa_nodes_count=2,
        nvidia_pci_devices=["0000:01:00.0 [NVIDIA Device 2330]"],
    )

    validator = PreflightValidator(system_inspector=mock_sys)
    report = validator.run_all_checks()

    assert report.passed_count >= 3
    # CPU arch, cgroups v2, IOMMU should pass
    rule_ids = {r.rule_id: r.status for r in report.rules}
    assert rule_ids["NVAIE-SYS-001"] == CheckStatus.PASS
    assert rule_ids["NVAIE-SYS-002"] == CheckStatus.PASS
    assert rule_ids["NVAIE-SYS-003"] == CheckStatus.PASS
    assert rule_ids["NVAIE-SYS-004"] == CheckStatus.PASS


def test_preflight_validator_flags_uncorrectable_ecc():
    mock_sys = SystemInspector()
    mock_sys.inspect = lambda: SystemReport(
        os_name="Linux",
        os_version="5.15.0",
        kernel_release="5.15.0",
        architecture="x86_64",
        cgroups_version="v2",
        iommu_enabled=True,
        hugepages_total=512,
        hugepages_free=256,
        hugepages_size_kb=2048,
        numa_nodes_count=1,
    )

    class MockGPUInspector:
        def inspect(self):
            return GPUClusterReport(
                driver_version="535.129.03",
                cuda_driver_version="12.2",
                gpu_count=1,
                gpus=[
                    GPUInstance(
                        index=0,
                        name="NVIDIA H100 80GB HBM3",
                        uuid="GPU-TEST",
                        driver_version="535.129.03",
                        cuda_version="12.2",
                        memory_total_mb=81920,
                        memory_used_mb=1000,
                        memory_free_mb=80920,
                        gpu_utilization_pct=10,
                        memory_utilization_pct=5,
                        temperature_gpu_c=45,
                        power_draw_w=150.0,
                        power_limit_w=700.0,
                        mig_mode="Disabled",
                        ecc_single_bit_errors=4,
                        ecc_double_bit_errors=2,  # Critical uncorrectable ECC!
                        pcie_gen_current=5,
                        pcie_gen_max=5,
                        pcie_width_current=16,
                        pcie_width_max=16,
                    )
                ],
                healthy=False,
            )

    validator = PreflightValidator(system_inspector=mock_sys, gpu_inspector=MockGPUInspector())
    report = validator.run_all_checks()

    assert report.overall_status == CheckStatus.FAIL
    rule_ids = {r.rule_id: r.status for r in report.rules}
    assert rule_ids["NVAIE-ECC-001"] == CheckStatus.FAIL


def test_network_fabric_validator_detects_rdma_and_peermem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        proc_dir = root / "proc"
        proc_dir.mkdir(parents=True)
        (proc_dir / "modules").write_text("nvidia_peermem 20480 0 - Live 0xffffffffc0000000\n")

        net_dir = root / "sys" / "class" / "net"
        mlx_dir = net_dir / "mlx5_0"
        mlx_dir.mkdir(parents=True)
        (mlx_dir / "mtu").write_text("9000\n")
        (mlx_dir / "operstate").write_text("up\n")

        validator = NetworkFabricValidator(sys_root=str(root))
        report = validator.validate()

        assert report.nvidia_peermem_loaded is True
        assert report.gpudirect_rdma_supported is True
        assert report.mtu_compliant is True
        assert len(report.rdma_interfaces) == 1
        assert report.rdma_interfaces[0].mtu == 9000

"""
Preflight Validator
===================
Executes pre-installation and pre-upgrade readiness checks for NVIDIA AI Enterprise (NVAIE)
covering host OS, kernel parameters, cgroups, IOMMU, driver/CUDA matrix, and container toolkit.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from nvaie_toolkit.core.gpu_inspector import GPUClusterReport, GPUInspector
from nvaie_toolkit.core.system_inspector import SystemInspector, SystemReport


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ValidationRuleResult:
    rule_id: str
    category: str
    title: str
    status: CheckStatus
    actual: str
    expected: str
    remediation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "title": self.title,
            "status": self.status.value,
            "actual": self.actual,
            "expected": self.expected,
            "remediation": self.remediation,
        }


@dataclass
class PreflightReport:
    overall_status: CheckStatus
    rules: List[ValidationRuleResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.rules if r.status == CheckStatus.PASS)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.rules if r.status == CheckStatus.WARN)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.rules if r.status == CheckStatus.FAIL)

    def to_dict(self) -> Dict:
        return {
            "overall_status": self.overall_status.value,
            "summary": {
                "passed": self.passed_count,
                "warnings": self.warning_count,
                "failures": self.failure_count,
                "total": len(self.rules),
            },
            "rules": [r.to_dict() for r in self.rules],
        }


# Certified NVAIE Production Driver branches
CERTIFIED_NVAIE_DRIVER_BRANCHES = [535, 550, 560, 570]


class PreflightValidator:
    """Evaluates host readiness for NVIDIA AI Enterprise."""

    def __init__(
        self,
        system_inspector: Optional[SystemInspector] = None,
        gpu_inspector: Optional[GPUInspector] = None,
        config_root: str = "/",
    ):
        self.sys_inspector = system_inspector or SystemInspector(sys_root=config_root)
        self.gpu_inspector = gpu_inspector or GPUInspector()
        self.config_root = Path(config_root)

    def run_all_checks(self) -> PreflightReport:
        rules: List[ValidationRuleResult] = []
        sys_report = self.sys_inspector.inspect()
        gpu_report = self.gpu_inspector.inspect()

        # Rule 1: Operating System & Architecture
        if sys_report.architecture in ("x86_64", "aarch64", "AMD64"):
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-001",
                    category="System",
                    title="CPU Architecture Compatibility",
                    status=CheckStatus.PASS,
                    actual=sys_report.architecture,
                    expected="x86_64 or aarch64",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-001",
                    category="System",
                    title="CPU Architecture Compatibility",
                    status=CheckStatus.FAIL,
                    actual=sys_report.architecture,
                    expected="x86_64 or aarch64",
                    remediation="NVIDIA AI Enterprise requires 64-bit x86_64 or ARM64 (Grace/Neoverse) architecture.",
                )
            )

        # Rule 2: Cgroups version (v2 strongly recommended for K8s 1.25+)
        if sys_report.cgroups_version == "v2":
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-002",
                    category="Kernel",
                    title="Control Groups (cgroups) Version",
                    status=CheckStatus.PASS,
                    actual="cgroups v2",
                    expected="cgroups v2",
                )
            )
        elif sys_report.cgroups_version == "v1":
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-002",
                    category="Kernel",
                    title="Control Groups (cgroups) Version",
                    status=CheckStatus.WARN,
                    actual="cgroups v1 legacy",
                    expected="cgroups v2",
                    remediation="Enable cgroups v2 by setting systemd.unified_cgroup_hierarchy=1 in GRUB kernel cmdline.",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-002",
                    category="Kernel",
                    title="Control Groups (cgroups) Version",
                    status=CheckStatus.WARN,
                    actual="unknown",
                    expected="cgroups v2",
                    remediation="Verify host cgroups mountpoints (/sys/fs/cgroup).",
                )
            )

        # Rule 3: IOMMU / VT-d
        if sys_report.iommu_enabled:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-003",
                    category="Virtualization",
                    title="IOMMU / VT-d Hardware Virtualization",
                    status=CheckStatus.PASS,
                    actual="Enabled",
                    expected="Enabled",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-003",
                    category="Virtualization",
                    title="IOMMU / VT-d Hardware Virtualization",
                    status=CheckStatus.WARN,
                    actual="Disabled",
                    expected="Enabled",
                    remediation="Enable intel_iommu=on or amd_iommu=on in kernel cmdline if utilizing vGPU or SR-IOV.",
                )
            )

        # Rule 4: HugePages Configuration
        if sys_report.hugepages_total > 0:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-004",
                    category="Memory",
                    title="HugePages Allocation",
                    status=CheckStatus.PASS,
                    actual=f"{sys_report.hugepages_total} pages ({sys_report.hugepages_size_kb} kB)",
                    expected="> 0 pages",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-SYS-004",
                    category="Memory",
                    title="HugePages Allocation",
                    status=CheckStatus.WARN,
                    actual="0 pages configured",
                    expected="Configured for high-throughput Triton / NCCL buffers",
                    remediation="Configure vm.nr_hugepages in /etc/sysctl.conf for optimized tensor DMA memory locks.",
                )
            )

        # Rule 5: NVIDIA Container Toolkit & Runtime
        toolkit_config = self.config_root / "etc" / "nvidia-container-runtime" / "config.toml"
        ctk_binary = shutil.which("nvidia-ctk")
        if toolkit_config.exists() or ctk_binary is not None:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-CTK-001",
                    category="Container Runtime",
                    title="NVIDIA Container Toolkit Presence",
                    status=CheckStatus.PASS,
                    actual="Configured and present",
                    expected="Present (/etc/nvidia-container-runtime or nvidia-ctk)",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-CTK-001",
                    category="Container Runtime",
                    title="NVIDIA Container Toolkit Presence",
                    status=CheckStatus.WARN,
                    actual="Not found in standard path",
                    expected="NVIDIA Container Toolkit installed",
                    remediation="Install nvidia-container-toolkit or enable Container Toolkit DaemonSet in GPU Operator.",
                )
            )

        # Rule 6: Driver Version Certified Matrix
        if gpu_report.driver_version != "N/A" and gpu_report.driver_version != "unknown":
            try:
                major_ver = int(gpu_report.driver_version.split(".")[0])
                if major_ver in CERTIFIED_NVAIE_DRIVER_BRANCHES:
                    rules.append(
                        ValidationRuleResult(
                            rule_id="NVAIE-DRV-001",
                            category="NVIDIA Driver",
                            title="Certified Production Driver Branch",
                            status=CheckStatus.PASS,
                            actual=f"Branch {major_ver} ({gpu_report.driver_version})",
                            expected=f"One of {CERTIFIED_NVAIE_DRIVER_BRANCHES}",
                        )
                    )
                else:
                    rules.append(
                        ValidationRuleResult(
                            rule_id="NVAIE-DRV-001",
                            category="NVIDIA Driver",
                            title="Certified Production Driver Branch",
                            status=CheckStatus.WARN,
                            actual=f"Branch {major_ver} ({gpu_report.driver_version})",
                            expected=f"One of {CERTIFIED_NVAIE_DRIVER_BRANCHES}",
                            remediation="Consider deploying certified NVAIE enterprise production driver branch (e.g. 535 or 550).",
                        )
                    )
            except Exception:
                pass
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-DRV-001",
                    category="NVIDIA Driver",
                    title="Certified Production Driver Branch",
                    status=CheckStatus.WARN,
                    actual="Driver not detected via nvidia-smi",
                    expected="NVAIE Certified Driver installed",
                    remediation="If deploying on Kubernetes via GPU Operator, ensure driver container is enabled.",
                )
            )

        # Rule 7: Double-bit ECC uncorrectable errors
        has_dbe = any(g.ecc_double_bit_errors > 0 for g in gpu_report.gpus)
        if has_dbe:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-ECC-001",
                    category="GPU Health",
                    title="Uncorrectable ECC Double-Bit Errors",
                    status=CheckStatus.FAIL,
                    actual="Double-bit uncorrectable ECC errors detected!",
                    expected="0 double-bit errors",
                    remediation="Hardware memory failure detected. Drain node and schedule hardware replacement.",
                )
            )
        else:
            rules.append(
                ValidationRuleResult(
                    rule_id="NVAIE-ECC-001",
                    category="GPU Health",
                    title="Uncorrectable ECC Double-Bit Errors",
                    status=CheckStatus.PASS,
                    actual="0 double-bit errors",
                    expected="0 double-bit errors",
                )
            )

        # Determine overall status
        if any(r.status == CheckStatus.FAIL for r in rules):
            overall = CheckStatus.FAIL
        elif any(r.status == CheckStatus.WARN for r in rules):
            overall = CheckStatus.WARN
        else:
            overall = CheckStatus.PASS

        return PreflightReport(overall_status=overall, rules=rules)

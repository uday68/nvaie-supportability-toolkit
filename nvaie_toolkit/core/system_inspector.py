"""
System Inspector
================
Gathers low-level operating system, kernel, cgroups, PCIe, and memory subsystem
telemetry required for NVIDIA AI Enterprise deployment readiness.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SystemReport:
    os_name: str
    os_version: str
    kernel_release: str
    architecture: str
    cgroups_version: str  # "v1", "v2", or "unknown"
    iommu_enabled: bool
    hugepages_total: int
    hugepages_free: int
    hugepages_size_kb: int
    numa_nodes_count: int
    nvidia_pci_devices: List[str] = field(default_factory=list)
    kernel_cmdline: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "kernel_release": self.kernel_release,
            "architecture": self.architecture,
            "cgroups_version": self.cgroups_version,
            "iommu_enabled": self.iommu_enabled,
            "hugepages": {
                "total": self.hugepages_total,
                "free": self.hugepages_free,
                "size_kb": self.hugepages_size_kb,
            },
            "numa_nodes_count": self.numa_nodes_count,
            "nvidia_pci_devices": self.nvidia_pci_devices,
            "kernel_cmdline": self.kernel_cmdline,
            "errors": self.errors,
        }


class SystemInspector:
    """Inspects host OS and hardware subsystem topology."""

    def __init__(self, sys_root: str = "/"):
        self.sys_root = Path(sys_root)

    def detect_cgroups_version(self) -> str:
        """Determines if the system is running cgroups v1 or v2."""
        cgroup_v2_path = self.sys_root / "sys" / "fs" / "cgroup" / "cgroup.controllers"
        if cgroup_v2_path.exists():
            return "v2"

        proc_cgroups = self.sys_root / "proc" / "cgroups"
        if proc_cgroups.exists():
            return "v1"

        return "unknown"

    def check_iommu(self) -> bool:
        """Checks if IOMMU (Intel VT-d or AMD-Vi) is enabled for PCI passthrough / vGPU."""
        iommu_dir = self.sys_root / "sys" / "kernel" / "iommu_groups"
        if iommu_dir.exists() and any(iommu_dir.iterdir()):
            return True

        cmdline_file = self.sys_root / "proc" / "cmdline"
        if cmdline_file.exists():
            try:
                cmdline = cmdline_file.read_text().strip()
                if "intel_iommu=on" in cmdline or "amd_iommu=on" in cmdline or "iommu=pt" in cmdline:
                    return True
            except Exception:
                pass
        return False

    def get_hugepages_info(self) -> tuple[int, int, int]:
        """Returns (HugePages_Total, HugePages_Free, Hugepagesize in kB)."""
        meminfo_path = self.sys_root / "proc" / "meminfo"
        total, free, size_kb = 0, 0, 2048
        if meminfo_path.exists():
            try:
                for line in meminfo_path.read_text().splitlines():
                    if line.startswith("HugePages_Total:"):
                        total = int(line.split()[1])
                    elif line.startswith("HugePages_Free:"):
                        free = int(line.split()[1])
                    elif line.startswith("Hugepagesize:"):
                        size_kb = int(line.split()[1])
            except Exception:
                pass
        return total, free, size_kb

    def get_numa_nodes_count(self) -> int:
        """Returns number of detected NUMA nodes."""
        node_dir = self.sys_root / "sys" / "devices" / "system" / "node"
        if node_dir.exists():
            try:
                nodes = [p for p in node_dir.iterdir() if p.name.startswith("node") and p.name[4:].isdigit()]
                return len(nodes) if nodes else 1
            except Exception:
                return 1
        return 1

    def detect_nvidia_pci_devices(self) -> List[str]:
        """Detects NVIDIA PCIe accelerators (PCI Vendor 10de)."""
        devices = []
        # On Linux, inspect sysfs directly if available
        bus_pci = self.sys_root / "sys" / "bus" / "pci" / "devices"
        if bus_pci.exists():
            try:
                for dev in bus_pci.iterdir():
                    vendor_file = dev / "vendor"
                    if vendor_file.exists():
                        vendor = vendor_file.read_text().strip().lower()
                        if vendor in ("0x10de", "10de"):
                            dev_id = ""
                            device_file = dev / "device"
                            if device_file.exists():
                                dev_id = device_file.read_text().strip()
                            devices.append(f"{dev.name} [NVIDIA Device {dev_id}]")
            except Exception:
                pass

        if not devices and platform.system() != "Windows":
            # Fallback to lspci
            try:
                out = subprocess.check_output(["lspci", "-d", "10de:"], text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    if line.strip():
                        devices.append(line.strip())
            except Exception:
                pass

        return devices

    def inspect(self) -> SystemReport:
        """Executes full system inspection."""
        errors: List[str] = []
        cmdline = ""
        cmdline_file = self.sys_root / "proc" / "cmdline"
        if cmdline_file.exists():
            try:
                cmdline = cmdline_file.read_text().strip()
            except Exception as e:
                errors.append(f"Failed to read /proc/cmdline: {e}")

        total_hp, free_hp, hp_size = self.get_hugepages_info()

        return SystemReport(
            os_name=platform.system(),
            os_version=platform.version(),
            kernel_release=platform.release(),
            architecture=platform.machine(),
            cgroups_version=self.detect_cgroups_version(),
            iommu_enabled=self.check_iommu(),
            hugepages_total=total_hp,
            hugepages_free=free_hp,
            hugepages_size_kb=hp_size,
            numa_nodes_count=self.get_numa_nodes_count(),
            nvidia_pci_devices=self.detect_nvidia_pci_devices(),
            kernel_cmdline=cmdline,
            errors=errors,
        )

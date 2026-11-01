"""
GPU Inspector
=============
Extracts deep NVIDIA GPU hardware, driver, ECC memory, MIG slicing,
and PCIe topology telemetry via nvidia-smi / NVML parsing with structured fallbacks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GPUInstance:
    index: int
    name: str
    uuid: str
    driver_version: str
    cuda_version: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    gpu_utilization_pct: int
    memory_utilization_pct: int
    temperature_gpu_c: int
    power_draw_w: float
    power_limit_w: float
    mig_mode: str  # "Enabled", "Disabled", "N/A"
    ecc_single_bit_errors: int
    ecc_double_bit_errors: int  # Critical uncorrectable
    pcie_gen_current: int
    pcie_gen_max: int
    pcie_width_current: int
    pcie_width_max: int
    throttle_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "memory": {
                "total_mb": self.memory_total_mb,
                "used_mb": self.memory_used_mb,
                "free_mb": self.memory_free_mb,
            },
            "utilization": {
                "gpu_pct": self.gpu_utilization_pct,
                "memory_pct": self.memory_utilization_pct,
            },
            "thermal_power": {
                "temperature_c": self.temperature_gpu_c,
                "power_draw_w": self.power_draw_w,
                "power_limit_w": self.power_limit_w,
            },
            "mig_mode": self.mig_mode,
            "ecc_errors": {
                "single_bit": self.ecc_single_bit_errors,
                "double_bit_uncorrectable": self.ecc_double_bit_errors,
            },
            "pcie": {
                "gen_current": self.pcie_gen_current,
                "gen_max": self.pcie_gen_max,
                "width_current": self.pcie_width_current,
                "width_max": self.pcie_width_max,
            },
            "throttle_reasons": self.throttle_reasons,
        }


@dataclass
class GPUClusterReport:
    driver_version: str
    cuda_driver_version: str
    gpu_count: int
    gpus: List[GPUInstance] = field(default_factory=list)
    healthy: bool = True
    critical_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "driver_version": self.driver_version,
            "cuda_driver_version": self.cuda_driver_version,
            "gpu_count": self.gpu_count,
            "gpus": [g.to_dict() for g in self.gpus],
            "healthy": self.healthy,
            "critical_alerts": self.critical_alerts,
            "warnings": self.warnings,
        }


class GPUInspector:
    """Discovers and evaluates NVIDIA GPU fleet status."""

    def __init__(self, nvidia_smi_bin: Optional[str] = None):
        self.nvidia_smi = nvidia_smi_bin or shutil.which("nvidia-smi")

    def is_available(self) -> bool:
        return self.nvidia_smi is not None

    def query_xml(self) -> Optional[str]:
        """Runs nvidia-smi -q -x and captures structured XML output."""
        if not self.nvidia_smi:
            return None
        try:
            return subprocess.check_output(
                [self.nvidia_smi, "-q", "-x"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception:
            return None

    def parse_xml_report(self, xml_str: str) -> GPUClusterReport:
        """Parses nvidia-smi XML output into strongly typed GPUClusterReport."""
        root = ET.fromstring(xml_str)
        driver_ver = root.findtext("driver_version", "unknown")
        cuda_ver = root.findtext("cuda_version", "unknown")

        gpus: List[GPUInstance] = []
        alerts: List[str] = []
        warnings: List[str] = []

        for idx, gpu_elem in enumerate(root.findall("gpu")):
            name = gpu_elem.findtext("product_name", "Unknown GPU")
            uuid = gpu_elem.findtext("uuid", f"GPU-UUID-{idx}")

            # Memory
            fb_mem = gpu_elem.find("fb_memory_usage")
            total_mb = int(fb_mem.findtext("total", "0 MiB").split()[0]) if fb_mem is not None else 0
            used_mb = int(fb_mem.findtext("used", "0 MiB").split()[0]) if fb_mem is not None else 0
            free_mb = int(fb_mem.findtext("free", "0 MiB").split()[0]) if fb_mem is not None else 0

            # Utilization
            util_elem = gpu_elem.find("utilization")
            gpu_util = int(util_elem.findtext("gpu_util", "0 %").split()[0]) if util_elem is not None else 0
            mem_util = int(util_elem.findtext("memory_util", "0 %").split()[0]) if util_elem is not None else 0

            # Thermal & Power
            temp_elem = gpu_elem.find("temperature")
            temp_c = int(temp_elem.findtext("gpu_temp", "0 C").split()[0]) if temp_elem is not None else 0

            power_elem = gpu_elem.find("power_readings") or gpu_elem.find("gpu_power_readings")
            power_draw = float(power_elem.findtext("power_draw", "0.0 W").split()[0]) if power_elem is not None else 0.0
            power_limit = float(power_elem.findtext("power_limit", "0.0 W").split()[0]) if power_elem is not None else 0.0

            # MIG Mode
            mig_elem = gpu_elem.find("mig_mode")
            mig_mode = mig_elem.findtext("current_mig", "N/A") if mig_elem is not None else "N/A"

            # ECC Errors
            ecc_elem = gpu_elem.find("ecc_errors")
            sbe, dbe = 0, 0
            if ecc_elem is not None:
                # aggregate uncorrected
                agg = ecc_elem.find("aggregate")
                if agg is not None:
                    try:
                        dbe = int(agg.findtext("dram_double", "0"))
                    except ValueError:
                        dbe = 0
                    try:
                        sbe = int(agg.findtext("dram_single", "0"))
                    except ValueError:
                        sbe = 0

            # PCIe Link Status
            pci_elem = gpu_elem.find("pci")
            gen_curr, gen_max, width_curr, width_max = 0, 0, 0, 0
            if pci_elem is not None:
                pci_link = pci_elem.find("pci_gpu_link_info")
                if pci_link is not None:
                    try:
                        gen_curr = int(pci_link.findtext("pcie_gen/current_link_gen", "0"))
                    except ValueError:
                        gen_curr = 0
                    try:
                        gen_max = int(pci_link.findtext("pcie_gen/max_link_gen", "0"))
                    except ValueError:
                        gen_max = 0
                    try:
                        raw_w_curr = pci_link.findtext("link_widths/current_link_width", "0").strip().rstrip("xX")
                        width_curr = int(raw_w_curr) if raw_w_curr.isdigit() else 0
                    except ValueError:
                        width_curr = 0
                    try:
                        raw_w_max = pci_link.findtext("link_widths/max_link_width", "0").strip().rstrip("xX")
                        width_max = int(raw_w_max) if raw_w_max.isdigit() else 0
                    except ValueError:
                        width_max = 0

            # Throttling
            throttle = []
            clocks_event = gpu_elem.find("clocks_event_reasons")
            if clocks_event is not None:
                for child in clocks_event:
                    if child.text and child.text.strip().lower() == "active":
                        throttle.append(child.tag)

            # Health validation rules
            if dbe > 0:
                alerts.append(f"GPU {idx} ({name}) has {dbe} uncorrectable double-bit ECC errors! Node requires drainage.")
            if gen_curr > 0 and gen_max > 0 and gen_curr < gen_max:
                warnings.append(f"GPU {idx} PCIe link degraded to Gen{gen_curr} (hardware supports Gen{gen_max}).")
            if width_curr > 0 and width_max > 0 and width_curr < width_max:
                warnings.append(f"GPU {idx} PCIe link width degraded to x{width_curr} (hardware supports x{width_max}).")
            if temp_c > 85:
                warnings.append(f"GPU {idx} operating at high thermal state: {temp_c}°C.")

            gpus.append(
                GPUInstance(
                    index=idx,
                    name=name,
                    uuid=uuid,
                    driver_version=driver_ver,
                    cuda_version=cuda_ver,
                    memory_total_mb=total_mb,
                    memory_used_mb=used_mb,
                    memory_free_mb=free_mb,
                    gpu_utilization_pct=gpu_util,
                    memory_utilization_pct=mem_util,
                    temperature_gpu_c=temp_c,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    mig_mode=mig_mode,
                    ecc_single_bit_errors=sbe,
                    ecc_double_bit_errors=dbe,
                    pcie_gen_current=gen_curr,
                    pcie_gen_max=gen_max,
                    pcie_width_current=width_curr,
                    pcie_width_max=width_max,
                    throttle_reasons=throttle,
                )
            )

        healthy = len(alerts) == 0
        return GPUClusterReport(
            driver_version=driver_ver,
            cuda_driver_version=cuda_ver,
            gpu_count=len(gpus),
            gpus=gpus,
            healthy=healthy,
            critical_alerts=alerts,
            warnings=warnings,
        )

    def inspect(self) -> GPUClusterReport:
        """Inspects GPUs via nvidia-smi XML, or returns mock report if no GPUs present."""
        xml_data = self.query_xml()
        if xml_data:
            return self.parse_xml_report(xml_data)

        # Graceful fallback report (simulating detection on systems without live GPU)
        return GPUClusterReport(
            driver_version="N/A",
            cuda_driver_version="N/A",
            gpu_count=0,
            gpus=[],
            healthy=True,
            critical_alerts=[],
            warnings=["No active nvidia-smi binary found. Running in CPU/diagnostic mode."],
        )

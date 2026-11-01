"""
Network Fabric Validator
========================
Validates high-performance networking fabric prerequisites for multi-GPU / multi-node
distributed training and inference (GPUDirect RDMA, InfiniBand / RoCE, MTU 9000, and NCCL).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NetworkInterfaceInfo:
    name: str
    mtu: int
    is_rdma: bool
    is_up: bool
    ip_address: Optional[str] = None


@dataclass
class FabricReport:
    gpudirect_rdma_supported: bool
    nvidia_peermem_loaded: bool
    rdma_interfaces: List[NetworkInterfaceInfo] = field(default_factory=list)
    mtu_compliant: bool = True
    nccl_recommendations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "gpudirect_rdma_supported": self.gpudirect_rdma_supported,
            "nvidia_peermem_loaded": self.nvidia_peermem_loaded,
            "rdma_interfaces": [
                {"name": i.name, "mtu": i.mtu, "is_rdma": i.is_rdma, "is_up": i.is_up}
                for i in self.rdma_interfaces
            ],
            "mtu_compliant": self.mtu_compliant,
            "nccl_recommendations": self.nccl_recommendations,
            "errors": self.errors,
        }


class NetworkFabricValidator:
    """Validates cluster networking fabric for GPUDirect RDMA and NCCL throughput."""

    def __init__(self, sys_root: str = "/"):
        self.sys_root = Path(sys_root)

    def check_peermem_module(self) -> bool:
        """Checks if nvidia-peermem (or nv_peer_mem) kernel module is loaded."""
        modules_file = self.sys_root / "proc" / "modules"
        if modules_file.exists():
            try:
                content = modules_file.read_text()
                return ("nvidia_peermem" in content) or ("nv_peer_mem" in content)
            except Exception:
                pass
        return False

    def get_interfaces(self) -> List[NetworkInterfaceInfo]:
        """Discovers network interfaces and MTU via sysfs."""
        net_dir = self.sys_root / "sys" / "class" / "net"
        infiniband_dir = self.sys_root / "sys" / "class" / "infiniband"
        ib_devices = set()

        if infiniband_dir.exists():
            try:
                ib_devices = {p.name for p in infiniband_dir.iterdir()}
            except Exception:
                pass

        interfaces = []
        if net_dir.exists():
            try:
                for iface in net_dir.iterdir():
                    if iface.name in ("lo", "docker0") or iface.name.startswith("veth") or iface.name.startswith("flannel"):
                        continue

                    mtu = 1500
                    mtu_file = iface / "mtu"
                    if mtu_file.exists():
                        try:
                            mtu = int(mtu_file.read_text().strip())
                        except ValueError:
                            pass

                    operstate_file = iface / "operstate"
                    is_up = False
                    if operstate_file.exists():
                        is_up = operstate_file.read_text().strip().lower() == "up"

                    is_rdma = any(dev in iface.name for dev in ("ib", "mlx5", "rdma")) or (iface.name in ib_devices)

                    interfaces.append(
                        NetworkInterfaceInfo(
                            name=iface.name,
                            mtu=mtu,
                            is_rdma=is_rdma,
                            is_up=is_up,
                        )
                    )
            except Exception:
                pass

        return interfaces

    def validate(self) -> FabricReport:
        peermem_loaded = self.check_peermem_module()
        interfaces = self.get_interfaces()
        rdma_ifaces = [i for i in interfaces if i.is_rdma]

        mtu_ok = True
        recommendations = []

        # Check MTU for RDMA interfaces (Enterprise RoCE/IB requires MTU >= 4096 or 9000 jumbo frames)
        for iface in rdma_ifaces:
            if iface.mtu < 4096 and iface.is_up:
                mtu_ok = False
                recommendations.append(
                    f"Interface '{iface.name}' MTU is {iface.mtu}. Enterprise RoCE fabric requires MTU >= 4096 or 9000 jumbo frames."
                )

        if not peermem_loaded:
            recommendations.append(
                "Kernel module 'nvidia-peermem' is not loaded. GPUDirect RDMA direct peer transfers will fall back to CPU staging."
            )

        if rdma_ifaces:
            primary_rdma = rdma_ifaces[0].name
            recommendations.append(
                f"Recommended multi-node NCCL configuration: export NCCL_IB_HCA={primary_rdma} NCCL_IB_DISABLE=0"
            )

        return FabricReport(
            gpudirect_rdma_supported=peermem_loaded and len(rdma_ifaces) > 0,
            nvidia_peermem_loaded=peermem_loaded,
            rdma_interfaces=rdma_ifaces,
            mtu_compliant=mtu_ok,
            nccl_recommendations=recommendations,
        )

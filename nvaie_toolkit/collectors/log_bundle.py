"""
Diagnostic Bundle Collector
===========================
Generates self-contained, sanitized diagnostic tarballs with SHA-256 integrity manifests
for escalation to NVIDIA AI Enterprise Sustaining Engineering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from nvaie_toolkit.collectors.sanitizer import DataSanitizer
from nvaie_toolkit.collectors.xid_analyzer import XidAnalyzer
from nvaie_toolkit.core.gpu_inspector import GPUInspector
from nvaie_toolkit.core.k8s_inspector import K8sOperatorInspector
from nvaie_toolkit.core.system_inspector import SystemInspector
from nvaie_toolkit.validators.network_fabric import NetworkFabricValidator
from nvaie_toolkit.validators.preflight import PreflightValidator


@dataclass
class BundleResult:
    archive_path: str
    sha256_checksum: str
    size_bytes: int
    files_included: List[str]
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "archive_path": self.archive_path,
            "sha256_checksum": self.sha256_checksum,
            "size_bytes": self.size_bytes,
            "files_included": self.files_included,
            "timestamp": self.timestamp,
        }


class DiagnosticBundleCollector:
    """Orchestrates comprehensive diagnostic collection, sanitization, and packaging."""

    def __init__(
        self,
        output_dir: str = "./diagnostics",
        sanitizer: Optional[DataSanitizer] = None,
        custom_keywords: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.sanitizer = sanitizer or DataSanitizer(custom_keywords=custom_keywords)

    def compute_sha256(self, file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def collect(
        self,
        include_k8s: bool = True,
        dmesg_content: Optional[str] = None,
    ) -> BundleResult:
        """Executes full diagnostic collection and archives bundle."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts_str = time.strftime("%Y%m%d-%H%M%S")
        bundle_name = f"nvaie-support-bundle-{ts_str}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_path = Path(tmp_dir) / bundle_name
            stage_path.mkdir(parents=True, exist_ok=True)

            files_included: List[str] = []

            # 1. System Telemetry
            sys_inspector = SystemInspector()
            sys_data = sys_inspector.inspect().to_dict()
            sanitized_sys = self.sanitizer.sanitize_dict(sys_data)
            (stage_path / "system_telemetry.json").write_text(json.dumps(sanitized_sys, indent=2))
            files_included.append("system_telemetry.json")

            # 2. GPU Cluster Status
            gpu_inspector = GPUInspector()
            gpu_data = gpu_inspector.inspect().to_dict()
            sanitized_gpu = self.sanitizer.sanitize_dict(gpu_data)
            (stage_path / "gpu_status.json").write_text(json.dumps(sanitized_gpu, indent=2))
            files_included.append("gpu_status.json")

            # 3. Preflight Readiness
            preflight = PreflightValidator()
            preflight_data = preflight.run_all_checks().to_dict()
            sanitized_pf = self.sanitizer.sanitize_dict(preflight_data)
            (stage_path / "preflight_audit.json").write_text(json.dumps(sanitized_pf, indent=2))
            files_included.append("preflight_audit.json")

            # 4. Network Fabric
            fabric = NetworkFabricValidator()
            fabric_data = fabric.validate().to_dict()
            (stage_path / "network_fabric.json").write_text(json.dumps(fabric_data, indent=2))
            files_included.append("network_fabric.json")

            # 5. Xid Analysis & Kernel dmesg
            xid_analyzer = XidAnalyzer()
            if dmesg_content:
                xid_events = xid_analyzer.parse_log_text(dmesg_content)
                sanitized_dmesg = self.sanitizer.sanitize_text(dmesg_content)
            else:
                xid_events = xid_analyzer.analyze_dmesg()
                sanitized_dmesg = "Live dmesg not captured (run with elevated root/admin if required)."

            (stage_path / "dmesg.log").write_text(sanitized_dmesg)
            files_included.append("dmesg.log")

            xid_summary = xid_analyzer.summarize(xid_events)
            (stage_path / "xid_triage.json").write_text(json.dumps(xid_summary, indent=2))
            files_included.append("xid_triage.json")

            # 6. Kubernetes Operator Status (if requested)
            if include_k8s:
                k8s_inspector = K8sOperatorInspector()
                k8s_data = k8s_inspector.audit().to_dict()
                sanitized_k8s = self.sanitizer.sanitize_dict(k8s_data)
                (stage_path / "k8s_operator_status.json").write_text(json.dumps(sanitized_k8s, indent=2))
                files_included.append("k8s_operator_status.json")

            # Build compressed tar.gz archive
            archive_path = self.output_dir / f"{bundle_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(stage_path, arcname=bundle_name)

            sha256 = self.compute_sha256(archive_path)
            size = archive_path.stat().st_size

            # Write checksum file
            checksum_path = self.output_dir / f"{bundle_name}.sha256"
            checksum_path.write_text(f"{sha256}  {archive_path.name}\n")

            return BundleResult(
                archive_path=str(archive_path),
                sha256_checksum=sha256,
                size_bytes=size,
                files_included=files_included,
                timestamp=ts_str,
            )

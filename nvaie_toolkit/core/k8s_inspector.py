"""
Kubernetes & NVIDIA GPU Operator Inspector
==========================================
Audits the health, DaemonSet status, driver readiness, and node resource
allocatables for NVIDIA GPU Operator in enterprise Kubernetes clusters.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DaemonSetStatus:
    name: str
    desired_number_scheduled: int
    current_number_scheduled: int
    number_ready: int
    number_available: int
    is_healthy: bool

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "desired": self.desired_number_scheduled,
            "current": self.current_number_scheduled,
            "ready": self.number_ready,
            "available": self.number_available,
            "is_healthy": self.is_healthy,
        }


@dataclass
class OperatorAuditReport:
    cluster_connected: bool
    namespace: str
    daemonsets: List[DaemonSetStatus] = field(default_factory=list)
    failed_pods: List[Dict[str, str]] = field(default_factory=list)
    gpu_nodes_count: int = 0
    total_allocatable_gpus: int = 0
    validation_status: str = "UNKNOWN"  # "PASSED", "DEGRADED", "FAILED"
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cluster_connected": self.cluster_connected,
            "namespace": self.namespace,
            "daemonsets": [ds.to_dict() for ds in self.daemonsets],
            "failed_pods": self.failed_pods,
            "gpu_nodes_count": self.gpu_nodes_count,
            "total_allocatable_gpus": self.total_allocatable_gpus,
            "validation_status": self.validation_status,
            "recommendations": self.recommendations,
        }


EXPECTED_OPERATOR_DAEMONSETS = [
    "nvidia-driver-daemonset",
    "nvidia-container-toolkit-daemonset",
    "nvidia-device-plugin-daemonset",
    "nvidia-dcgm-exporter",
    "gpu-feature-discovery",
]


class K8sOperatorInspector:
    """Evaluates NVIDIA GPU Operator status in Kubernetes."""

    def __init__(self, namespace: str = "gpu-operator", kubectl_bin: Optional[str] = None):
        self.namespace = namespace
        self.kubectl = kubectl_bin or shutil.which("kubectl")

    def run_kubectl(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Runs kubectl command and parses JSON response."""
        if not self.kubectl:
            return None
        cmd = [self.kubectl] + args + ["-o", "json"]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15)
            return json.loads(out)
        except Exception:
            return None

    def audit_from_json(
        self, ds_json: Optional[Dict], pod_json: Optional[Dict], node_json: Optional[Dict]
    ) -> OperatorAuditReport:
        """Audits operator state from raw Kubernetes API JSON dicts."""
        if not ds_json and not pod_json and not node_json:
            return OperatorAuditReport(
                cluster_connected=False,
                namespace=self.namespace,
                validation_status="DISCONNECTED",
                recommendations=["Configure KUBECONFIG or ensure kubectl can reach Kubernetes cluster API."],
            )

        daemonset_list: List[DaemonSetStatus] = []
        failed_pods: List[Dict[str, str]] = []
        recommendations: List[str] = []

        # 1. Audit DaemonSets
        if ds_json and "items" in ds_json:
            for item in ds_json["items"]:
                name = item["metadata"]["name"]
                status = item.get("status", {})
                desired = status.get("desiredNumberScheduled", 0)
                current = status.get("currentNumberScheduled", 0)
                ready = status.get("numberReady", 0)
                available = status.get("numberAvailable", 0)
                is_healthy = (desired > 0) and (ready == desired)

                daemonset_list.append(
                    DaemonSetStatus(
                        name=name,
                        desired_number_scheduled=desired,
                        current_number_scheduled=current,
                        number_ready=ready,
                        number_available=available,
                        is_healthy=is_healthy,
                    )
                )

                if not is_healthy and desired > 0:
                    recommendations.append(
                        f"DaemonSet '{name}' degraded: {ready}/{desired} ready. Inspect pod logs on unready nodes."
                    )

        # 2. Audit Pod Failures
        if pod_json and "items" in pod_json:
            for pod in pod_json["items"]:
                pod_name = pod["metadata"]["name"]
                status = pod.get("status", {})
                phase = status.get("phase", "Unknown")

                # Check container statuses
                container_statuses = status.get("containerStatuses", [])
                for cs in container_statuses:
                    waiting = cs.get("state", {}).get("waiting", {})
                    reason = waiting.get("reason", "")
                    if reason in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError"):
                        failed_pods.append(
                            {
                                "pod": pod_name,
                                "container": cs.get("name", "unknown"),
                                "reason": reason,
                                "message": waiting.get("message", ""),
                            }
                        )

        # 3. Audit GPU Nodes & Allocatable Resources
        gpu_nodes = 0
        total_gpus = 0
        if node_json and "items" in node_json:
            for node in node_json["items"]:
                allocatable = node.get("status", {}).get("allocatable", {})
                gpu_count_str = allocatable.get("nvidia.com/gpu", "0")
                try:
                    gpu_count = int(gpu_count_str)
                except ValueError:
                    gpu_count = 0

                if gpu_count > 0:
                    gpu_nodes += 1
                    total_gpus += gpu_count

        if gpu_nodes == 0:
            recommendations.append(
                "No Kubernetes nodes advertise 'nvidia.com/gpu' allocatable capacity. Check nvidia-device-plugin status."
            )

        # Determine validation status
        if failed_pods:
            val_status = "FAILED"
        elif any(not ds.is_healthy for ds in daemonset_list):
            val_status = "DEGRADED"
        elif total_gpus > 0:
            val_status = "PASSED"
        else:
            val_status = "WARNING"

        return OperatorAuditReport(
            cluster_connected=True,
            namespace=self.namespace,
            daemonsets=daemonset_list,
            failed_pods=failed_pods,
            gpu_nodes_count=gpu_nodes,
            total_allocatable_gpus=total_gpus,
            validation_status=val_status,
            recommendations=recommendations,
        )

    def audit(self) -> OperatorAuditReport:
        """Audits live cluster using kubectl."""
        if not self.kubectl:
            return OperatorAuditReport(
                cluster_connected=False,
                namespace=self.namespace,
                validation_status="DISCONNECTED",
                recommendations=["kubectl command not found on PATH."],
            )

        ds_json = self.run_kubectl(["get", "daemonsets", "-n", self.namespace])
        pod_json = self.run_kubectl(["get", "pods", "-n", self.namespace])
        node_json = self.run_kubectl(["get", "nodes"])

        return self.audit_from_json(ds_json, pod_json, node_json)

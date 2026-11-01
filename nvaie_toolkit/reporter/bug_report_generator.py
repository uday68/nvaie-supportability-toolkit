"""
NVIDIA Engineering Bug Report & RFE Generator
=============================================
Formats and synthesizes customer escalation telemetry, repro steps, environment matrices,
and diagnostic bundle manifests into standardized NVIDIA Engineering bug reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EnvironmentMatrix:
    csp_or_datacenter: str  # e.g. "AWS EKS", "Azure AKS", "GCP GKE", "On-Prem DGX SuperPOD"
    kubernetes_version: str
    gpu_model: str  # e.g. "NVIDIA H100 80GB HBM3 (8x SXM5)"
    driver_version: str
    cuda_version: str
    container_toolkit_version: str
    nvaie_release: str  # e.g. "NVAIE 4.1 / NGC 24.03"


@dataclass
class BugReportPayload:
    title: str
    severity: str  # "Sev1 - Critical Customer Outage", "Sev2 - Major Feature Blocked", "Sev3"
    impact_analysis: str
    environment: EnvironmentMatrix
    reproducible_steps: List[str]
    observed_behavior: str
    expected_behavior: str
    attached_diagnostic_bundle: str
    bundle_sha256: str
    immediate_mitigation: str
    proposed_engineering_fix: Optional[str] = None

    def to_markdown(self) -> str:
        steps_md = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(self.reproducible_steps))
        fix_md = f"### 6. Proposed Code-Level Fix / Upstream Patch\n{self.proposed_engineering_fix}\n" if self.proposed_engineering_fix else ""

        return f"""# [BUG REPORT] {self.title}

**Severity:** `{self.severity}`  
**Author:** NVIDIA AI Enterprise Supportability & Sustaining Engineering  

---

### 1. Executive Summary & Impact Analysis
{self.impact_analysis}

---

### 2. Environment Matrix
| Dimension | Specification |
|:---|:---|
| **Platform / CSP** | `{self.environment.csp_or_datacenter}` |
| **Kubernetes Version** | `{self.environment.kubernetes_version}` |
| **GPU Hardware** | `{self.environment.gpu_model}` |
| **NVIDIA Driver** | `{self.environment.driver_version}` |
| **CUDA Toolkit** | `{self.environment.cuda_version}` |
| **Container Toolkit** | `{self.environment.container_toolkit_version}` |
| **NVAIE Release** | `{self.environment.nvaie_release}` |

---

### 3. Minimal Step-by-Step Reproduction
{steps_md}

---

### 4. Observed vs Expected Behavior
* **Observed:**  
  {self.observed_behavior}

* **Expected:**  
  {self.expected_behavior}

---

### 5. Immediate Customer Mitigation Applied
{self.immediate_mitigation}

---

{fix_md}
### 7. Attached Supporting Diagnostic Artifacts
* **Bundle:** `{self.attached_diagnostic_bundle}`
* **SHA-256 Checksum:** `{self.bundle_sha256}`
"""

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "severity": self.severity,
            "impact_analysis": self.impact_analysis,
            "environment": {
                "csp_or_datacenter": self.environment.csp_or_datacenter,
                "kubernetes_version": self.environment.kubernetes_version,
                "gpu_model": self.environment.gpu_model,
                "driver_version": self.environment.driver_version,
                "cuda_version": self.environment.cuda_version,
                "container_toolkit_version": self.environment.container_toolkit_version,
                "nvaie_release": self.environment.nvaie_release,
            },
            "reproducible_steps": self.reproducible_steps,
            "observed_behavior": self.observed_behavior,
            "expected_behavior": self.expected_behavior,
            "immediate_mitigation": self.immediate_mitigation,
            "proposed_engineering_fix": self.proposed_engineering_fix,
            "attached_diagnostic_bundle": self.attached_diagnostic_bundle,
            "bundle_sha256": self.bundle_sha256,
        }


class BugReportGenerator:
    """Generates standardized engineering bug reports and RFEs."""

    @classmethod
    def generate_sample_sev1_report(cls) -> BugReportPayload:
        """Constructs an audited sample Sev1 bug report for Triton OOM / Xid 31 on Kubernetes."""
        env = EnvironmentMatrix(
            csp_or_datacenter="AWS EKS (us-east-1) / p5.48xlarge (8x H100 SXM5)",
            kubernetes_version="v1.28.6-eks",
            gpu_model="NVIDIA H100 80GB HBM3",
            driver_version="535.129.03 (Certified NVAIE Branch)",
            cuda_version="CUDA 12.2",
            container_toolkit_version="1.14.3",
            nvaie_release="NVAIE 4.1 (Triton 24.01-py3)",
        )

        return BugReportPayload(
            title="Triton Inference Server crashes with Xid 31 during Dynamic Batching on H100",
            severity="Sev1 - Critical Customer Outage",
            impact_analysis=(
                "Tier-1 financial customer reported a complete outage of their real-time fraud scoring API. "
                "All Triton inference pods entered CrashLoopBackOff with exit code 139 (SIGSEGV) and Xid 31 page faults "
                "in dmesg after sudden traffic surge to 2,500 RPS."
            ),
            environment=env,
            reproducible_steps=[
                "Deploy Triton Inference Server using Helm chart with `max_batch_size: 64` and `max_queue_delay_microseconds: 5000`.",
                "Warm up model with batch size 1 to initialize CUDA context.",
                "Send 100 concurrent requests with input sequence length 2048 using `nvaie-toolkit repro triton --concurrency 100`.",
                "Observe GPU memory allocation spike leading to illegal address read in dynamic batch concatenation kernel.",
            ],
            observed_behavior=(
                "Triton pod crashes with `SIGSEGV`. Kernel dmesg logs: "
                "`NVRM: Xid (PCI:0000:1b:00): 31, pid=4812, name=tritonserver, Page Fault on VA 0x7fa281000000`."
            ),
            expected_behavior=(
                "Triton dynamic batcher should reject excess requests with HTTP 429 / 503 or bound queue size "
                "without triggering GPU MMU page faults or crashing the container process."
            ),
            immediate_mitigation=(
                "Reduced `max_batch_size` in Triton `config.pbtxt` from 64 to 32 and enabled HorizontalPodAutoscaler (HPA) "
                "to scale from 4 to 12 Triton replicas. Service restored within 25 minutes of triage."
            ),
            proposed_engineering_fix=(
                "In Triton backend tensor concatenation logic, add bounds check before calling `cudaMemcpyAsync` "
                "when input buffer strides exceed aligned memory allocation boundaries."
            ),
            attached_diagnostic_bundle="nvaie-support-bundle-20261101-213000.tar.gz",
            bundle_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

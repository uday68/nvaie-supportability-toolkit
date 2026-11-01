"""
NVIDIA Xid Error Triage & Diagnostic Analyzer
=============================================
Parses Linux kernel ring buffers (dmesg) and syslogs for NVIDIA Driver Xid error events.
Maps error signatures to official NVIDIA root causes, hardware vs software attribution,
and immediate customer-facing mitigations.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class XidSeverity(str, Enum):
    SEV1_CRITICAL = "SEV1_CRITICAL"      # Node drop, hardware failure, uncorrectable ECC
    SEV2_HIGH = "SEV2_HIGH"              # GPU engine hung, preemption failure, app crash
    SEV3_MEDIUM = "SEV3_MEDIUM"          # Driver warnings, recoverable bus retries


@dataclass
class XidMetadata:
    code: int
    name: str
    category: str  # "Application / CUDA Bug", "Driver Bug", "Hardware / PCIe / Thermal", "ECC Memory"
    severity: XidSeverity
    description: str
    mitigation: str
    action_item: str


# Knowledge Base of standard NVIDIA Xid Error Codes
XID_KNOWLEDGE_BASE: Dict[int, XidMetadata] = {
    13: XidMetadata(
        code=13,
        name="Graphics Engine Exception",
        category="Application / CUDA Bug",
        severity=XidSeverity.SEV2_HIGH,
        description="A graphics or compute engine exception occurred, usually due to invalid shaders or corrupted command buffers.",
        mitigation="Restart container workload; inspect CUDA compute kernel parameters.",
        action_item="Capture compute shader / PTX trace.",
    ),
    31: XidMetadata(
        code=31,
        name="GPU Memory Page Fault (MMU Fault)",
        category="Application / CUDA Bug",
        severity=XidSeverity.SEV2_HIGH,
        description="GPU Memory Management Unit (MMU) fault. A CUDA kernel attempted an illegal out-of-bounds read/write or accessed an unmapped virtual address.",
        mitigation="Inspect CUDA / PyTorch application for illegal memory access or indexing bugs; run with compute-sanitizer.",
        action_item="Run customer application under 'compute-sanitizer --tool memcheck'.",
    ),
    32: XidMetadata(
        code=32,
        name="Corrupted Pushbuffer Stream",
        category="Driver / Hardware Bug",
        severity=XidSeverity.SEV2_HIGH,
        description="The pushbuffer data sent to the GPU was invalid or out of sync, indicating memory corruption or driver race condition.",
        mitigation="Restart application; verify driver compatibility with container CUDA runtime.",
        action_item="Check host RAM ECC and GPU driver version match.",
    ),
    43: XidMetadata(
        code=43,
        name="GPU Stopped Processing (Engine Timeout)",
        category="Application / Driver Hang",
        severity=XidSeverity.SEV2_HIGH,
        description="GPU engine timed out waiting for work to complete. Often caused by an infinite loop in a CUDA kernel or deadlocked synchronizations.",
        mitigation="Kill stuck PID; investigate kernel execution bounds and watchdog timer limits.",
        action_item="Inspect application threads; verify if kernel exceeds hardware watchdog timeout.",
    ),
    45: XidMetadata(
        code=45,
        name="Compute Preemption Failure",
        category="Driver / Compute Engine",
        severity=XidSeverity.SEV2_HIGH,
        description="The driver requested a compute engine preemption (context switch), but the GPU did not yield within the timeout window.",
        mitigation="Isolate long-running CUDA kernels; ensure cooperative groups and cooperative launch are used correctly.",
        action_item="Profile kernel execution time; avoid persistent kernels holding SMs indefinitely.",
    ),
    62: XidMetadata(
        code=62,
        name="Internal Microcode Breakpoint",
        category="Driver / Microcode",
        severity=XidSeverity.SEV2_HIGH,
        description="An internal microcode assertion or halt occurred inside the GPU microcontroller (Falcon / GSP firmware).",
        mitigation="Perform GPU reset (nvidia-smi -r); upgrade to certified NVAIE driver with latest GSP firmware.",
        action_item="Collect GSP firmware logs and escalate to NVIDIA Driver Engineering.",
    ),
    79: XidMetadata(
        code=79,
        name="GPU has Fallen off the Bus",
        category="Hardware / PCIe / Thermal",
        severity=XidSeverity.SEV1_CRITICAL,
        description="The GPU stopped responding to PCIe configuration cycles. Caused by PCIe link down, severe thermal trip, or power supply drop.",
        mitigation="Drain Kubernetes node immediately; warm/cold reboot server; inspect PCIe AER logs and server BMC/IPMI.",
        action_item="Check 'lspci -vvv' for AER errors; check server power supplies and PCIe riser seating.",
    ),
    92: XidMetadata(
        code=92,
        name="High Uncorrectable ECC Error Rate",
        category="ECC Memory / Hardware",
        severity=XidSeverity.SEV1_CRITICAL,
        description="Multiple uncorrectable double-bit DRAM ECC errors occurred, risking data corruption and silent calculation errors.",
        mitigation="Drain node and remove from Kubernetes scheduling. Hardware HBM/GDDR replacement required.",
        action_item="Engage datacenter hardware vendor for GPU RMA replacement.",
    ),
}


@dataclass
class DetectedXidEvent:
    timestamp: str
    pci_bus_id: str
    xid_code: int
    pid: Optional[int]
    process_name: Optional[str]
    raw_message: str
    metadata: Optional[XidMetadata] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "pci_bus_id": self.pci_bus_id,
            "xid_code": self.xid_code,
            "pid": self.pid,
            "process_name": self.process_name,
            "raw_message": self.raw_message,
            "metadata": {
                "name": self.metadata.name if self.metadata else "Unknown Xid",
                "category": self.metadata.category if self.metadata else "Unknown",
                "severity": self.metadata.severity.value if self.metadata else "SEV2_HIGH",
                "description": self.metadata.description if self.metadata else "Uncataloged Xid code.",
                "mitigation": self.metadata.mitigation if self.metadata else "Investigate system logs and run nvidia-bug-report.sh.",
                "action_item": self.metadata.action_item if self.metadata else "Contact NVIDIA AI Enterprise Support.",
            },
        }


# Regex to match: NVRM: Xid (PCI:0000:01:00): 31, pid=1234, name=python
XID_REGEX = re.compile(
    r"(?:\[\s*([0-9\.]+)\]\s*)?NVRM:\s*Xid\s*\((?:PCI:)?([0-9a-fA-F:\.]+)\):\s*(\d+)(?:,\s*pid=(\d+))?(?:,\s*name=([^\s,]+))?",
    re.IGNORECASE,
)


class XidAnalyzer:
    """Parses and triages NVIDIA Xid errors."""

    def parse_log_text(self, log_text: str) -> List[DetectedXidEvent]:
        """Parses log contents for NVIDIA Xid signatures."""
        events: List[DetectedXidEvent] = []
        for line in log_text.splitlines():
            match = XID_REGEX.search(line)
            if match:
                ts = match.group(1) or "unknown"
                pci_id = match.group(2)
                code = int(match.group(3))
                pid = int(match.group(4)) if match.group(4) else None
                pname = match.group(5) if match.group(5) else None

                metadata = XID_KNOWLEDGE_BASE.get(code)
                events.append(
                    DetectedXidEvent(
                        timestamp=ts,
                        pci_bus_id=pci_id,
                        xid_code=code,
                        pid=pid,
                        process_name=pname,
                        raw_message=line.strip(),
                        metadata=metadata,
                    )
                )
        return events

    def analyze_dmesg(self) -> List[DetectedXidEvent]:
        """Reads live kernel ring buffer via dmesg."""
        try:
            out = subprocess.check_output(["dmesg", "-T"], text=True, stderr=subprocess.DEVNULL)
            return self.parse_log_text(out)
        except Exception:
            return []

    def summarize(self, events: List[DetectedXidEvent]) -> Dict:
        """Returns structured triage summary."""
        sev1_count = sum(1 for e in events if e.metadata and e.metadata.severity == XidSeverity.SEV1_CRITICAL)
        sev2_count = sum(1 for e in events if e.metadata and e.metadata.severity == XidSeverity.SEV2_HIGH)

        return {
            "total_xid_events": len(events),
            "sev1_critical_count": sev1_count,
            "sev2_high_count": sev2_count,
            "events": [e.to_dict() for e in events],
            "requires_node_drain": sev1_count > 0,
        }

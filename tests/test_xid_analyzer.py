"""Unit tests for NVIDIA Kernel Xid Error Analyzer."""

from nvaie_toolkit.collectors.xid_analyzer import (
    XidAnalyzer,
    XidSeverity,
)

SAMPLE_DMESG_WITH_XIDS = """
[  120.450123] input: Power Button as /devices/LNXSYSTM:00/LNXPWRBN:00/input/input0
[  145.892011] NVRM: Xid (PCI:0000:1b:00): 31, pid=4812, name=tritonserver, Page Fault on VA 0x7fa281000000
[  210.119283] NVRM: Xid (PCI:0000:3a:00): 79, GPU has fallen off the bus.
[  340.551299] NVRM: Xid (PCI:0000:1b:00): 45, pid=5019, name=python, Compute Preemption Failure
"""


def test_xid_analyzer_extracts_correct_events():
    analyzer = XidAnalyzer()
    events = analyzer.parse_log_text(SAMPLE_DMESG_WITH_XIDS)

    assert len(events) == 3

    # Check Event 1: Xid 31
    e1 = events[0]
    assert e1.xid_code == 31
    assert e1.pci_bus_id == "0000:1b:00"
    assert e1.pid == 4812
    assert e1.process_name == "tritonserver"
    assert e1.metadata is not None
    assert e1.metadata.name == "GPU Memory Page Fault (MMU Fault)"
    assert e1.metadata.severity == XidSeverity.SEV2_HIGH

    # Check Event 2: Xid 79 (Sev1)
    e2 = events[1]
    assert e2.xid_code == 79
    assert e2.pci_bus_id == "0000:3a:00"
    assert e2.metadata.severity == XidSeverity.SEV1_CRITICAL
    assert "fallen off the bus" in e2.metadata.name.lower()

    # Check Event 3: Xid 45
    e3 = events[2]
    assert e3.xid_code == 45
    assert e3.pid == 5019
    assert e3.process_name == "python"


def test_xid_analyzer_summary():
    analyzer = XidAnalyzer()
    events = analyzer.parse_log_text(SAMPLE_DMESG_WITH_XIDS)
    summary = analyzer.summarize(events)

    assert summary["total_xid_events"] == 3
    assert summary["sev1_critical_count"] == 1
    assert summary["sev2_high_count"] == 2
    assert summary["requires_node_drain"] is True

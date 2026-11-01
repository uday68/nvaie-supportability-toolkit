"""
NVIDIA AI Enterprise (NVAIE) Supportability CLI
===============================================
Command-line interface for preflight validation, GPU Operator auditing,
diagnostic log bundle creation, Xid triage, and Triton reproducers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvaie_toolkit.collectors.log_bundle import DiagnosticBundleCollector
from nvaie_toolkit.collectors.xid_analyzer import XidAnalyzer
from nvaie_toolkit.core.gpu_inspector import GPUInspector
from nvaie_toolkit.core.k8s_inspector import K8sOperatorInspector
from nvaie_toolkit.core.system_inspector import SystemInspector
from nvaie_toolkit.reporter.bug_report_generator import BugReportGenerator
from nvaie_toolkit.repro_harness.triton_bench import TritonReproHarness
from nvaie_toolkit.validators.network_fabric import NetworkFabricValidator
from nvaie_toolkit.validators.preflight import CheckStatus, PreflightValidator


def print_banner():
    banner = r"""
  _   ___      _____ ____  ___      _        ___       _ _   _ _   
 | \ | \ \    / /_ _|  _ \|_ _|    / \      |_ _|___  (_) |_(_) |_ 
 |  \| |\ \  / / | || | | || |    / _ \ _____| |/ _ \ | | __| | __|
 | |\  | \ \/ /  | || |_| || |   / ___ \_____| | (_) || | |_| | |_ 
 |_| \_|  \__/  |___|____/|___| /_/   \_\   |___\___/ |_|\__|_|\__|
   NVIDIA AI Enterprise Supportability & Deployment Diagnostics Toolkit
"""
    print(banner, file=sys.stderr)


def cmd_check_preflight(args):
    validator = PreflightValidator(config_root=args.root)
    report = validator.run_all_checks()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n[*] NVAIE Preflight Readiness Status: [{report.overall_status.value}]")
        print(f"    Passed: {report.passed_count} | Warnings: {report.warning_count} | Failures: {report.failure_count}\n")
        print(f"{'RULE ID':<15} {'CATEGORY':<18} {'STATUS':<8} {'TITLE'}")
        print("-" * 75)
        for r in report.rules:
            status_color = r.status.value
            print(f"{r.rule_id:<15} {r.category:<18} [{status_color:<4}] {r.title}")
            if r.status != CheckStatus.PASS and r.remediation:
                print(f"  -> Actual: {r.actual}")
                print(f"  -> Remediation: {r.remediation}\n")

    return 0 if report.overall_status != CheckStatus.FAIL else 1


def cmd_check_operator(args):
    inspector = K8sOperatorInspector(namespace=args.namespace)
    report = inspector.audit()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n[*] NVIDIA GPU Operator Audit (Namespace: {report.namespace})")
        print(f"    Status: [{report.validation_status}] | Connected: {report.cluster_connected}")
        print(f"    GPU Nodes: {report.gpu_nodes_count} | Total Allocatable GPUs: {report.total_allocatable_gpus}\n")

        if report.daemonsets:
            print(f"{'DAEMONSET':<38} {'DESIRED':<8} {'READY':<8} {'HEALTH'}")
            print("-" * 65)
            for ds in report.daemonsets:
                h = "HEALTHY" if ds.is_healthy else "DEGRADED"
                print(f"{ds.name:<38} {ds.desired_number_scheduled:<8} {ds.number_ready:<8} {h}")

        if report.failed_pods:
            print("\n[!] Failed Pods in Operator Namespace:")
            for fp in report.failed_pods:
                print(f"    - Pod: {fp['pod']} ({fp['container']}) -> {fp['reason']}: {fp['message']}")

        if report.recommendations:
            print("\n[*] Recommendations:")
            for rec in report.recommendations:
                print(f"    - {rec}")

    return 0 if report.validation_status in ("PASSED", "WARNING", "DISCONNECTED") else 1


def cmd_check_fabric(args):
    validator = NetworkFabricValidator(sys_root=args.root)
    report = validator.validate()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n[*] High-Performance Network Fabric & RDMA Audit")
        print(f"    GPUDirect RDMA Supported: {report.gpudirect_rdma_supported}")
        print(f"    nvidia-peermem Module:    {report.nvidia_peermem_loaded}")
        print(f"    MTU Compliance (>=4096):  {report.mtu_compliant}\n")

        if report.rdma_interfaces:
            print(f"{'INTERFACE':<16} {'MTU':<8} {'RDMA CAPABLE':<14} {'STATE'}")
            print("-" * 50)
            for iface in report.rdma_interfaces:
                state = "UP" if iface.is_up else "DOWN"
                print(f"{iface.name:<16} {iface.mtu:<8} {str(iface.is_rdma):<14} {state}")

        if report.nccl_recommendations:
            print("\n[*] NCCL & Network Recommendations:")
            for rec in report.nccl_recommendations:
                print(f"    - {rec}")

    return 0


def cmd_triage_xid(args):
    analyzer = XidAnalyzer()
    if args.log_file:
        content = Path(args.log_file).read_text()
        events = analyzer.parse_log_text(content)
    else:
        events = analyzer.analyze_dmesg()

    summary = analyzer.summarize(events)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n[*] NVIDIA Kernel Xid Error Triage")
        print(f"    Total Xid Events: {summary['total_xid_events']}")
        print(f"    Sev1 Critical:    {summary['sev1_critical_count']}")
        print(f"    Sev2 High:        {summary['sev2_high_count']}")
        print(f"    Requires Drain:   {summary['requires_node_drain']}\n")

        if events:
            for ev in events:
                sev = ev.metadata.severity.value if ev.metadata else "UNKNOWN"
                name = ev.metadata.name if ev.metadata else "Uncataloged"
                print(f"[Xid {ev.xid_code}] ({ev.pci_bus_id}) [{sev}] {name}")
                print(f"  Raw: {ev.raw_message}")
                if ev.metadata:
                    print(f"  Root Cause: {ev.metadata.description}")
                    print(f"  Mitigation: {ev.metadata.mitigation}")
                    print(f"  Action:     {ev.metadata.action_item}\n")
        else:
            print("  No NVIDIA Xid errors detected in examined log source.")

    return 0 if summary["sev1_critical_count"] == 0 else 2


def cmd_collect_bundle(args):
    keywords = [args.sanitize_keyword] if args.sanitize_keyword else []
    collector = DiagnosticBundleCollector(output_dir=args.output, custom_keywords=keywords)

    dmesg_text = None
    if args.dmesg_file:
        dmesg_text = Path(args.dmesg_file).read_text()

    print("[*] Collecting system, GPU, operator, and network diagnostics...")
    result = collector.collect(include_k8s=not args.no_k8s, dmesg_content=dmesg_text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n[+] Diagnostic bundle generated successfully:")
        print(f"    Archive:  {result.archive_path}")
        print(f"    Size:     {result.size_bytes} bytes")
        print(f"    SHA-256:  {result.sha256_checksum}")
        print("    Files Included:")
        for f in result.files_included:
            print(f"      - {f}")
        print("\n[i] Bundle is sanitized and ready for upload to NVIDIA Support.")

    return 0


def cmd_repro_triton(args):
    harness = TritonReproHarness(base_url=args.url)
    print(f"[*] Executing Triton reproducer harness against {args.url} (Model: {args.model})...")

    result = harness.run_benchmark(
        model_name=args.model,
        total_requests=args.requests,
        concurrency=args.concurrency,
        simulate_if_offline=args.simulate,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n[*] Triton Benchmark / Repro Results:")
        print(f"    Target URL:      {result.target_url}")
        print(f"    Model Name:      {result.model_name}")
        print(f"    Throughput:      {result.throughput_rps} req/sec")
        print(f"    Success / Fail:  {result.successful_requests} / {result.failed_requests}")
        print(f"    Latency P50:     {result.latency_p50_ms:.2f} ms")
        print(f"    Latency P90:     {result.latency_p90_ms:.2f} ms")
        print(f"    Latency P99:     {result.latency_p99_ms:.2f} ms")
        if result.error_summary:
            print(f"    Errors:          {result.error_summary}")

    return 0 if result.failed_requests == 0 else 1


def cmd_report_sample(args):
    report = BugReportGenerator.generate_sample_sev1_report()
    content = report.to_markdown() if args.format == "md" else json.dumps(report.to_dict(), indent=2)

    if args.output:
        Path(args.output).write_text(content)
        print(f"[+] Bug report generated at {args.output}")
    else:
        print(content)

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="nvaie-toolkit",
        description="NVIDIA AI Enterprise (NVAIE) Supportability, Preflight Validation & Triage Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # check preflight
    p_preflight = subparsers.add_parser("check-preflight", help="Validate host OS, kernel, cgroups, and driver readiness")
    p_preflight.add_argument("--json", action="store_true", help="Output as JSON")
    p_preflight.add_argument("--root", default="/", help="Root directory for sysfs/proc inspection")
    p_preflight.set_defaults(func=cmd_check_preflight)

    # check operator
    p_operator = subparsers.add_parser("check-operator", help="Audit NVIDIA GPU Operator DaemonSets and pods")
    p_operator.add_argument("--namespace", default="gpu-operator", help="GPU Operator namespace")
    p_operator.add_argument("--json", action="store_true", help="Output as JSON")
    p_operator.set_defaults(func=cmd_check_operator)

    # check fabric
    p_fabric = subparsers.add_parser("check-fabric", help="Validate GPUDirect RDMA and InfiniBand/RoCE network fabric")
    p_fabric.add_argument("--json", action="store_true", help="Output as JSON")
    p_fabric.add_argument("--root", default="/", help="Root directory for sysfs inspection")
    p_fabric.set_defaults(func=cmd_check_fabric)

    # triage xid
    p_xid = subparsers.add_parser("triage-xid", help="Analyze dmesg / logs for NVIDIA Xid error codes")
    p_xid.add_argument("--log-file", help="Path to dmesg or syslog file")
    p_xid.add_argument("--json", action="store_true", help="Output as JSON")
    p_xid.set_defaults(func=cmd_triage_xid)

    # collect bundle
    p_bundle = subparsers.add_parser("collect-bundle", help="Gather, sanitize, and tar diagnostic logs")
    p_bundle.add_argument("--output", default="./diagnostics", help="Output directory for tarball")
    p_bundle.add_argument("--no-k8s", action="store_true", help="Skip Kubernetes API calls")
    p_bundle.add_argument("--dmesg-file", help="Custom dmesg log file to embed")
    p_bundle.add_argument("--sanitize-keyword", help="Custom organization name to sanitize")
    p_bundle.add_argument("--json", action="store_true", help="Output metadata as JSON")
    p_bundle.set_defaults(func=cmd_collect_bundle)

    # repro triton
    p_repro = subparsers.add_parser("repro-triton", help="Run Triton Inference Server benchmark and stress test")
    p_repro.add_argument("--url", default="http://localhost:8000", help="Triton base URL")
    p_repro.add_argument("--model", default="simple_model", help="Model name")
    p_repro.add_argument("--requests", type=int, default=50, help="Total requests to send")
    p_repro.add_argument("--concurrency", type=int, default=4, help="Concurrent client threads")
    p_repro.add_argument("--simulate", action="store_true", help="Run synthetic simulation if Triton is offline")
    p_repro.add_argument("--json", action="store_true", help="Output as JSON")
    p_repro.set_defaults(func=cmd_repro_triton)

    # report sample
    p_report = subparsers.add_parser("report-sample", help="Generate audited sample NVIDIA Sev1 Bug Report")
    p_report.add_argument("--format", choices=["md", "json"], default="md", help="Output format")
    p_report.add_argument("--output", help="Write report to file")
    p_report.set_defaults(func=cmd_report_sample)

    args = parser.parse_args()
    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    try:
        sys.exit(args.func(args))
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Architecture Specification: NVAIE Supportability Toolkit

```mermaid
graph TD
    subgraph "Core Subsystems"
        SI[System Inspector] --> PF[Preflight Validator]
        GI[GPU Inspector] --> PF
        GI --> BC[Bundle Collector]
        K8S[K8s Operator Inspector] --> BC
        XA[Xid Kernel Analyzer] --> BC
        NF[Network Fabric Validator] --> PF
    end

    subgraph "Enterprise Privacy & Triage"
        SAN[Data Sanitizer] --> BC
        BC --> TGZ["Sanitized Diagnostics Tarball (.tar.gz) + SHA-256"]
        XA --> XID_SUM["Structured Xid Triage Summary"]
    end

    subgraph "Reproduction & Sustaining Engineering"
        TB[Triton Repro Harness] --> BR[Bug Report Generator]
        FI[Fault Injector] --> BR
        BR --> BUG["NVIDIA Engineering Bug Report (.md / .json)"]
    end
```

## Architectural Principles

1. **First-Principles Reliability:** Operates independently of complex third-party dependencies using Python standard library primitives wherever possible.
2. **Zero-Trust Enterprise Sanitization:** Guarantees no customer IP addresses, credentials, NGC API tokens, or proprietary organizational identifiers leak into bug reports or diagnostic bundles.
3. **Cross-Platform Diagnostic Portability:** Functions identically across bare-metal DGX SuperPODs, enterprise hypervisors (vSphere / Tanzu), Cloud Service Providers (AWS EKS, Azure AKS, GCP GKE), and Windows test environments.
4. **Actionable Remediation Mapping:** Every failing rule or detected kernel Xid error is directly mapped to concrete operator remediation commands and hardware triage actions.

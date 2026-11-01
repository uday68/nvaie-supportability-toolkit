"""Unit tests for Sanitizer and Diagnostic Log Bundle Collector."""

import tarfile
import tempfile
from pathlib import Path

from nvaie_toolkit.collectors.log_bundle import DiagnosticBundleCollector
from nvaie_toolkit.collectors.sanitizer import DataSanitizer


def test_sanitizer_masks_confidential_patterns():
    sanitizer = DataSanitizer(custom_keywords=["AcmeCorp"])

    raw_text = (
        "Customer AcmeCorp report: Host 192.168.1.100 crashed while calling "
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token123.sig "
        "with key nvapi-abcdef1234567890abcdef1234567890 and password='mysecretpassword' "
        "contact admin@acmecorp.com"
    )

    clean = sanitizer.sanitize_text(raw_text)

    assert "192.168.1.100" not in clean
    assert "[SANITIZED_IP]" in clean
    assert "nvapi-" not in clean
    assert "[SANITIZED_NGC_API_KEY]" in clean
    assert "[SANITIZED_JWT_TOKEN]" in clean
    assert "mysecretpassword" not in clean
    assert "[SANITIZED_SECRET]" in clean
    assert "admin@acmecorp.com" not in clean
    assert "[SANITIZED_EMAIL]" in clean
    assert "AcmeCorp" not in clean
    assert "[SANITIZED_ORG]" in clean


def test_sanitizer_dict_recursive():
    sanitizer = DataSanitizer()
    payload = {
        "cluster_ip": "10.244.0.15",
        "nested": {
            "token": "Bearer eyJhbGciOi.payload.sig",
            "count": 42,
        },
        "list": ["172.16.5.20", "clean_text"],
    }

    sanitized = sanitizer.sanitize_dict(payload)
    assert sanitized["cluster_ip"] == "[SANITIZED_IP]"
    assert sanitized["nested"]["token"] == "Bearer [SANITIZED_JWT_TOKEN]"
    assert sanitized["nested"]["count"] == 42
    assert sanitized["list"][0] == "[SANITIZED_IP]"
    assert sanitized["list"][1] == "clean_text"


def test_diagnostic_bundle_creates_valid_tar_and_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        collector = DiagnosticBundleCollector(output_dir=tmp_dir)
        result = collector.collect(
            include_k8s=False,
            dmesg_content="[ 10.0 ] NVRM: Xid (PCI:0000:01:00): 31, pid=100, Page Fault",
        )

        assert Path(result.archive_path).exists()
        assert len(result.sha256_checksum) == 64
        assert result.size_bytes > 0

        # Verify tar contents
        with tarfile.open(result.archive_path, "r:gz") as tar:
            names = tar.getnames()
            assert any("system_telemetry.json" in n for n in names)
            assert any("gpu_status.json" in n for n in names)
            assert any("xid_triage.json" in n for n in names)
            assert any("preflight_audit.json" in n for n in names)

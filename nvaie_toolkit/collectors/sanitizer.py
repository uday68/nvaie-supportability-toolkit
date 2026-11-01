"""
Data Sanitizer & Anonymizer
===========================
Sanitizes sensitive enterprise data (IP addresses, corporate hostnames, NGC API keys,
bearer tokens, and cloud secrets) before packaging diagnostic bundles for NVIDIA Engineering.
"""

from __future__ import annotations

import re
from typing import Dict, List


IPV4_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
NGC_API_KEY_REGEX = re.compile(r"\bnvapi-[A-Za-z0-9_-]{30,}\b")
BEARER_TOKEN_REGEX = re.compile(r"\b(?:Bearer\s+)[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.?[A-Za-z0-9\-_=]*\b", re.IGNORECASE)
SECRET_ASSIGN_REGEX = re.compile(r"(?i)(password|token|secret|api_key|access_key)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.\$\/]{4,})['\"]?")


class DataSanitizer:
    """Enterprise privacy sanitizer for logs and diagnostic artifacts."""

    def __init__(self, custom_keywords: List[str] = None):
        self.custom_keywords = custom_keywords or []

    def sanitize_text(self, text: str) -> str:
        """Replaces sensitive patterns with sanitized placeholders."""
        if not text:
            return text

        # 1. Mask NGC API Keys
        text = NGC_API_KEY_REGEX.sub("[SANITIZED_NGC_API_KEY]", text)

        # 2. Mask Bearer / JWT Tokens
        text = BEARER_TOKEN_REGEX.sub("Bearer [SANITIZED_JWT_TOKEN]", text)

        # 3. Mask Key-Value Secrets (e.g. password=secret123)
        def secret_replace(match):
            key = match.group(1)
            return f"{key}=[SANITIZED_SECRET]"

        text = SECRET_ASSIGN_REGEX.sub(secret_replace, text)

        # 4. Mask Email Addresses
        text = EMAIL_REGEX.sub("[SANITIZED_EMAIL]", text)

        # 5. Mask non-loopback IPv4 Addresses
        def ip_replace(match):
            ip = match.group(0)
            if ip.startswith("127.") or ip == "0.0.0.0":
                return ip
            return "[SANITIZED_IP]"

        text = IPV4_REGEX.sub(ip_replace, text)

        # 6. Mask Custom Enterprise Keywords (e.g. corporate domain)
        for kw in self.custom_keywords:
            if kw:
                text = re.sub(re.escape(kw), "[SANITIZED_ORG]", text, flags=re.IGNORECASE)

        return text

    def sanitize_dict(self, data: Dict) -> Dict:
        """Recursively sanitizes string values in a dictionary."""
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, str):
                sanitized[k] = self.sanitize_text(v)
            elif isinstance(v, dict):
                sanitized[k] = self.sanitize_dict(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    self.sanitize_dict(item) if isinstance(item, dict)
                    else self.sanitize_text(item) if isinstance(item, str)
                    else item
                    for item in v
                ]
            else:
                sanitized[k] = v
        return sanitized

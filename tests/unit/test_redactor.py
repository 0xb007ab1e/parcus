"""Tests for secret/PII detection and masking."""

from __future__ import annotations

import pytest

from parcus.redact import Redactor, placeholder_for

ANTHROPIC = "sk-ant-api03-" + "A" * 24
OPENAI = "sk-" + "B" * 24
GITHUB = "ghp_" + "c" * 36
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
SLACK = "xoxb-0123456789-abcdefghij"
PRIVATE_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----"
BEARER = "Authorization: Bearer " + "d" * 30
ASSIGNMENT = 'password = "hunter2hunter2"'
EMAIL = "reach me at jane.doe@example.com please"


class TestHasSecret:
    @pytest.mark.parametrize(
        "text",
        [ANTHROPIC, OPENAI, GITHUB, AWS, SLACK, PRIVATE_KEY, BEARER, ASSIGNMENT],
    )
    def test_detects_credentials(self, text: str) -> None:
        assert Redactor().has_secret(text) is True

    def test_email_is_pii_not_a_secret(self) -> None:
        # PII must not trip the credential bypass (would gut cache hit-rate otherwise).
        assert Redactor().has_secret(EMAIL) is False

    def test_clean_text_has_no_secret(self) -> None:
        assert Redactor().has_secret("refactor the parser to handle empty input") is False


class TestRedact:
    def test_masks_and_removes_the_secret(self) -> None:
        redacted, report = Redactor().redact(f"key is {ANTHROPIC} ok")
        assert ANTHROPIC not in redacted
        assert placeholder_for("anthropic_key") in redacted
        assert report.has_secrets is True
        assert report.total == 1
        assert report.categories == ("anthropic_key",)

    def test_masks_email_but_reports_pii_category(self) -> None:
        redacted, report = Redactor().redact(EMAIL)
        assert "jane.doe@example.com" not in redacted
        assert report.categories == ("email",)

    def test_counts_multiple_and_sorts_categories(self) -> None:
        _, report = Redactor().redact(f"{AWS} and {ANTHROPIC} and {AWS}")
        assert report.total == 3
        assert report.categories == ("anthropic_key", "aws_access_key")

    def test_clean_text_is_unchanged(self) -> None:
        text = "just some ordinary prose"
        redacted, report = Redactor().redact(text)
        assert redacted == text
        assert report.has_secrets is False
        assert report.total == 0

    def test_anthropic_not_matched_as_openai(self) -> None:
        # The OpenAI rule excludes the sk-ant- prefix, so an Anthropic key is one category.
        _, report = Redactor().redact(ANTHROPIC)
        assert report.categories == ("anthropic_key",)

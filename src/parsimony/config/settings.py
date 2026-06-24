"""Typed, environment-driven configuration (12-factor), validated at startup (fail fast)."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings"]

# Values we REFUSE to bind (never public/all-interfaces — tailnet rule). The 0.0.0.0 literal
# here is a denylist entry, not a bind target.
_FORBIDDEN_BIND = {"0.0.0.0", "::", ""}  # noqa: S104  # nosec B104


class Settings(BaseSettings):
    """All runtime configuration, read from ``PARSIMONY_*`` env vars / ``.env``.

    Secure defaults: loopback bind, lossless on, lossy passes off, cache + redaction on.
    """

    model_config = SettingsConfigDict(
        env_prefix="PARSIMONY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8787
    tailnet_ip: str | None = None

    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"

    lossless: bool = True
    filler: bool = False
    learned: bool = False

    cache: bool = True
    cache_ttl_seconds: int = 86_400
    cache_path: str = ".parsimony/cache.sqlite"
    cache_nocache_patterns: str = ""
    salt: str = ""

    redact: bool = True
    log_level: str = "INFO"
    metrics: bool = True

    # Track B graph memory (off by default; injection changes the request, so it is opt-in).
    memory: bool = False
    memory_inject: bool = False
    memory_keep_recent: int = 4
    memory_retrieve: int = 3
    memory_min_messages: int = 8

    @field_validator("host")
    @classmethod
    def _reject_public_bind(cls, value: str) -> str:
        """Refuse to bind to a public/all-interfaces address (fail closed)."""
        if value.strip() in _FORBIDDEN_BIND:
            raise ValueError(
                f"refusing to bind host {value!r}: dev services must bind loopback/tailnet, "
                "never 0.0.0.0/public"
            )
        return value

    def nocache_patterns(self) -> list[str]:
        """Return the configured no-cache regex strings (comma-separated env → list)."""
        return [p.strip() for p in self.cache_nocache_patterns.split(",") if p.strip()]

"""Typed, environment-driven configuration (12-factor), validated at startup (fail fast)."""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from parsimony.quota import RateLimit

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
    metrics_path: str = ".parsimony/metrics.sqlite"
    invariant_sample_rate: float = 1.0  # fraction of requests to run the invariant self-check on

    # Graph memory (off by default; compaction changes the request, so it is opt-in).
    memory: bool = False
    memory_inject: bool = False  # Track B: compact via retrieval
    memory_summarize: bool = False  # Track C: replace older turns with a rolling summary
    memory_keep_recent: int = 4
    memory_retrieve: int = 3
    memory_summary_items: int = 5
    memory_min_messages: int = 8

    # Hosted/multi-tenant mode (off by default — local single-user is the default deployment).
    # When on, the tenant is derived server-side from the inbound credential and the response
    # cache is namespaced per tenant so tenants never share cached data.
    multi_tenant: bool = False
    # Optional edge authorization: comma-separated allow-list of permitted tenant ids (the
    # digests from `parsimony tenant-id`). Empty = open. Requires multi_tenant (validated below).
    allowed_tenants: str = ""
    # Optional per-tenant rate limit (token bucket). 0 = disabled. Keyed by the derived tenant id
    # (one shared bucket in single-tenant mode); burst 0 defaults capacity to one minute's worth.
    rate_limit_per_minute: float = 0.0
    rate_limit_burst: float = 0.0

    @field_validator("rate_limit_per_minute", "rate_limit_burst")
    @classmethod
    def _reject_negative_rate(cls, value: float) -> float:
        """Reject a negative rate/burst (fail fast on misconfig)."""
        if value < 0:
            raise ValueError("rate limit values must be >= 0 (0 disables limiting)")
        return value

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

    @model_validator(mode="after")
    def _require_multi_tenant_for_allow_list(self) -> Settings:
        """Reject an edge allow-list without multi-tenant mode (fail fast on misconfig).

        An allow-list keys on the credential-derived tenant id, which is only computed in
        multi-tenant mode; configuring one without ``multi_tenant`` would silently deny every
        request. Refuse to start instead.
        """
        if self.allowed_tenant_set() and not self.multi_tenant:
            raise ValueError(
                "PARSIMONY_ALLOWED_TENANTS requires PARSIMONY_MULTI_TENANT=true "
                "(the allow-list keys on the credential-derived tenant id)"
            )
        return self

    def nocache_patterns(self) -> list[str]:
        """Return the configured no-cache regex strings (comma-separated env → list)."""
        return [p.strip() for p in self.cache_nocache_patterns.split(",") if p.strip()]

    def allowed_tenant_set(self) -> frozenset[str]:
        """Return the configured edge-authorization allow-list (comma-separated env → set)."""
        return frozenset(t.strip() for t in self.allowed_tenants.split(",") if t.strip())

    def rate_limit(self) -> RateLimit | None:
        """Return the configured per-tenant rate limit, or ``None`` when disabled (rate 0)."""
        if self.rate_limit_per_minute <= 0:
            return None
        return RateLimit.per_minute(self.rate_limit_per_minute, self.rate_limit_burst)

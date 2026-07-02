"""Typed, environment-driven configuration (12-factor), validated at startup (fail fast)."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from parcus.quota import RateLimit

__all__ = ["Settings"]

_ENCRYPTION_KEY_LEN = 32  # AES-256


def _decode_key(encoded: str) -> bytes | None:
    """Decode a base64 key to raw bytes, or ``None`` if it isn't valid base64 of 32 bytes."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == _ENCRYPTION_KEY_LEN else None


# Values we REFUSE to bind (never public/all-interfaces — tailnet rule). The 0.0.0.0 literal
# here is a denylist entry, not a bind target.
_FORBIDDEN_BIND = {"0.0.0.0", "::", ""}  # noqa: S104  # nosec B104


class Settings(BaseSettings):
    """All runtime configuration, read from ``PARCUS_*`` env vars / ``.env``.

    Secure defaults: loopback bind, lossless on, lossy passes off, cache + redaction on.
    """

    model_config = SettingsConfigDict(
        env_prefix="PARCUS_",
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
    filler_aggressive: bool = False  # use the larger AGGRESSIVE_FILLERS set (validate offline)
    learned: bool = False  # Tier-2 local learned compressor (opt-in; needs the 'learned' extra)
    learned_ratio: float = 0.5  # target fraction of prose tokens to keep

    cache: bool = True
    cache_ttl_seconds: int = 86_400
    cache_path: str = ".parcus/cache.sqlite"
    cache_nocache_patterns: str = ""
    salt: str = ""

    # Provider prompt-cache injection (opt-in, OFF by default). When on, parcus adds a provider
    # cache breakpoint (Anthropic `cache_control`) to a large stable prefix so the provider serves
    # it from its prompt cache on the next turn — the dominant cost lever for tool/history-heavy
    # harnesses. Off until validated against a real provider key: a malformed breakpoint would 400
    # a live request. Only explicit-breakpoint providers act on it. See
    # docs/design/token-reduction-roadmap.md §2.1 (M1b).
    cache_inject: bool = False
    # When injecting, only do so once a prefix has been seen before (within the provider cache
    # window) — the ~1.25x cache-write premium is then only paid when a repeat (a ~0.1x read) is
    # likely, keeping injection never-cost-more in expectation (issue #56). Off = always inject on
    # first sighting.
    cache_inject_repeat_aware: bool = True
    # Canonicalize structured requests (tool_use/tool_result/image blocks, OpenAI tool calls) by
    # carrying those messages verbatim, instead of passing them through untouched (M1d slice 1).
    # Structured messages round-trip byte-for-byte and are left untouched by optimization here;
    # off by default. See docs/design/structured-content-parser.md.
    parse_structured: bool = False
    # Lossy: replace stale tool_result payloads (older than elide_keep_recent messages) with a
    # compact stub — the biggest history-token lever for tool-using harnesses. Off by default;
    # only affects structured turns (needs parse_structured) and, being lossy, should be validated
    # on the answer-preservation eval before enabling (M1d slice 3).
    elide_tool_results: bool = False
    elide_keep_recent: int = 4

    # Opt-in semantic (near-duplicate) cache — serve a cached response for a *similar* request.
    # OFF by default (trades correctness for tokens); validate the threshold with
    # `parcus eval --similarity` before raising it.
    similarity_cache: bool = False
    similarity_threshold: float = 0.97  # cosine; deliberately high (near-duplicate only)
    similarity_max_entries: int = 2048
    # Embedder: 'local' (sentence-transformers; the SAFE semantic default) | 'hashing' (lexical,
    # dep-free). The lexical embedder is UNSAFE for caching (can't tell "10 replicas" from "2");
    # using it requires the explicit acknowledgement below (ADR 0004).
    similarity_embedder: str = "local"
    similarity_allow_lexical: bool = False

    # At-rest cache encryption (opt-in). Key is base64(32 bytes) for AES-256, from env or a
    # keyfile (never in code/VCS). Enabling without a valid key fails closed at startup.
    cache_encryption: bool = False
    cache_encryption_key: SecretStr = SecretStr("")
    cache_encryption_keyfile: str = ""
    # Retired keys (comma-separated base64(32 bytes)) kept for DECRYPT-ONLY during a rotation
    # window: rotate by making the new key current and moving the old key here; drop it once old
    # entries have expired by TTL.
    cache_encryption_previous_keys: SecretStr = SecretStr("")
    # Crypto-shredding (right-to-erasure): comma-separated tenant ids whose per-tenant key is
    # withheld → their cached data is instantly inaccessible. Requires multi_tenant + encryption.
    cache_shredded_tenants: str = ""

    redact: bool = True
    log_level: str = "INFO"
    metrics: bool = True
    metrics_path: str = ".parcus/metrics.sqlite"
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
    # digests from `parcus tenant-id`). Empty = open. Requires multi_tenant (validated below).
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

    @field_validator("similarity_threshold")
    @classmethod
    def _reject_out_of_range_threshold(cls, value: float) -> float:
        """Require the cosine threshold in [0, 1] (fail fast on misconfig)."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")
        return value

    @field_validator("learned_ratio")
    @classmethod
    def _reject_out_of_range_ratio(cls, value: float) -> float:
        """Require the learned keep-ratio in (0, 1] (fail fast on misconfig)."""
        if not 0.0 < value <= 1.0:
            raise ValueError("learned_ratio must be in (0.0, 1.0]")
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
    def _require_valid_key_for_encryption(self) -> Settings:
        """Reject enabling at-rest encryption without a valid 32-byte key (fail closed)."""
        if self.cache_encryption and self.cache_encryption_key_bytes() is None:
            raise ValueError(
                "PARCUS_CACHE_ENCRYPTION=true requires a valid base64-encoded 32-byte key "
                "(PARCUS_CACHE_ENCRYPTION_KEY or PARCUS_CACHE_ENCRYPTION_KEYFILE)"
            )
        # Any configured previous (rotation) key must also be a valid 32-byte key.
        raw = self._previous_key_strings()
        if self.cache_encryption and len(self.cache_encryption_previous_key_bytes()) != len(raw):
            raise ValueError(
                "PARCUS_CACHE_ENCRYPTION_PREVIOUS_KEYS must all be valid base64-encoded "
                "32-byte keys"
            )
        # Crypto-shredding needs per-tenant DEKs, which exist only under multi_tenant + encryption.
        if self.cache_shredded_tenant_set() and not (self.cache_encryption and self.multi_tenant):
            raise ValueError(
                "PARCUS_CACHE_SHREDDED_TENANTS requires PARCUS_CACHE_ENCRYPTION=true and "
                "PARCUS_MULTI_TENANT=true (shredding withholds a per-tenant encryption key)"
            )
        return self

    @model_validator(mode="after")
    def _require_safe_similarity_embedder(self) -> Settings:
        """Default the similarity cache to the safe local embedder; gate the unsafe lexical one.

        The lexical ('hashing') embedder can't distinguish requests differing only in numbers or
        entities, so it false-hits — unsafe for serving cached responses (ADR 0004). Using it for
        the similarity cache therefore requires an explicit risk acknowledgement; fail fast on
        misconfiguration or an unknown embedder.
        """
        if not self.similarity_cache:
            return self
        if self.similarity_embedder not in ("local", "hashing"):
            raise ValueError("similarity_embedder must be 'local' or 'hashing'")
        if self.similarity_embedder == "hashing" and not self.similarity_allow_lexical:
            raise ValueError(
                "the lexical 'hashing' embedder is UNSAFE for the similarity cache (it can't "
                "distinguish requests differing only in numbers/entities — ADR 0004); set "
                "similarity_embedder=local (install the 'embeddings' extra) or, to accept the "
                "risk, set similarity_allow_lexical=true"
            )
        return self

    @model_validator(mode="after")
    def _require_multi_tenant_for_allow_list(self) -> Settings:
        """Reject an edge allow-list without multi-tenant mode (fail fast on misconfig).

        An allow-list keys on the credential-derived tenant id, which is only computed in
        multi-tenant mode; configuring one without ``multi_tenant`` would silently deny every
        request. Refuse to start instead.
        """
        if self.allowed_tenant_set() and not self.multi_tenant:
            raise ValueError(
                "PARCUS_ALLOWED_TENANTS requires PARCUS_MULTI_TENANT=true "
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

    def cache_encryption_key_bytes(self) -> bytes | None:
        """Resolve the at-rest encryption key to 32 raw bytes, or ``None`` if unset/invalid.

        Prefers the keyfile (base64 text) over the inline key; both decode base64 to exactly 32
        bytes (AES-256). Never logs the key. Returns ``None`` on any decode/length/read failure
        so the caller can fail closed.
        """
        encoded: str | None = None
        if self.cache_encryption_keyfile:
            try:
                encoded = Path(self.cache_encryption_keyfile).read_text(encoding="utf-8").strip()
            except OSError:
                return None
        elif self.cache_encryption_key.get_secret_value():
            encoded = self.cache_encryption_key.get_secret_value()
        return _decode_key(encoded) if encoded else None

    def _previous_key_strings(self) -> list[str]:
        """Return the configured previous-key base64 strings (comma-separated → list)."""
        return [
            k.strip()
            for k in self.cache_encryption_previous_keys.get_secret_value().split(",")
            if k.strip()
        ]

    def cache_encryption_previous_key_bytes(self) -> tuple[bytes, ...]:
        """Resolve the retired (rotation) keys to 32-byte values, dropping any that don't decode."""
        return tuple(
            raw for s in self._previous_key_strings() if (raw := _decode_key(s)) is not None
        )

    def cache_shredded_tenant_set(self) -> frozenset[str]:
        """Return the set of crypto-shredded tenant ids (comma-separated env → set)."""
        return frozenset(t.strip() for t in self.cache_shredded_tenants.split(",") if t.strip())

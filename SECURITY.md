# Security Policy

## Reporting a vulnerability

`parcus` is a local development tool that, when running, **holds your model-provider API
keys** and may cache prompt/response content. Treat security issues seriously.

- Report privately via a GitHub Security Advisory (preferred) or by opening an issue marked
  **security** with minimal detail and a request for a private channel.
- Please include: affected version/commit, reproduction, impact, and any PoC.
- Expect an acknowledgement within a few days; we triage on the severity taxonomy in the
  master ruleset (Critical/High/Medium/Low) and fix under embargo before public disclosure.

## Scope & threat model

The proxy's trust boundaries and STRIDE analysis live in
[`docs/security/threat-model.md`](docs/security/threat-model.md). Highlights:

- **Secrets:** provider API keys come from env/secret store, are never logged, never cached,
  never committed. Report any leak path as Critical.
- **Cache confidentiality:** cached data is classified confidential — redacted, TTL-bound,
  opt-out-able. Report any unredacted-persist or cross-context leak.
- **Binding:** the proxy must bind loopback + tailnet only, never `0.0.0.0`/public.
- **Fail open for availability, fail closed for security:** report any path where an
  optimization decision can alter a model result incorrectly, or where a security check is
  bypassable.

## Supported versions

Pre-1.0: only the latest `main` is supported. Once released, see the changelog for the
support window.

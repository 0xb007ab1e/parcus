"""parsimony — a local-first, token-thrift inference proxy for agentic harnesses.

The package is organised as a functional core (pure, provider-agnostic transform and
cache-decision logic) wrapped by an imperative shell (HTTP ingress, upstream forwarding,
persistence). See ``PLAN.md`` and ``docs/adr/0001-proxy-architecture-and-fail-open.md``.

Design tenets (enforced throughout):

* **Fail open** — on any uncertainty in the optimization path, forward the original,
  unmodified request upstream. The proxy must never break a harness or change a result.
* **Correctness over tokens** — no lossy transform ships without a measured no-regression
  result on the eval set.
* **Local-only models** — never issue an outbound inference call to save one.
"""

__version__ = "0.0.1"

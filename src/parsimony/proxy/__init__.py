"""The imperative shell: HTTP ingress, dialect handling, and upstream forwarding.

The pure core (model, compress, cache, redact) knows nothing about this package; wiring
happens at the composition root (:mod:`parsimony.cli`).
"""

from parsimony.proxy.app import create_app
from parsimony.proxy.engine import EngineConfig, ProxyEngine, ProxyResult

__all__ = ["EngineConfig", "ProxyEngine", "ProxyResult", "create_app"]

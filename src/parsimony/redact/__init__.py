"""Secret/PII redaction for logs, telemetry, and derived (graph) content.

Redaction is applied to data that is *persisted or logged* — never to the request forwarded
upstream nor to a replayed cache response, which must stay byte-for-byte intact. Credential
detection (:meth:`Redactor.has_secret`) additionally powers the cache no-cache bypass.
"""

from parsimony.redact.redactor import Redactor, placeholder_for

__all__ = ["Redactor", "placeholder_for"]

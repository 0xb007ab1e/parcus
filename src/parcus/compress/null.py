"""A no-op compressor for when compression is disabled."""

from __future__ import annotations

from parcus.model import CanonicalRequest, CompressionStats

__all__ = ["NullCompressor"]


class NullCompressor:
    """Returns the request unchanged. Implements :class:`parcus.ports.CompressorPort`."""

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Return ``request`` unchanged with no statistics."""
        return request, ()

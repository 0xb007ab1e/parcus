"""Compose multiple compression passes into one (e.g. lossless then filler)."""

from __future__ import annotations

from collections.abc import Iterable

from parcus.model import CanonicalRequest, CompressionStats
from parcus.ports import CompressorPort

__all__ = ["ChainCompressor"]


class ChainCompressor:
    """Run compressors in sequence, threading the request and concatenating stats.

    Implements :class:`parcus.ports.CompressorPort`. Each constituent pass already fails
    open individually, so the chain inherits that behaviour.

    Args:
        compressors: The passes to apply, in order.
    """

    def __init__(self, compressors: Iterable[CompressorPort]) -> None:
        """Store the ordered passes."""
        self._compressors = tuple(compressors)

    def compress(
        self, request: CanonicalRequest
    ) -> tuple[CanonicalRequest, tuple[CompressionStats, ...]]:
        """Apply every pass in order and return the final request with all per-pass stats."""
        current = request
        stats: list[CompressionStats] = []
        for compressor in self._compressors:
            current, pass_stats = compressor.compress(current)
            stats.extend(pass_stats)
        return current, tuple(stats)

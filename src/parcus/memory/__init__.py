"""Graph memory (Tracks B & C of the plan): a property graph of prior context.

Foundation slice — model, in-memory store, model-free lexical ingest/retrieval, exposed via
:class:`GraphMemory` (a :class:`parcus.ports.MemoryPort`). SQLite + sqlite-vec persistence,
optional local-embedding retrieval, conversation compaction, and engine wiring are later,
eval-gated slices. See ``docs/adr/0002-graph-memory.md``.
"""

from parcus.memory.compaction import compact_by_summary, compact_with_memory
from parcus.memory.embedding import (
    EmbedderPort,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from parcus.memory.graph import GraphMemory
from parcus.memory.ingest import ingest_request
from parcus.memory.model import MemoryEdge, MemoryKind, MemoryNode, node_id
from parcus.memory.provider import (
    MemoryProvider,
    PerTenantMemoryProvider,
    SharedMemoryProvider,
)
from parcus.memory.retrieval import jaccard, relevant
from parcus.memory.sqlite_store import SqliteGraphStore
from parcus.memory.store import GraphStore, InMemoryGraphStore
from parcus.memory.summary import ExtractiveSummarizer, LLMSummarizer, Summarizer
from parcus.memory.terms import extract_terms

__all__ = [
    "EmbedderPort",
    "ExtractiveSummarizer",
    "GraphMemory",
    "GraphStore",
    "HashingEmbedder",
    "InMemoryGraphStore",
    "LLMSummarizer",
    "MemoryEdge",
    "MemoryKind",
    "MemoryNode",
    "MemoryProvider",
    "PerTenantMemoryProvider",
    "SentenceTransformerEmbedder",
    "SharedMemoryProvider",
    "SqliteGraphStore",
    "Summarizer",
    "compact_by_summary",
    "compact_with_memory",
    "cosine",
    "extract_terms",
    "ingest_request",
    "jaccard",
    "node_id",
    "relevant",
]

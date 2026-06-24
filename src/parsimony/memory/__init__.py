"""Graph memory (Tracks B & C of the plan): a property graph of prior context.

Foundation slice — model, in-memory store, model-free lexical ingest/retrieval, exposed via
:class:`GraphMemory` (a :class:`parsimony.ports.MemoryPort`). SQLite + sqlite-vec persistence,
optional local-embedding retrieval, conversation compaction, and engine wiring are later,
eval-gated slices. See ``docs/adr/0002-graph-memory.md``.
"""

from parsimony.memory.compaction import compact_by_summary, compact_with_memory
from parsimony.memory.embedding import (
    EmbedderPort,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from parsimony.memory.graph import GraphMemory
from parsimony.memory.ingest import ingest_request
from parsimony.memory.model import MemoryEdge, MemoryKind, MemoryNode, node_id
from parsimony.memory.retrieval import jaccard, relevant
from parsimony.memory.sqlite_store import SqliteGraphStore
from parsimony.memory.store import GraphStore, InMemoryGraphStore
from parsimony.memory.summary import ExtractiveSummarizer, LLMSummarizer, Summarizer
from parsimony.memory.terms import extract_terms

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
    "SentenceTransformerEmbedder",
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

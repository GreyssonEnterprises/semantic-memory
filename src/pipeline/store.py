"""Chroma vector store manager with filesystem-level agent isolation."""

import chromadb
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import json

# Default base path for vector stores — override with SEMANTIC_MEMORY_STORE env var
import os
_store_env = os.environ.get("SEMANTIC_MEMORY_STORE")
DEFAULT_VECTOR_BASE = Path(_store_env) if _store_env else (Path.home() / "clawd" / "projects" / "semantic-memory" / "data" / "vectors")

# Agent namespaces — filesystem-isolated
AGENTS = ("bob", "patterson", "dean", "shared")

# Source type weights for retrieval scoring
SOURCE_WEIGHTS = {
    "vault": 2.0,       # memory/vault/* — critical, never-forget
    "procedure": 1.5,   # memory/procedures/* — learned workflows
    "graph": 1.3,       # memory/graph/* — entity knowledge
    "daily": 1.0,       # memory/*.md (daily logs) — standard
    "session_summary": 0.9,  # session-level summaries
    "session_segment": 0.8,  # segment-level chunks — high volume
}


class VectorStore:
    """Manages per-agent Chroma collections with filesystem isolation."""

    def __init__(self, agent: str, base_path: Optional[Path] = None):
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent '{agent}'. Must be one of {AGENTS}")

        self.agent = agent
        self.base_path = (base_path or DEFAULT_VECTOR_BASE) / agent
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Each agent gets its own persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.base_path))

        # Two collections per agent: segments and session summaries
        self.segments = self.client.get_or_create_collection(
            name=f"{agent}_segments",
            metadata={"hnsw:space": "cosine"},
        )
        self.summaries = self.client.get_or_create_collection(
            name=f"{agent}_summaries",
            metadata={"hnsw:space": "cosine"},
        )

    def add_segment(
        self,
        segment_id: str,
        text: str,
        embedding: list[float],
        *,
        session_id: str,
        date: str,
        domain: str = "general",
        topic: str = "",
        tone: str = "",
        participants: Optional[list[str]] = None,
        source_type: str = "session_segment",
        unresolved_threads: Optional[list[str]] = None,
        status: str = "",  # decided|open|interrupted
    ) -> None:
        """Add a topic segment embedding to the segments collection."""
        metadata = {
            "session_id": session_id,
            "date": date,
            "domain": domain,
            "topic": topic,
            "tone": tone,
            "participants": json.dumps(participants or []),
            "unresolved_threads": json.dumps(unresolved_threads or []),
            "status": status,
            "source_type": source_type,
            "source_weight": SOURCE_WEIGHTS.get(source_type, 0.8),
            "agent": self.agent,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.segments.upsert(
            ids=[segment_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def add_session_summary(
        self,
        session_id: str,
        summary_text: str,
        embedding: list[float],
        *,
        date: str,
        domain: str = "general",
        topics: Optional[list[str]] = None,
        emotional_tenor: str = "",
        decisions: Optional[list[str]] = None,
        outcome: str = "",
        participants: Optional[list[str]] = None,
        unresolved_threads: Optional[list[str]] = None,
    ) -> None:
        """Add a session-level summary embedding to the summaries collection."""
        metadata = {
            "session_id": session_id,
            "date": date,
            "domain": domain,
            "topics": json.dumps(topics or []),
            "emotional_tenor": emotional_tenor,
            "decisions": json.dumps(decisions or []),
            "unresolved_threads": json.dumps(unresolved_threads or []),
            "outcome": outcome,
            "participants": json.dumps(participants or []),
            "source_type": "session_summary",
            "source_weight": SOURCE_WEIGHTS["session_summary"],
            "agent": self.agent,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.summaries.upsert(
            ids=[session_id],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[metadata],
        )

    def search_segments(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """Search segment-level embeddings (specific recall)."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self.segments.query(**kwargs)

    def search_summaries(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """Search session-level summary embeddings (pattern matching)."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self.summaries.query(**kwargs)

    def has_session(self, session_id: str) -> bool:
        """Check if a session has already been ingested (has a summary)."""
        try:
            result = self.summaries.get(ids=[session_id])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def ingested_session_ids(self) -> set[str]:
        """Return set of all session IDs that have been ingested."""
        try:
            result = self.summaries.get(include=[])
            return set(result["ids"])
        except Exception:
            return set()

    def stats(self) -> dict:
        """Return collection sizes."""
        return {
            "agent": self.agent,
            "path": str(self.base_path),
            "segments_count": self.segments.count(),
            "summaries_count": self.summaries.count(),
        }

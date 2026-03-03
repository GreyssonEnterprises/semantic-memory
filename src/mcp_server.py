#!/usr/bin/env python3
"""
Semantic Memory MCP Server — FastMCP stdio transport.

Exposes the semantic memory pipeline (ChromaDB + Ollama embeddings) to
ephemeral agents via MCP protocol. Wraps existing retrieval and ingest
code without modification.

Usage:
    uv run --directory ~/clawd/projects/semantic-memory src/mcp_server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure project root is on sys.path for src.* imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("semantic-memory-mcp")

mcp = FastMCP(
    name="openclaw-semantic-memory",
    instructions="Circle Zero Semantic Memory — semantic search across agent memory with cognitive scoring. Use search_memory for semantic retrieval, store_memory to persist new facts, list_recent for episodic logs, get_procedures for workflow guides.",
)

# ---------------------------------------------------------------------------
# Default agent and memory paths
# ---------------------------------------------------------------------------
DEFAULT_AGENT = os.environ.get("SEMANTIC_MEMORY_AGENT", "bob")
MEMORY_BASE = Path(os.environ.get("OPENCLAW_MEMORY_PATH", Path.home() / "clawd" / "memory"))


def _get_store(agent: str | None = None):
    """Lazy-load VectorStore to avoid import errors if ChromaDB is down."""
    from src.pipeline.store import VectorStore

    return VectorStore(agent=agent or DEFAULT_AGENT)


# ---------------------------------------------------------------------------
# Tool: search_memory
# ---------------------------------------------------------------------------
@mcp.tool()
def search_memory(
    query: str,
    scope: str = "all",
    domain: str | None = None,
    limit: int = 10,
) -> str:
    """Search semantic memory for past conversations, decisions, and context.

    Args:
        query: Natural language search query (e.g. "what did we decide about auth?")
        scope: Search scope — "segments" (precise), "summaries" (thematic), or "all" (blended, default)
        domain: Optional domain filter (redhat, trading, ecommerce, consulting, security-research, health, household)
        limit: Max results to return (default 10)

    Returns:
        JSON array of scored results with text, metadata, and relevance scores.
    """
    mode_map = {"segments": "specific", "summaries": "pattern", "all": "blended"}
    mode = mode_map.get(scope, "blended")

    try:
        from src.retrieval.recall import search

        results = search(
            query=query,
            agent=DEFAULT_AGENT,
            mode=mode,
            top_k=min(limit, 50),
            domain=domain,
        )

        output = []
        for r in results:
            output.append({
                "score": round(r["score"], 4),
                "distance": round(r["distance"], 4),
                "source": r["source"],
                "date": r["metadata"].get("date", ""),
                "domain": r["metadata"].get("domain", ""),
                "topic": r["metadata"].get("topic", ""),
                "session_id": r["metadata"].get("session_id", ""),
                "text": r["doc"][:800],
            })

        if not output:
            return json.dumps({"results": [], "message": "No matching memories found."})

        return json.dumps({"results": output, "count": len(output)}, indent=2)

    except Exception as e:
        logger.exception("search_memory failed")
        return json.dumps({"error": str(e), "hint": "ChromaDB or Ollama may be unavailable"})


# ---------------------------------------------------------------------------
# Tool: store_memory
# ---------------------------------------------------------------------------
@mcp.tool()
def store_memory(
    content: str,
    type: str = "auto",
    domain: str | None = None,
) -> str:
    """Store a new memory segment into the vector store.

    Use this to persist important context, decisions, or facts that should
    be retrievable in future sessions.

    Args:
        content: The text content to store (a fact, decision, observation, etc.)
        type: Memory type — "segment", "summary", or "auto" (default: auto → segment)
        domain: Optional domain tag (redhat, trading, ecommerce, consulting, security-research, health, household)

    Returns:
        JSON confirmation with the stored segment ID.
    """
    try:
        from src.pipeline.embedder import embed_text
        import uuid

        store = _get_store()
        embedding = embed_text(content)

        segment_id = f"mcp-{uuid.uuid4().hex[:12]}"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        store.add_segment(
            segment_id=segment_id,
            text=content,
            embedding=embedding,
            session_id=f"mcp-direct-{today}",
            date=today,
            domain=domain or "general",
            topic="",
            source_type="session_segment",
        )

        return json.dumps({
            "status": "stored",
            "segment_id": segment_id,
            "date": today,
            "domain": domain or "general",
            "chars": len(content),
        })

    except Exception as e:
        logger.exception("store_memory failed")
        return json.dumps({"error": str(e), "hint": "ChromaDB or Ollama may be unavailable"})


# ---------------------------------------------------------------------------
# Tool: list_recent
# ---------------------------------------------------------------------------
@mcp.tool()
def list_recent(days: int = 7) -> str:
    """List recent daily memory log files.

    Reads the memory/YYYY-MM-DD.md episodic log files to show what happened
    in recent sessions.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        JSON array of recent memory entries with dates and content previews.
    """
    try:
        entries = []
        today = datetime.now(timezone.utc)

        for i in range(days):
            from datetime import timedelta

            day = today - timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            filepath = MEMORY_BASE / f"{date_str}.md"

            if filepath.exists():
                content = filepath.read_text(encoding="utf-8", errors="replace")
                entries.append({
                    "date": date_str,
                    "file": str(filepath),
                    "size_chars": len(content),
                    "preview": content[:500],
                })

        if not entries:
            return json.dumps({"entries": [], "message": f"No memory logs found in last {days} days."})

        return json.dumps({"entries": entries, "count": len(entries)}, indent=2)

    except Exception as e:
        logger.exception("list_recent failed")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool: get_procedures
# ---------------------------------------------------------------------------
@mcp.tool()
def get_procedures(query: str) -> str:
    """Search procedural memory for learned workflows and how-to guides.

    Scans memory/procedures/ directory for markdown files matching the query.

    Args:
        query: Search term to match against procedure filenames and content.

    Returns:
        JSON array of matching procedures with content.
    """
    try:
        procedures_dir = MEMORY_BASE / "procedures"
        if not procedures_dir.exists():
            return json.dumps({"procedures": [], "message": "No procedures directory found."})

        query_lower = query.lower()
        matches = []

        for md_file in sorted(procedures_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8", errors="replace")
            name = md_file.stem

            # Match against filename or content
            if query_lower in name.lower() or query_lower in content.lower():
                matches.append({
                    "name": name,
                    "file": str(md_file),
                    "content": content[:2000],
                    "size_chars": len(content),
                })

        if not matches:
            # List available procedures as hints
            available = [f.stem for f in procedures_dir.glob("*.md")]
            return json.dumps({
                "procedures": [],
                "message": f"No procedures matching '{query}'.",
                "available": available[:20],
            })

        return json.dumps({"procedures": matches, "count": len(matches)}, indent=2)

    except Exception as e:
        logger.exception("get_procedures failed")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()

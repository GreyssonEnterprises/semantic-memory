#!/usr/bin/env python3
"""
Context loader for semantic memory — callable from agent sessions.

Provides a simple function that agents can call to retrieve relevant
context from their semantic memory store during conversations.

Usage from Python:
    from src.retrieval.context_loader import recall
    context = recall("what did we decide about the chunking strategy", agent="bob")

Usage from CLI:
    pai-memory-recall "query" --agent bob --context
    # Returns formatted context block suitable for injection into agent prompts
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.recall import search


def recall(
    query: str,
    agent: str = "bob",
    mode: str = "blended",
    top_k: int = 3,
    min_score: float = 0.15,
    domain: str | None = None,
) -> str:
    """
    Retrieve relevant context from semantic memory.
    
    Returns a formatted string suitable for injection into agent context.
    Returns empty string if no sufficiently relevant results found.
    """
    results = search(
        query=query,
        agent=agent,
        mode=mode,
        top_k=top_k,
        domain=domain,
    )
    
    # Filter by minimum score
    results = [r for r in results if r["score"] >= min_score]
    
    if not results:
        return ""
    
    # Format for agent context
    lines = ["[Semantic Memory Recall]"]
    for r in results:
        date = r["metadata"].get("date", "?")
        topic = r["metadata"].get("topic", "")
        score = r["score"]
        source = r["source"]
        
        # Truncate text for context injection (don't overwhelm the prompt)
        text = r["doc"][:400].strip()
        if len(r["doc"]) > 400:
            text += "..."
        
        lines.append(f"  [{date}] (relevance={score:.2f}, {source})")
        if topic:
            lines.append(f"  Topic: {topic}")
        lines.append(f"  {text}")
        lines.append("")
    
    lines.append("[/Semantic Memory Recall]")
    return "\n".join(lines)


def recall_json(
    query: str,
    agent: str = "bob",
    mode: str = "blended",
    top_k: int = 3,
    min_score: float = 0.15,
    domain: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant context as structured data.
    
    Returns list of dicts with score, date, topic, text, source.
    """
    results = search(
        query=query,
        agent=agent,
        mode=mode,
        top_k=top_k,
        domain=domain,
    )
    
    return [
        {
            "score": round(r["score"], 4),
            "date": r["metadata"].get("date", ""),
            "topic": r["metadata"].get("topic", ""),
            "domain": r["metadata"].get("domain", ""),
            "source": r["source"],
            "text": r["doc"][:500],
            "session_id": r["metadata"].get("session_id", ""),
        }
        for r in results
        if r["score"] >= min_score
    ]


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) < 2:
        print("Usage: context_loader.py <query> [agent]")
        sys.exit(1)
    
    query = sys.argv[1]
    agent = sys.argv[2] if len(sys.argv) > 2 else "bob"
    
    result = recall(query, agent=agent)
    if result:
        print(result)
    else:
        print("(no relevant memories found)")

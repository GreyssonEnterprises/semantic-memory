#!/usr/bin/env python3
"""
pai-memory-recall — Semantic memory retrieval CLI.

Searches embedded session segments and summaries using vector similarity.
Supports three retrieval modes:
  - specific:  Segment-level search (precise recall of conversations)
  - pattern:   Summary-level search (find sessions by theme/vibe)
  - blended:   Both, merged and re-ranked (default)

Usage:
    pai-memory-recall "what were we working on last Tuesday"
    pai-memory-recall --mode specific "Grimm's Mounjaro dosage"
    pai-memory-recall --mode pattern --domain redhat "customer escalation patterns"
    pai-memory-recall --agent bob --top 5 "semantic memory architecture decisions"
    pai-memory-recall --json "NFS mount issues"

Time-weighted scoring: recent results rank higher via 1/(1+log(age+1)).
"""

import argparse
import json
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.embedder import embed_text, time_weight
from src.pipeline.store import VectorStore


def parse_date(date_str: str) -> datetime | None:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def compute_final_score(
    distance: float,
    date_str: str,
    source_weight: float = 1.0,
) -> float:
    """
    Combine vector similarity, time decay, and source weight into final score.
    
    Chroma returns distances (lower = better). Convert to similarity first.
    similarity = 1 / (1 + distance)  [for L2 distance]
    
    Final = similarity * time_weight * source_weight
    """
    similarity = 1.0 / (1.0 + distance)
    
    # Time weight
    date = parse_date(date_str)
    if date:
        age_days = (datetime.now(timezone.utc) - date).days
        tw = time_weight(max(0, age_days))
    else:
        tw = 0.5  # Unknown date gets middle weight
    
    return similarity * tw * source_weight


def format_result(
    rank: int,
    doc: str,
    metadata: dict,
    score: float,
    distance: float,
    verbose: bool = False,
) -> str:
    """Format a single result for human-readable output."""
    lines = []
    
    # Header
    date = metadata.get("date", "?")
    domain = metadata.get("domain", "general")
    topic = metadata.get("topic", "")
    session_id = metadata.get("session_id", "?")[:12]
    source_type = metadata.get("source_type", "segment")
    
    icon = "📝" if source_type == "segment" else "📋"
    lines.append(f"  {icon} [{rank}] score={score:.3f}  date={date}  domain={domain}")
    
    if topic:
        lines.append(f"      topic: {topic}")
    
    # Content preview
    preview = doc[:300].replace("\n", " ").strip()
    if len(doc) > 300:
        preview += "..."
    lines.append(f"      {preview}")
    
    if verbose:
        lines.append(f"      session: {session_id}  distance: {distance:.4f}")
        if metadata.get("emotional_tenor"):
            lines.append(f"      tenor: {metadata['emotional_tenor']}")
        if metadata.get("decisions"):
            try:
                decisions = json.loads(metadata["decisions"])
                if decisions:
                    lines.append(f"      decisions: {', '.join(decisions[:3])}")
            except (json.JSONDecodeError, TypeError):
                pass
        if metadata.get("unresolved_threads"):
            try:
                threads = json.loads(metadata["unresolved_threads"])
                if threads:
                    lines.append(f"      unresolved: {', '.join(threads[:3])}")
            except (json.JSONDecodeError, TypeError):
                pass
    
    return "\n".join(lines)


def search(
    query: str,
    agent: str = "bob",
    mode: str = "blended",
    top_k: int = 5,
    domain: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    verbose: bool = False,
    as_json: bool = False,
) -> list[dict]:
    """
    Main search function. Returns ranked results.
    """
    store = VectorStore(agent=agent)
    query_embedding = embed_text(query)
    
    # Build Chroma where filter
    where = {}
    if domain:
        where["domain"] = domain
    # Chroma's where filtering is limited — date range filtering done post-query
    
    # Fetch more results than needed for re-ranking
    fetch_k = min(top_k * 3, 50)
    
    results = []
    
    # Segment search (specific mode)
    if mode in ("specific", "blended"):
        seg_results = store.search_segments(
            query_embedding, n_results=fetch_k, where=where if where else None
        )
        if seg_results and seg_results.get("documents"):
            docs = seg_results["documents"][0]
            metas = seg_results["metadatas"][0]
            dists = seg_results["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                source_weight = float(meta.get("source_weight", 1.0))
                score = compute_final_score(dist, meta.get("date", ""), source_weight)
                
                # Date range filter
                if date_from and meta.get("date", "") < date_from:
                    continue
                if date_to and meta.get("date", "") > date_to:
                    continue
                
                results.append({
                    "doc": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": score,
                    "source": "segment",
                })
    
    # Summary search (pattern mode)
    if mode in ("pattern", "blended"):
        sum_results = store.search_summaries(
            query_embedding, n_results=fetch_k, where=where if where else None
        )
        if sum_results and sum_results.get("documents"):
            docs = sum_results["documents"][0]
            metas = sum_results["metadatas"][0]
            dists = sum_results["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                source_weight = float(meta.get("source_weight", 0.7))
                score = compute_final_score(dist, meta.get("date", ""), source_weight)
                
                if date_from and meta.get("date", "") < date_from:
                    continue
                if date_to and meta.get("date", "") > date_to:
                    continue
                
                results.append({
                    "doc": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": score,
                    "source": "summary",
                })
    
    # Sort by final score (highest first)
    results.sort(key=lambda r: r["score"], reverse=True)
    
    # Deduplicate: if a summary and its segments both appear, prefer segments
    seen_sessions = set()
    deduped = []
    for r in results:
        sid = r["metadata"].get("session_id", "")
        source = r["source"]
        
        if source == "summary" and sid in seen_sessions:
            continue  # Already have segments from this session
        
        seen_sessions.add(sid)
        deduped.append(r)
    
    return deduped[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Semantic memory retrieval — search past conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pai-memory-recall "what were we working on last Tuesday"
  pai-memory-recall --mode specific "Grimm's medication changes"
  pai-memory-recall --mode pattern --domain redhat "customer escalation"
  pai-memory-recall --agent bob --top 10 --verbose "NFS mount issues"
  pai-memory-recall --json "semantic memory decisions"
  pai-memory-recall --from 2026-02-01 --to 2026-02-28 "trading strategies"
        """,
    )
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--agent", default="bob", help="Agent namespace (bob/patterson/dean)")
    parser.add_argument("--mode", default="blended", choices=["specific", "pattern", "blended"],
                        help="Retrieval mode (default: blended)")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--domain", help="Filter by domain (redhat, draupnir, mga, rec, wardstone, health, household)")
    parser.add_argument("--from", dest="date_from", help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra metadata")
    parser.add_argument("--json", "-j", action="store_true", dest="as_json", help="Output as JSON")
    parser.add_argument("--context", "-c", action="store_true", help="Output as context block for agent prompts")
    
    args = parser.parse_args()
    
    results = search(
        query=args.query,
        agent=args.agent,
        mode=args.mode,
        top_k=args.top,
        domain=args.domain,
        date_from=args.date_from,
        date_to=args.date_to,
        verbose=args.verbose,
        as_json=args.as_json,
    )
    
    if args.as_json:
        # JSON output for programmatic use
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
                "text": r["doc"][:500],
            })
        print(json.dumps(output, indent=2))
        return
    
    if args.context:
        # Context block output for agent prompt injection
        from src.retrieval.context_loader import recall as context_recall
        result = context_recall(
            args.query,
            agent=args.agent,
            mode=args.mode,
            top_k=args.top,
            domain=args.domain,
        )
        if result:
            print(result)
        else:
            print("(no relevant memories found)")
        return
    
    # Human-readable output
    store = VectorStore(agent=args.agent)
    stats = store.stats()
    
    print(f"\n🧠 Semantic Memory — \"{args.query}\"")
    print(f"   agent={args.agent}  mode={args.mode}  store={stats['segments_count']} segments, {stats['summaries_count']} summaries")
    print(f"{'─' * 70}")
    
    if not results:
        print("   No results found.")
        print(f"{'─' * 70}")
        return
    
    for i, r in enumerate(results, 1):
        print(format_result(i, r["doc"], r["metadata"], r["score"], r["distance"], args.verbose))
        if i < len(results):
            print()
    
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

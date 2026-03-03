#!/usr/bin/env python3
"""
Bulk Ingest Pipeline — Wires exporter → chunker → embedder → store.

Usage:
    python -m src.pipeline.ingest --agent bob --limit 30 [--mode topic|sliding|blended]
    python -m src.pipeline.ingest --agent bob --session <session_id>
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.exporter import extract_conversation, list_sessions
from src.pipeline.embedder import embed_text, embed_batch
from src.pipeline.store import VectorStore
from src.chunker.topic_chunker import chunk_session, ChunkerConfig


def ingest_session(
    session_path: Path,
    store: VectorStore,
    config: ChunkerConfig,
    verbose: bool = False,
    agent: str = "bob",
    min_messages: int = 4,
) -> dict:
    """
    Full pipeline for one session: export → chunk → embed → store.
    
    Returns stats dict.
    """
    # 1. Export (clean the raw JSONL)
    session_data = extract_conversation(str(session_path))
    
    if not session_data or not session_data.get("messages"):
        return {"status": "skipped", "reason": "empty_or_invalid"}
    
    msg_count = len(session_data["messages"])
    if msg_count < min_messages:
        return {"status": "skipped", "reason": "too_few_messages", "messages": msg_count}
    
    session_id = session_data.get("session_id", session_path.stem)
    date = session_data.get("date", "")
    
    if verbose:
        print(f"  Exported: {msg_count} messages, date={date}")
    
    # 2. Chunk (topic segmentation via Patterson's chunker)
    chunked = chunk_session(session_data, config, agent=agent)
    segments = chunked.get("segments", [])
    summary = chunked.get("session_summary")
    
    if not segments:
        return {"status": "skipped", "reason": "no_segments", "messages": msg_count}
    
    if verbose:
        print(f"  Chunked: {len(segments)} segments ({config.mode} mode)")
    
    # 3. Embed segments (filter out empty texts)
    embeddable_texts = [s["embeddable_text"] for s in segments if s.get("embeddable_text", "").strip()]
    if not embeddable_texts:
        return {"status": "skipped", "reason": "no_embeddable_text", "messages": msg_count}
    
    # Filter segments to match
    valid_segments = [s for s in segments if s.get("embeddable_text", "").strip()]
    
    t0 = time.time()
    segment_embeddings = embed_batch(embeddable_texts)
    embed_time = time.time() - t0
    
    if verbose:
        print(f"  Embedded: {len(segment_embeddings)} segments in {embed_time:.2f}s")
    
    # 4. Store segments
    for seg, embedding in zip(valid_segments, segment_embeddings):
        store.add_segment(
            segment_id=seg["segment_id"],
            session_id=seg.get("session_id", session_id),
            text=seg["embeddable_text"],
            embedding=embedding,
            date=seg.get("date", date),
            domain=seg.get("domain", "general"),
            topic=seg.get("topic", ""),
            tone=seg.get("metadata", {}).get("tone", ""),
        )
    
    # 5. Embed and store session summary
    summary_stored = False
    if summary and summary.get("embeddable_text"):
        summary_embedding = embed_text(summary["embeddable_text"])
        store.add_session_summary(
            session_id=session_id,
            summary_text=summary["embeddable_text"],
            embedding=summary_embedding,
            date=date,
            domain=summary.get("domain", "general"),
            topics=summary.get("metadata", {}).get("all_topics", []),
        )
        summary_stored = True
    
    return {
        "status": "ingested",
        "session_id": session_id,
        "date": date,
        "messages": msg_count,
        "segments": len(segments),
        "summary": summary_stored,
        "embed_time": round(embed_time, 2),
    }


def run_bulk_ingest(
    agent: str,
    limit: int = 30,
    mode: str = "topic",
    verbose: bool = False,
    session_dir: str | None = None,
    min_messages: int = 4,
):
    """Run bulk ingest for an agent."""
    # Find session files
    if session_dir:
        sessions_path = Path(session_dir)
        session_files = sorted(sessions_path.glob("*.jsonl"), reverse=True)
    else:
        # Use the smart listing that filters probes, deleted, and tiny sessions
        session_list = list_sessions(limit=limit if limit > 0 else 9999, min_messages=min_messages)
        session_files = [Path(s["path"]) for s in session_list]
    
    print(f"Found {len(session_files)} eligible session files")
    
    # Initialize store and config
    store = VectorStore(agent=agent)
    config = ChunkerConfig(mode=mode)
    
    # Deduplication: skip sessions already in the store
    # Session IDs are UUIDs, but filenames may have suffixes like -topic-<ts>
    existing_ids = store.ingested_session_ids()
    if existing_ids:
        import re
        def extract_uuid(stem: str) -> str:
            """Extract UUID from filename stem (handles -topic-<ts> suffixes)."""
            m = re.match(r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', stem)
            return m.group(1) if m else stem
        
        before = len(session_files)
        session_files = [f for f in session_files if extract_uuid(f.stem) not in existing_ids]
        skipped_dedup = before - len(session_files)
        if skipped_dedup:
            print(f"Skipping {skipped_dedup} already-ingested sessions (incremental mode)")
    
    if not session_files:
        print("No new sessions to ingest.")
        return {"ingested": 0, "skipped": 0, "failed": 0, "total_segments": 0, "total_messages": 0, "total_embed_time": 0}
    
    print(f"\n{'='*60}")
    print(f"SEMANTIC MEMORY — Bulk Ingest")
    print(f"{'='*60}")
    print(f"Agent: {agent}")
    print(f"Mode: {mode}")
    print(f"Sessions: {len(session_files)}")
    print(f"Store: {store.base_path}")
    print(f"{'='*60}\n")
    
    # Process each session
    stats = {
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "total_segments": 0,
        "total_messages": 0,
        "total_embed_time": 0,
    }
    
    for i, session_file in enumerate(session_files, 1):
        if verbose:
            print(f"\n[{i}/{len(session_files)}] {session_file.name}")
        else:
            print(f"  [{i}/{len(session_files)}] {session_file.name[:40]}...", end=" ")
        
        try:
            result = ingest_session(session_file, store, config, verbose, agent=agent, min_messages=min_messages)
            
            if result["status"] == "ingested":
                stats["ingested"] += 1
                stats["total_segments"] += result["segments"]
                stats["total_messages"] += result["messages"]
                stats["total_embed_time"] += result.get("embed_time", 0)
                if not verbose:
                    print(f"✓ {result['segments']} segs, {result['messages']} msgs")
            else:
                stats["skipped"] += 1
                if not verbose:
                    print(f"⊘ {result.get('reason', 'unknown')}")
                    
        except Exception as e:
            stats["failed"] += 1
            if verbose:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"✗ {str(e)[:60]}")
    
    # Print summary
    store_info = store.stats()
    print(f"\n{'='*60}")
    print(f"INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Ingested:  {stats['ingested']} sessions")
    print(f"  Skipped:   {stats['skipped']} sessions")
    print(f"  Failed:    {stats['failed']} sessions")
    print(f"  Segments:  {stats['total_segments']} total")
    print(f"  Messages:  {stats['total_messages']} processed")
    print(f"  Embed time: {stats['total_embed_time']:.1f}s total")
    print(f"  Store:     {store_info['segments_count']} segments, {store_info['summaries_count']} summaries")
    print(f"{'='*60}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Semantic Memory Bulk Ingest")
    parser.add_argument("--agent", default="bob", help="Agent namespace (bob/patterson/dean)")
    parser.add_argument("--limit", type=int, default=30, help="Max sessions to ingest (0=all)")
    parser.add_argument("--mode", default="topic", choices=["topic", "sliding"], help="Chunking mode")
    parser.add_argument("--session-dir", help="Override session directory path")
    parser.add_argument("--min-messages", type=int, default=4, help="Min messages to include a session (default 4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    run_bulk_ingest(
        agent=args.agent,
        limit=args.limit,
        mode=args.mode,
        verbose=args.verbose,
        session_dir=args.session_dir,
        min_messages=args.min_messages,
    )


if __name__ == "__main__":
    main()

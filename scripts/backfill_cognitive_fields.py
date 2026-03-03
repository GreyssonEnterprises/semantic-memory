#!/usr/bin/env python3
"""Backfill cognitive primitive metadata for pre-existing memories.

Reads all memories from Chroma, adds default cognitive fields,
and optionally infers access_count from heuristic signals.

Usage:
    python scripts/backfill_cognitive_fields.py --agent bob [--dry-run] [--infer-access]
    python scripts/backfill_cognitive_fields.py --agent bob --dry-run
    python scripts/backfill_cognitive_fields.py --agent shared --infer-access
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.store import VectorStore

# Heuristic access count inference
INFERRED_ACCESS_COUNTS = {
    "vault": 10,
    "procedure": 5,
    "graph": 3,
    "daily": 1,
    "session_summary": 2,
    "session_segment": 1,
}

COGNITIVE_DEFAULTS = {
    "access_count": 1,
    "last_n_access_timestamps": json.dumps([]),
    "confidence": 1.0,
    "last_pushed_at": "",
}


def backfill_collection(collection, agent: str, dry_run: bool, infer_access: bool) -> dict:
    """Backfill cognitive fields for a single collection.

    Returns stats dict with counts of updated/skipped records.
    """
    stats = {"total": 0, "updated": 0, "skipped": 0}

    # Get all documents with metadata
    all_docs = collection.get(include=["metadatas"])
    if not all_docs or not all_docs["ids"]:
        return stats

    for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
        stats["total"] += 1

        # Check if already backfilled (idempotency check)
        if "access_count" in metadata and "confidence" in metadata and "last_pushed_at" in metadata:
            stats["skipped"] += 1
            continue

        # Build update
        updates = {}
        for field, default_value in COGNITIVE_DEFAULTS.items():
            if field not in metadata:
                if field == "access_count" and infer_access:
                    source_type = metadata.get("source_type", "session_segment")
                    updates[field] = INFERRED_ACCESS_COUNTS.get(source_type, 1)
                else:
                    updates[field] = default_value

        if not updates:
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would update {doc_id}: {updates}")
        else:
            new_metadata = {**metadata, **updates}
            collection.update(ids=[doc_id], metadatas=[new_metadata])

        stats["updated"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill cognitive metadata fields")
    parser.add_argument("--agent", required=True, choices=["bob", "patterson", "dean", "shared"],
                        help="Agent store to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--infer-access", action="store_true",
                        help="Infer access_count from source_type heuristics")
    args = parser.parse_args()

    print(f"Backfilling cognitive fields for agent: {args.agent}")
    if args.dry_run:
        print("  MODE: dry-run (no changes will be written)")
    if args.infer_access:
        print("  MODE: inferring access counts from source_type")

    store = VectorStore(agent=args.agent)

    print(f"\nProcessing segments collection...")
    seg_stats = backfill_collection(store.segments, args.agent, args.dry_run, args.infer_access)
    print(f"  Segments: {seg_stats['total']} total, {seg_stats['updated']} updated, {seg_stats['skipped']} skipped")

    print(f"\nProcessing summaries collection...")
    sum_stats = backfill_collection(store.summaries, args.agent, args.dry_run, args.infer_access)
    print(f"  Summaries: {sum_stats['total']} total, {sum_stats['updated']} updated, {sum_stats['skipped']} skipped")

    total_updated = seg_stats["updated"] + sum_stats["updated"]
    total_skipped = seg_stats["skipped"] + sum_stats["skipped"]
    print(f"\nDone. Updated: {total_updated}, Skipped: {total_skipped}")


if __name__ == "__main__":
    main()

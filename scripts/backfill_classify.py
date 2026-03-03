#!/usr/bin/env python3
"""Backfill classifier metadata on existing Chroma segments.

Reads segments from a Chroma collection, classifies each via Ollama,
and updates metadata in place (no duplicates, no new markdown files).

Usage:
    uv run python scripts/backfill_classify.py --agent bob
    uv run python scripts/backfill_classify.py --agent bob --dry-run
    uv run python scripts/backfill_classify.py --agent bob --limit 10
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import chromadb
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.1.22:11434")
CLASSIFY_MODEL = os.environ.get("CLASSIFY_MODEL", "gemma3:4b")
VECTOR_STORE_BASE = os.environ.get("VECTOR_STORE_BASE", "/opt/vector-store")

VALID_TYPES = {"decision", "learning", "task", "person", "procedure", "insight", "meeting_debrief"}

# Domain mapping: LLM might return simplified names — map to canonical taxonomy names
DOMAIN_ALIASES = {
    "redhat": "red-hat-tam",
    "red-hat": "red-hat-tam",
    "tam": "red-hat-tam",
    "psirt": "red-hat-psirt",
    "mga": "mga-business",
    "rec": "rec-business",
    "trading": "trading",
    "household": "household-logistics",
    "home": "household-logistics",
    "wardstone": "wardstone",
    "security": "wardstone",
    "personal": "personal",
    "health": "medical-health",
    "medical": "medical-health",
    "school": "caroline-school",
    "cricut": "cricut-crafting",
    "crafting": "cricut-crafting",
    "books": "books-reading",
    "reading": "books-reading",
    "cooking": "meal-planning",
    "meals": "meal-planning",
    "gifts": "gift-bags",
    "deals": "deals-shopping",
    "shopping": "deals-shopping",
    "adhd": "adhd-support",
}

# Load valid domains from taxonomy JSON if available
def load_valid_domains() -> set[str]:
    """Load all domain names from taxonomy JSON."""
    taxonomy_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "domain-taxonomy.json"),
        os.path.expanduser("~/clawd/projects/semantic-memory/data/domain-taxonomy.json"),
    ]
    for path in taxonomy_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            domains = set()
            for agent_data in data.get("agents", {}).values():
                for domain_name in agent_data.get("domains", {}).keys():
                    domains.add(domain_name)
            # Add base domains the LLM might return
            domains.update({"general", "personal", "infrastructure", "trading", "wardstone", "cross-agent", "coding"})
            return domains
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    # Fallback if no taxonomy file found
    return {"redhat", "red-hat-tam", "red-hat-psirt", "trading", "mga-business", "mga-rec-business",
            "rec-business", "household-logistics", "household", "infrastructure", "wardstone", "personal",
            "general", "coding", "cross-agent", "books-reading", "cricut-crafting", "caroline-school",
            "meal-planning", "gift-bags", "deals-shopping", "medical-health", "adhd-support"}

VALID_DOMAINS = load_valid_domains()

def normalize_domain(raw: str) -> str:
    """Normalize LLM domain output to canonical taxonomy name."""
    raw = raw.lower().strip()
    if raw in VALID_DOMAINS:
        return raw
    if raw in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[raw]
    # Try partial matching
    for alias, canonical in DOMAIN_ALIASES.items():
        if alias in raw or raw in alias:
            return canonical
    return "general"

SYSTEM_PROMPT = """You are a memory classification system. Analyze the content and return a JSON object with these fields:

{
  "type": one of: "decision" (a choice made with rationale), "learning" (something learned or discovered), "task" (action item or to-do), "person" (information about a person), "procedure" (step-by-step process or SOP), "insight" (observation or analysis), "meeting_debrief" (notes from a meeting or call),
  "domain": one of: "red-hat-tam" (Red Hat TAM work, cases, customer support), "red-hat-psirt" (PSIRT, CVEs, vulnerability management), "trading" (Draupnir trading, markets, portfolio), "mga-business" (MGA/Midnight Garden Apothecary), "rec-business" (REC/Raven's Eye Consulting retail), "mga-rec-business" (combined MGA/REC business), "household-logistics" (family, home, logistics), "infrastructure" (tech, servers, tools, AI agents), "wardstone" (security, pen testing), "personal" (personal health, relationships, growth), "coding" (software development, programming), "cross-agent" (inter-agent coordination), "books-reading" (book discussions, recommendations), "cricut-crafting" (Cricut machine, vinyl, crafting), "caroline-school" (school activities, homework), "meal-planning" (cooking, recipes, meal prep), "medical-health" (medical, health, doctor visits), "general" (doesn't fit any specific domain),
  "topics": array of 1-5 short topic tags extracted from the content,
  "people": array of person names mentioned (use "First Last" format),
  "action_items": array of action items or to-dos found in the content (empty array if none),
  "confidence": number between 0.0 and 1.0 indicating how confident you are in the type and domain classification
}

Return ONLY valid JSON. No explanation or extra text."""


def classify_content(content: str) -> dict[str, Any]:
    """Classify content via Ollama. Returns classification dict or fallback."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CLASSIFY_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": f"Classify this memory content:\n\n{content[:4000]}",
                "format": "json",
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json().get("response", "{}"))
    except Exception as e:
        print(f"  ⚠ classify failed: {e}")
        return make_fallback()

    raw_type = str(parsed.get("type", "")).lower().strip()
    raw_domain = str(parsed.get("domain", "")).lower().strip()

    if raw_type not in VALID_TYPES:
        print(f"  ⚠ invalid type={raw_type!r}, using fallback")
        return make_fallback()

    # Normalize domain to canonical taxonomy name
    raw_domain = normalize_domain(raw_domain)

    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))

    # Apply confidence thresholds (matches classifier.ts behavior)
    result_type = raw_type
    low_confidence = False
    if confidence >= 0.7:
        low_confidence = False
    elif confidence >= 0.4:
        low_confidence = True
    else:
        result_type = "insight"
        low_confidence = True

    return {
        "type": result_type,
        "domain": raw_domain,
        "topics": parsed.get("topics", []),
        "people": parsed.get("people", []),
        "action_items": parsed.get("action_items", []),
        "confidence": confidence,
        "low_confidence": low_confidence,
    }


def make_fallback() -> dict[str, Any]:
    return {
        "type": "insight",
        "domain": "personal",
        "topics": [],
        "people": [],
        "action_items": [],
        "confidence": 0.0,
        "low_confidence": True,
    }


def backfill_agent(agent: str, dry_run: bool = False, limit: int | None = None, skip_classified: bool = False):
    store_path = os.path.join(VECTOR_STORE_BASE, agent)
    if not os.path.exists(store_path):
        print(f"✗ Store not found: {store_path}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=store_path)
    collections = client.list_collections()
    print(f"Agent: {agent}")
    print(f"Store: {store_path}")
    print(f"Collections: {[c.name for c in collections]}")

    total = 0
    updated = 0
    errors = 0
    type_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    confidence_buckets = {"high": 0, "medium": 0, "low": 0, "fallback": 0}

    for collection in collections:
        # Get all segments
        batch_size = 100
        offset = 0
        while True:
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if not results["ids"]:
                break

            for i, doc_id in enumerate(results["ids"]):
                if limit and total >= limit:
                    break

                doc = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}

                if not doc or not doc.strip():
                    total += 1
                    continue

                # Skip already-classified segments if requested
                if skip_classified and meta.get("type") and meta["type"] != "unknown":
                    total += 1
                    continue

                total += 1
                print(f"  [{total}] {doc_id[:40]}... ", end="", flush=True)

                classification = classify_content(doc)

                # Update metadata
                new_meta = {**meta}
                new_meta["type"] = classification["type"]
                new_meta["domain"] = classification["domain"]
                new_meta["topics"] = json.dumps(classification["topics"])
                new_meta["people"] = json.dumps(classification["people"])
                new_meta["action_items"] = json.dumps(classification["action_items"])
                new_meta["confidence"] = classification["confidence"]
                new_meta["low_confidence"] = classification["low_confidence"]

                # Track stats
                t = classification["type"]
                d = classification["domain"]
                type_counts[t] = type_counts.get(t, 0) + 1
                domain_counts[d] = domain_counts.get(d, 0) + 1

                conf = classification["confidence"]
                if conf >= 0.7:
                    confidence_buckets["high"] += 1
                elif conf >= 0.4:
                    confidence_buckets["medium"] += 1
                elif conf > 0:
                    confidence_buckets["low"] += 1
                else:
                    confidence_buckets["fallback"] += 1

                if not dry_run:
                    try:
                        collection.update(
                            ids=[doc_id],
                            metadatas=[new_meta],
                        )
                        updated += 1
                        print(f"→ {t}/{d} (conf={conf:.2f})")
                    except Exception as e:
                        errors += 1
                        print(f"✗ update failed: {e}")
                else:
                    print(f"→ [DRY RUN] {t}/{d} (conf={conf:.2f})")
                    updated += 1

            if limit and total >= limit:
                break
            offset += batch_size

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BACKFILL {'(DRY RUN) ' if dry_run else ''}COMPLETE — {agent}")
    print(f"{'=' * 60}")
    print(f"  Total segments:  {total}")
    print(f"  Updated:         {updated}")
    print(f"  Errors:          {errors}")
    print(f"\n  Type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = (c / updated * 100) if updated else 0
        print(f"    {t:20s} {c:4d} ({pct:.1f}%)")
    print(f"\n  Domain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = (c / updated * 100) if updated else 0
        print(f"    {d:20s} {c:4d} ({pct:.1f}%)")
    general_ratio = domain_counts.get("general", 0) / updated * 100 if updated else 0
    print(f"\n  Confidence buckets:")
    print(f"    High (≥0.7):    {confidence_buckets['high']}")
    print(f"    Medium (0.4-7): {confidence_buckets['medium']}")
    print(f"    Low (<0.4):     {confidence_buckets['low']}")
    print(f"    Fallback (0):   {confidence_buckets['fallback']}")
    print(f"\n  General ratio:   {general_ratio:.1f}% (target: <20%)")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Backfill classifier metadata on Chroma segments")
    parser.add_argument("--agent", required=True, help="Agent name (bob, patterson, dean)")
    parser.add_argument("--dry-run", action="store_true", help="Classify but don't update Chroma")
    parser.add_argument("--limit", type=int, help="Max segments to process")
    parser.add_argument("--skip-classified", action="store_true", help="Skip segments that already have type != 'unknown'")
    args = parser.parse_args()

    start = time.time()
    backfill_agent(args.agent, dry_run=args.dry_run, limit=args.limit, skip_classified=args.skip_classified)
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()

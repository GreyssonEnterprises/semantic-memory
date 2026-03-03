#!/usr/bin/env python3
"""
Topic-Segmented Chunker — Core of the Semantic Memory Pipeline

Two chunking strategies:
1. Topic Segmentation: Detects topic boundaries via sentence-level cosine similarity.
   Best for clean conversations (Bob/Patterson sessions).
2. Sliding Window: Overlapping windows with topic-transition prefixes.
   Best for ADHD-brain sessions where topic shifts are fast and
   the connections BETWEEN topics carry signal.

Both strategies produce chunks with:
- Summary prefix describing the topic/flow
- Source metadata (session, timestamp range, participants)
- The actual conversation text
"""

import json
import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """A segment of conversation ready for embedding."""
    text: str
    summary_prefix: str
    topic: str
    start_index: int  # message index in session
    end_index: int
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    chunk_type: str = "segment"  # "segment" or "session_summary"
    metadata: dict = field(default_factory=dict)
    # Integration fields for Bob's pipeline
    segment_id: str = ""
    session_id: str = ""
    date: str = ""
    domain: str = "general"

    def to_embeddable_text(self) -> str:
        """Text that gets sent to the embedding model."""
        return f"[{self.summary_prefix}]\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embeddable_text"] = self.to_embeddable_text()
        return d


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""
    # Topic segmentation params
    similarity_threshold: float = 0.3  # Below this = topic boundary
    min_segment_messages: int = 2      # Don't create segments smaller than this
    max_segment_messages: int = 50     # Force a break at this many messages

    # Sliding window params
    window_size_tokens: int = 500
    window_overlap_tokens: int = 100
    window_size_messages: int = 8      # Alternative: window by message count
    window_overlap_messages: int = 2

    # General
    mode: str = "topic"  # "topic" or "sliding"
    include_thinking: bool = False


def load_domain_taxonomy(taxonomy_path: Optional[str] = None) -> Optional[dict]:
    """
    Load the domain taxonomy JSON file.
    Searches in order:
    1. Explicit path
    2. Adjacent data/ directory
    3. NAS shared location
    """
    search_paths = []
    if taxonomy_path:
        search_paths.append(Path(taxonomy_path))

    # Relative to this file
    this_dir = Path(__file__).parent
    search_paths.extend([
        this_dir.parent / "data" / "domain-taxonomy.json",
        this_dir / "domain-taxonomy.json",
    ])

    for p in search_paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def classify_domain(
    text: str,
    agent: str = "patterson",
    taxonomy: Optional[dict] = None,
) -> str:
    """
    Classify a text segment into a domain using keyword matching.
    Returns the domain with the most keyword hits.
    Ties (or zero hits) → default_domain from taxonomy (usually "general").

    No LLM calls — pure keyword counting in the hot path.
    """
    if not taxonomy:
        return "general"

    default_domain = taxonomy.get("default_domain", "general")
    agents = taxonomy.get("agents", {})

    # Get this agent's domain definitions
    agent_domains = agents.get(agent, {}).get("domains", {})
    if not agent_domains:
        return default_domain

    text_lower = text.lower()
    # Pre-tokenize once
    text_words = set(re.findall(r'\b[a-zA-Z_-]{3,}\b', text_lower))

    scores: dict[str, int] = {}
    for domain_key, domain_def in agent_domains.items():
        keywords = domain_def.get("keywords", [])
        # Count keyword hits (both word-boundary and substring for hyphenated terms)
        hits = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in text_words:
                hits += 1
            elif "-" in kw_lower and kw_lower in text_lower:
                # Hyphenated terms might not match word boundaries
                hits += 1
        if hits > 0:
            scores[domain_key] = hits

    if not scores:
        return default_domain

    # Winner takes all; ties go to the domain with more keywords matched
    winner = max(scores, key=lambda k: scores[k])

    # Only classify if there's meaningful signal (at least 2 keyword hits)
    if scores[winner] < 2:
        return default_domain

    return winner


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def extract_topics_from_text(text: str) -> list[str]:
    """
    Extract likely topic keywords from a text block.
    Simple heuristic: nouns and noun phrases that appear multiple times.
    """
    # Remove common words, keep substantive terms
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Count frequencies
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # Return words that appear 2+ times, sorted by frequency
    topics = sorted(
        [w for w, c in freq.items() if c >= 2],
        key=lambda w: freq[w],
        reverse=True,
    )
    return topics[:5]


def generate_summary_prefix(messages: list[dict], topics: list[str]) -> str:
    """
    Generate a human-readable summary prefix for a chunk.
    This is a heuristic version — production would use an LLM.
    """
    participants = set()
    for msg in messages:
        if msg["role"] == "user":
            # Try to extract sender from content
            for content in msg.get("content", []):
                match = re.match(r'\[(\w+)\]:', content)
                if match:
                    participants.add(match.group(1))
        else:
            participants.add("assistant")

    topic_str = ", ".join(topics[:3]) if topics else "general discussion"
    participant_str = " & ".join(sorted(participants)) if participants else "conversation"

    return f"Topic: {topic_str} | Participants: {participant_str}"


def chunk_by_topic_segmentation(
    messages: list[dict],
    config: ChunkerConfig,
) -> list[Chunk]:
    """
    Split messages into topic segments based on content similarity.

    Without embeddings available locally (those live on studio with Ollama),
    we use a heuristic approach:
    - Track keyword overlap between adjacent message windows
    - When overlap drops below threshold, mark a topic boundary
    - This is v1 — we'll upgrade to embedding-based similarity when the
      pipeline connects to Ollama

    Returns list of Chunks.
    """
    if not messages:
        return []

    chunks = []
    current_segment_start = 0
    current_keywords = set()

    def finalize_segment(start: int, end: int) -> Optional[Chunk]:
        """Create a Chunk from a range of messages."""
        segment_msgs = messages[start:end]
        if len(segment_msgs) < config.min_segment_messages:
            return None

        # Build text
        text_parts = []
        for msg in segment_msgs:
            for content in msg.get("content", []):
                text_parts.append(content)

        text = "\n\n".join(text_parts)
        topics = extract_topics_from_text(text)
        prefix = generate_summary_prefix(segment_msgs, topics)

        return Chunk(
            text=text,
            summary_prefix=prefix,
            topic=", ".join(topics[:3]),
            start_index=start,
            end_index=end,
            start_timestamp=segment_msgs[0].get("timestamp"),
            end_timestamp=segment_msgs[-1].get("timestamp"),
            chunk_type="segment",
        )

    # Use a sliding window of recent keywords (last N messages) instead of
    # accumulating everything — accumulated sets always converge to high overlap
    keyword_window_size = 3
    recent_keyword_sets = []

    for i, msg in enumerate(messages):
        # Get keywords for this message
        msg_text = " ".join(msg.get("content", []))
        msg_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', msg_text.lower()))

        # Remove very common words that add noise
        stopwords = {
            "that", "this", "with", "from", "have", "been", "will", "what",
            "when", "where", "which", "their", "there", "they", "them",
            "then", "than", "your", "about", "would", "could", "should",
            "just", "also", "some", "more", "other", "into", "very",
            "here", "were", "does", "done", "like", "make", "each",
        }
        msg_keywords -= stopwords

        if i == 0:
            recent_keyword_sets.append(msg_keywords)
            continue

        # Compare this message against the recent window (not accumulated total)
        window_keywords = set()
        for ks in recent_keyword_sets[-keyword_window_size:]:
            window_keywords |= ks

        if window_keywords and msg_keywords:
            intersection = window_keywords & msg_keywords
            union = window_keywords | msg_keywords
            similarity = len(intersection) / len(union) if union else 0
        else:
            similarity = 0

        # Track for debugging
        recent_keyword_sets.append(msg_keywords)

        # Check for topic boundary
        segment_length = i - current_segment_start
        force_break = segment_length >= config.max_segment_messages

        if similarity < config.similarity_threshold or force_break:
            # Finalize current segment
            chunk = finalize_segment(current_segment_start, i)
            if chunk:
                chunks.append(chunk)
            current_segment_start = i

    # Don't forget the last segment
    if current_segment_start < len(messages):
        chunk = finalize_segment(current_segment_start, len(messages))
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_by_sliding_window(
    messages: list[dict],
    config: ChunkerConfig,
) -> list[Chunk]:
    """
    Sliding window chunking with overlap.
    Preserves topic transitions by keeping overlap between windows.
    Best for sessions with rapid topic switching.
    """
    if not messages:
        return []

    chunks = []
    window_size = config.window_size_messages
    overlap = config.window_overlap_messages

    step = max(1, window_size - overlap)
    i = 0

    while i < len(messages):
        end = min(i + window_size, len(messages))
        window_msgs = messages[i:end]

        # Build text
        text_parts = []
        for msg in window_msgs:
            for content in msg.get("content", []):
                text_parts.append(content)

        text = "\n\n".join(text_parts)
        topics = extract_topics_from_text(text)

        # For sliding windows, the prefix captures flow/transitions
        # Detect if topics shift within this window
        first_half = " ".join(text_parts[: len(text_parts) // 2])
        second_half = " ".join(text_parts[len(text_parts) // 2 :])
        first_topics = extract_topics_from_text(first_half)
        second_topics = extract_topics_from_text(second_half)

        if first_topics and second_topics and first_topics[:2] != second_topics[:2]:
            flow = f"{', '.join(first_topics[:2])} → {', '.join(second_topics[:2])}"
            prefix = f"Topic flow: {flow}"
        else:
            prefix = generate_summary_prefix(window_msgs, topics)

        chunks.append(
            Chunk(
                text=text,
                summary_prefix=prefix,
                topic=", ".join(topics[:3]),
                start_index=i,
                end_index=end,
                start_timestamp=window_msgs[0].get("timestamp"),
                end_timestamp=window_msgs[-1].get("timestamp"),
                chunk_type="segment",
            )
        )

        i += step
        if end >= len(messages):
            break

    return chunks


def generate_session_summary(
    messages: list[dict],
    session_meta: dict,
    chunks: list[Chunk],
) -> Chunk:
    """
    Generate a session-level summary chunk.
    This captures the "overall shape" of the conversation for pattern matching.

    V1: heuristic summary from chunk topics and metadata.
    V2: LLM-generated summary (post-pipeline integration).
    """
    # Collect all topics across chunks
    all_topics = []
    for chunk in chunks:
        all_topics.extend(chunk.topic.split(", "))

    # Deduplicate while preserving order
    seen = set()
    unique_topics = []
    for t in all_topics:
        if t and t not in seen:
            seen.add(t)
            unique_topics.append(t)

    # Build session summary text
    summary_parts = [
        f"Session from {session_meta.get('started_at', 'unknown date')}.",
        f"Participants: {', '.join(session_meta.get('participants', ['unknown']))}.",
        f"Topics covered: {', '.join(unique_topics[:10])}.",
        f"Total segments: {len(chunks)}.",
        f"Total messages: {session_meta.get('message_count', '?')}.",
    ]

    # Include first and last chunk summaries for "shape"
    if chunks:
        summary_parts.append(f"Started with: {chunks[0].summary_prefix}")
        if len(chunks) > 1:
            summary_parts.append(f"Ended with: {chunks[-1].summary_prefix}")

    summary_text = " ".join(summary_parts)

    return Chunk(
        text=summary_text,
        summary_prefix=f"Session summary: {', '.join(unique_topics[:5])}",
        topic=", ".join(unique_topics[:5]),
        start_index=0,
        end_index=len(messages),
        start_timestamp=messages[0].get("timestamp") if messages else None,
        end_timestamp=messages[-1].get("timestamp") if messages else None,
        chunk_type="session_summary",
        metadata={
            "all_topics": unique_topics,
            "segment_count": len(chunks),
        },
    )


def chunk_session(
    session_data: dict,
    config: Optional[ChunkerConfig] = None,
    agent: str = "patterson",
    taxonomy_path: Optional[str] = None,
) -> dict:
    """
    Main entry point: chunk an exported session into segments + session summary.

    Accepts EITHER:
    - Patterson's exporter format: {meta: {session_id, started_at, ...}, messages: [...]}
    - Bob's exporter format: {session_id, date, messages: [{role, text, timestamp}], raw_text}

    Returns:
        Dict with segments, session_summary, and metadata
    """
    if config is None:
        config = ChunkerConfig()

    # Load domain taxonomy for classification
    taxonomy = load_domain_taxonomy(taxonomy_path)

    # Normalize input format — support both exporters
    if "meta" in session_data:
        # Patterson's format
        messages = session_data.get("messages", [])
        meta = session_data.get("meta", {})
        session_id = meta.get("session_id", "unknown")
        date = (meta.get("started_at") or "")[:10]
    else:
        # Bob's format — convert messages to our internal format
        raw_messages = session_data.get("messages", [])
        session_id = session_data.get("session_id", "unknown")
        date = session_data.get("date", "")
        messages = []
        for m in raw_messages:
            messages.append({
                "role": m.get("role", "user"),
                "content": [m.get("text", "")],
                "timestamp": m.get("timestamp", ""),
            })
        meta = {
            "session_id": session_id,
            "started_at": date,
            "message_count": len(messages),
            "participants": list(set(
                m.get("role", "") for m in raw_messages
            )),
        }

    if not messages:
        return {"segments": [], "session_summary": None, "meta": meta}

    # Choose chunking strategy
    if config.mode == "sliding":
        segments = chunk_by_sliding_window(messages, config)
    else:
        segments = chunk_by_topic_segmentation(messages, config)

    # Populate integration fields on each segment
    import hashlib
    for i, seg in enumerate(segments):
        seg.session_id = session_id
        seg.date = date
        # Classify domain from segment text
        seg.domain = classify_domain(seg.text, agent=agent, taxonomy=taxonomy)
        # Generate a stable segment_id from session + position
        id_input = f"{session_id}:{seg.start_index}:{seg.end_index}:{config.mode}"
        seg.segment_id = hashlib.sha256(id_input.encode()).hexdigest()[:16]

    # Generate session-level summary
    session_summary = generate_session_summary(messages, meta, segments)
    session_summary.session_id = session_id
    session_summary.date = date
    # Session summary domain = most common domain across segments
    domain_counts: dict[str, int] = {}
    for seg in segments:
        domain_counts[seg.domain] = domain_counts.get(seg.domain, 0) + 1
    if domain_counts:
        session_summary.domain = max(domain_counts, key=lambda k: domain_counts[k])
    session_summary.segment_id = hashlib.sha256(
        f"{session_id}:summary".encode()
    ).hexdigest()[:16]

    return {
        "segments": [s.to_dict() for s in segments],
        "session_summary": session_summary.to_dict(),
        "meta": {
            **meta,
            "chunker_mode": config.mode,
            "segment_count": len(segments),
        },
    }


if __name__ == "__main__":
    """Process an exported session file and output chunks."""
    if len(sys.argv) < 2:
        print("Usage: chunker.py <exported_session.json> [mode=topic|sliding]")
        print("\nExample:")
        print("  python chunker.py data/exported/abc123.json topic")
        sys.exit(1)

    input_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "topic"

    with open(input_file) as f:
        session_data = json.load(f)

    config = ChunkerConfig(mode=mode)
    result = chunk_session(session_data, config)

    print(f"\nSession: {result['meta'].get('session_id', '?')}")
    print(f"Mode: {mode}")
    print(f"Segments: {result['meta']['segment_count']}")
    print(f"Messages: {result['meta'].get('message_count', '?')}")

    print(f"\n--- Session Summary ---")
    if result["session_summary"]:
        print(result["session_summary"]["summary_prefix"])
        print(result["session_summary"]["text"][:500])

    print(f"\n--- Segments ---")
    for i, seg in enumerate(result["segments"]):
        print(f"\n[{i+1}] {seg['summary_prefix']}")
        print(f"    Messages {seg['start_index']}-{seg['end_index']}")
        preview = seg["text"][:150].replace("\n", " ")
        print(f"    Preview: {preview}...")

    # Write output
    output_file = input_file.replace(".json", f".chunks.{mode}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to: {output_file}")

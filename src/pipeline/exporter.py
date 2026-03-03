"""Session exporter — reads OpenClaw JSONL sessions and extracts clean conversation text."""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# OpenClaw session storage path
SESSION_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"

# Message types/roles to skip during cleaning
SKIP_ROLES = {"toolResult", "tool_result"}
NOISE_PATTERNS = [
    "HEARTBEAT_OK",
    "NO_REPLY",
    "[System Message]",
    "heartbeat_poll",
]


def list_sessions(limit: int = 30, min_messages: int = 4) -> list[dict]:
    """List available sessions sorted by date (newest first).
    
    Returns list of {session_id, date, path, message_count}.
    Skips deleted sessions and tiny ones (< min_messages).
    """
    sessions = []
    for f in SESSION_DIR.glob("*.jsonl"):
        if ".deleted." in f.name:
            continue
        if "probe-" in f.name:
            continue

        # Quick scan: count messages and get date
        msg_count = 0
        session_date = None
        session_id = None
        try:
            with open(f, errors="replace") as fp:
                for line in fp:
                    d = json.loads(line.strip())
                    if d.get("type") == "session":
                        session_id = d.get("id", f.stem)
                        ts = d.get("timestamp", "")
                        if ts:
                            session_date = ts[:10]  # YYYY-MM-DD
                    elif d.get("type") == "message":
                        msg = d.get("message", {})
                        if msg.get("role") in ("user", "assistant"):
                            msg_count += 1
        except (json.JSONDecodeError, IOError):
            continue

        if msg_count >= min_messages:
            sessions.append({
                "session_id": session_id or f.stem,
                "date": session_date or "unknown",
                "path": str(f),
                "message_count": msg_count,
            })

    sessions.sort(key=lambda s: s["date"], reverse=True)
    return sessions[:limit]


def extract_conversation(session_path: str, strip_tools: bool = True) -> dict:
    """Extract clean conversation from a session JSONL file.
    
    Returns:
        {
            "session_id": str,
            "date": str,
            "model": str,
            "messages": [{"role": str, "text": str, "timestamp": str}],
            "raw_text": str  # concatenated clean conversation
        }
    """
    messages = []
    session_id = ""
    session_date = ""
    model = ""

    with open(session_path, errors="replace") as fp:
        for line in fp:
            try:
                d = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            msg_type = d.get("type", "")

            if msg_type == "session":
                session_id = d.get("id", "")
                ts = d.get("timestamp", "")
                session_date = ts[:10] if ts else ""

            elif msg_type == "model_change":
                model = d.get("modelId", model)

            elif msg_type == "message":
                msg = d.get("message", {})
                role = msg.get("role", "")
                timestamp = d.get("timestamp", "")

                # Skip tool results and non-conversation roles
                if role in SKIP_ROLES:
                    continue
                if strip_tools and role not in ("user", "assistant"):
                    continue

                # Extract text content
                content = msg.get("content", "")
                text = _extract_text(content)

                if not text or not text.strip():
                    continue

                # Skip noise
                if any(p in text for p in NOISE_PATTERNS):
                    continue

                # Skip tool use blocks in assistant messages
                if role == "assistant" and text.startswith('{"type":"tool_use"'):
                    continue

                messages.append({
                    "role": role,
                    "text": text.strip(),
                    "timestamp": timestamp,
                })

    # Build raw conversation text
    raw_parts = []
    for m in messages:
        label = "Human" if m["role"] == "user" else "Assistant"
        raw_parts.append(f"[{label}]: {m['text']}")

    return {
        "session_id": session_id,
        "date": session_date,
        "model": model,
        "messages": messages,
        "raw_text": "\n\n".join(raw_parts),
    }


def _extract_text(content) -> str:
    """Extract plain text from message content (handles str and list formats)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    # Skip thinking blocks
                    continue
                elif block.get("type") == "tool_use":
                    # Skip tool use
                    continue
                elif block.get("type") == "tool_result":
                    continue
        return "\n".join(texts)
    return str(content) if content else ""


def export_sessions(
    limit: int = 30,
    output_dir: Optional[Path] = None,
    min_messages: int = 4,
) -> list[dict]:
    """Export clean conversation text from recent sessions.
    
    Returns list of extracted session dicts.
    If output_dir is provided, also writes JSON files.
    """
    sessions = list_sessions(limit=limit, min_messages=min_messages)
    results = []

    for s in sessions:
        conv = extract_conversation(s["path"])
        if not conv["messages"]:
            continue

        results.append(conv)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{conv['session_id']}.json"
            with open(out_file, "w") as fp:
                json.dump(conv, fp, indent=2)

    return results


if __name__ == "__main__":
    import sys

    # Quick test: list sessions and export one
    sessions = list_sessions(limit=5, min_messages=6)
    print(f"Found {len(sessions)} sessions (showing top 5):\n")
    for s in sessions:
        print(f"  {s['date']}  {s['session_id'][:12]}...  ({s['message_count']} messages)")

    if sessions:
        print(f"\n--- Extracting first session ---")
        conv = extract_conversation(sessions[0]["path"])
        print(f"Session: {conv['session_id']}")
        print(f"Date: {conv['date']}")
        print(f"Model: {conv['model']}")
        print(f"Clean messages: {len(conv['messages'])}")
        print(f"Raw text length: {len(conv['raw_text'])} chars")
        if conv["messages"]:
            print(f"\nFirst message preview:")
            print(f"  [{conv['messages'][0]['role']}]: {conv['messages'][0]['text'][:200]}")

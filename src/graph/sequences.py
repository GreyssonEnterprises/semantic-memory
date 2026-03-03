"""Sequential pattern detection for memory retrieval.

Logs ordered retrieval sequences and enables predictive recall
by analyzing which memories tend to follow others.

Storage: SQLite table in the same coactivation.db per agent.
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter


class SequenceTracker:
    """Logs and analyzes retrieval sequences."""

    def __init__(self, agent: str, db_path: str | None = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            import os
            _store_env = os.environ.get("SEMANTIC_MEMORY_STORE")
            base = Path(_store_env) if _store_env else (Path.home() / "clawd" / "projects" / "semantic-memory" / "data" / "vectors")
            self.db_path = base / agent / "coactivation.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent = agent
        self._init_db()

    def _init_db(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence TEXT NOT NULL,
                query_text TEXT NOT NULL,
                agent TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_context TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seq_agent ON retrieval_sequences(agent)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seq_timestamp ON retrieval_sequences(timestamp)")
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))


def log_retrieval_sequence(
    memory_ids: list[str],
    query_text: str,
    agent: str,
    session_context: str | None = None,
    db_path: str | None = None,
) -> None:
    if not memory_ids:
        return

    tracker = SequenceTracker(agent, db_path=db_path)
    now = datetime.now(timezone.utc).isoformat()

    conn = tracker._connect()
    conn.execute(
        "INSERT INTO retrieval_sequences (sequence, query_text, agent, timestamp, session_context) VALUES (?, ?, ?, ?, ?)",
        (json.dumps(memory_ids), query_text, agent, now, session_context)
    )
    conn.commit()
    conn.close()


def predict_next_memories(
    current_memory_ids: list[str],
    agent: str,
    lookback_days: int = 30,
    min_pattern_count: int = 3,
    db_path: str | None = None,
) -> list[dict]:
    if not current_memory_ids:
        return []

    tracker = SequenceTracker(agent, db_path=db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    conn = tracker._connect()
    rows = conn.execute(
        "SELECT sequence FROM retrieval_sequences WHERE agent = ? AND timestamp > ?",
        (agent, cutoff)
    ).fetchall()
    conn.close()

    if not rows:
        return []

    current_set = set(current_memory_ids)
    next_candidates = Counter()

    for (seq_json,) in rows:
        seq = json.loads(seq_json)
        seq_set = set(seq)

        if current_set.issubset(seq_set):
            for mem_id in seq:
                if mem_id not in current_set:
                    next_candidates[mem_id] += 1

    results = []
    total_sequences = len(rows)
    for mem_id, count in next_candidates.most_common():
        if count >= min_pattern_count:
            results.append({
                "memory_id": mem_id,
                "prediction_confidence": count / total_sequences,
                "pattern_count": count,
            })

    return results

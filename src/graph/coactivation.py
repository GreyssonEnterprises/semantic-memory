"""Hebbian co-activation graph for semantic memories.

Tracks which memories are frequently retrieved together and uses
association strength to boost related memories during search.

Storage: SQLite database per agent, co-located with Chroma store.

References:
    Hebb, D.O. (1949). The Organization of Behavior. Wiley.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DECAY_RATE = 0.05  # Per day
MAX_WEIGHT = 5.0
PRUNE_THRESHOLD = 0.05


class CoactivationGraph:
    """Manages co-activation edges between memories."""

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
            CREATE TABLE IF NOT EXISTS coactivation_edges (
                memory_id_a TEXT NOT NULL,
                memory_id_b TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.1,
                co_retrieval_count INTEGER NOT NULL DEFAULT 1,
                last_co_retrieval TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (memory_id_a, memory_id_b)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_a ON coactivation_edges(memory_id_a)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_b ON coactivation_edges(memory_id_b)")
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    @staticmethod
    def _canonical(id_a: str, id_b: str) -> tuple[str, str]:
        return (id_a, id_b) if id_a < id_b else (id_b, id_a)

    def record_co_retrieval(self, memory_ids: list[str], learning_rate: float = DEFAULT_LEARNING_RATE) -> None:
        if len(memory_ids) < 2:
            return

        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()

        for id_a, id_b in combinations(memory_ids, 2):
            a, b = self._canonical(id_a, id_b)

            row = conn.execute(
                "SELECT weight, co_retrieval_count FROM coactivation_edges WHERE memory_id_a = ? AND memory_id_b = ?",
                (a, b)
            ).fetchone()

            if row:
                new_weight = min(row[0] + learning_rate, MAX_WEIGHT)
                new_count = row[1] + 1
                conn.execute(
                    "UPDATE coactivation_edges SET weight = ?, co_retrieval_count = ?, last_co_retrieval = ? WHERE memory_id_a = ? AND memory_id_b = ?",
                    (new_weight, new_count, now, a, b)
                )
            else:
                conn.execute(
                    "INSERT INTO coactivation_edges (memory_id_a, memory_id_b, weight, co_retrieval_count, last_co_retrieval, created_at) VALUES (?, ?, ?, 1, ?, ?)",
                    (a, b, learning_rate, now, now)
                )

        conn.commit()
        conn.close()

    def get_associations(self, memory_id: str, min_weight: float = 0.3, limit: int = 10) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT memory_id_a, memory_id_b, weight, co_retrieval_count, last_co_retrieval
            FROM coactivation_edges
            WHERE (memory_id_a = ? OR memory_id_b = ?) AND weight >= ?
            ORDER BY weight DESC
            LIMIT ?
        """, (memory_id, memory_id, min_weight, limit)).fetchall()
        conn.close()

        results = []
        for a, b, weight, count, last in rows:
            other_id = b if a == memory_id else a
            results.append({
                "memory_id": other_id,
                "weight": weight,
                "co_retrieval_count": count,
                "last_co_retrieval": last,
            })
        return results

    def apply_decay(self, days_elapsed: float = 1.0, decay_rate: float = DEFAULT_DECAY_RATE) -> int:
        decay_factor = (1 - decay_rate) ** days_elapsed

        conn = self._connect()
        conn.execute("UPDATE coactivation_edges SET weight = weight * ?", (decay_factor,))
        cursor = conn.execute("DELETE FROM coactivation_edges WHERE weight < ?", (PRUNE_THRESHOLD,))
        pruned = cursor.rowcount
        conn.commit()
        conn.close()
        return pruned

    def hebbian_boost(self, query_results: list[dict], boost_factor: float = 0.15) -> list[dict]:
        if len(query_results) < 2:
            return query_results

        result_ids = {r.get("id", ""): i for i, r in enumerate(query_results) if r.get("id")}
        if len(result_ids) < 2:
            return query_results

        conn = self._connect()

        for id_a, id_b in combinations(result_ids.keys(), 2):
            a, b = self._canonical(id_a, id_b)
            row = conn.execute(
                "SELECT weight FROM coactivation_edges WHERE memory_id_a = ? AND memory_id_b = ?",
                (a, b)
            ).fetchone()

            if row:
                boost = row[0] * boost_factor
                query_results[result_ids[id_a]]["score"] += boost
                query_results[result_ids[id_b]]["score"] += boost

        conn.close()
        return query_results

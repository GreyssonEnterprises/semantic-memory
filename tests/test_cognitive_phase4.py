import pytest
import json
from src.graph.sequences import log_retrieval_sequence, predict_next_memories, SequenceTracker


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_sequences.db")


def test_log_retrieval_sequence(db_path):
    """Logging a sequence stores it in SQLite"""
    log_retrieval_sequence(["a", "b", "c"], "test query", "bob", db_path=db_path)

    tracker = SequenceTracker("bob", db_path=db_path)
    conn = tracker._connect()
    rows = conn.execute("SELECT sequence, query_text FROM retrieval_sequences").fetchall()
    conn.close()

    assert len(rows) == 1
    assert json.loads(rows[0][0]) == ["a", "b", "c"]
    assert rows[0][1] == "test query"


def test_predict_next_basic(db_path):
    """Known A->B pattern -> predict B after A"""
    for _ in range(4):
        log_retrieval_sequence(["a", "b", "c"], "query", "bob", db_path=db_path)

    predictions = predict_next_memories(["a"], "bob", db_path=db_path, min_pattern_count=3)
    pred_ids = [p["memory_id"] for p in predictions]
    assert "b" in pred_ids
    assert "c" in pred_ids


def test_predict_next_min_count(db_path):
    """Pattern seen once with min_pattern_count=3 -> no prediction"""
    log_retrieval_sequence(["a", "b"], "query", "bob", db_path=db_path)

    predictions = predict_next_memories(["a"], "bob", db_path=db_path, min_pattern_count=3)
    assert len(predictions) == 0


def test_predict_next_no_patterns(db_path):
    """No history -> empty predictions"""
    predictions = predict_next_memories(["x"], "bob", db_path=db_path)
    assert len(predictions) == 0


def test_predict_next_lookback_window(db_path):
    """Pattern older than lookback_days -> not returned"""
    import sqlite3
    from datetime import datetime, timezone, timedelta

    tracker = SequenceTracker("bob", db_path=db_path)
    conn = tracker._connect()
    old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    for _ in range(5):
        conn.execute(
            "INSERT INTO retrieval_sequences (sequence, query_text, agent, timestamp) VALUES (?, ?, ?, ?)",
            (json.dumps(["a", "b"]), "old query", "bob", old_time)
        )
    conn.commit()
    conn.close()

    predictions = predict_next_memories(["a"], "bob", lookback_days=30, db_path=db_path, min_pattern_count=3)
    assert len(predictions) == 0

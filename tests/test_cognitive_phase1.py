"""Tests for Phase 1: ACT-R Base-Level Activation.

Unit tests only — no ollama or external services required.
"""

import json
import math
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.embedder import actr_activation, time_weight
from src.retrieval.recall import compute_final_score, record_access


# ── actr_activation core tests ──────────────────────────────────────────────

class TestActrActivation:
    def test_new_memory(self):
        """n=1, age_days=0.01 → ≈ 0.97 (very fresh, high activation)"""
        result = actr_activation(1, 0.01)
        assert abs(result - 0.97) < 0.05, f"Expected ~0.97, got {result}"

    def test_daily_use(self):
        """n=7, age_days=7 → ≈ 0.88"""
        result = actr_activation(7, 7)
        assert abs(result - 0.88) < 0.05, f"Expected ~0.88, got {result}"

    def test_old_single_access(self):
        """n=1, age_days=365 → ≈ 0.13 (old, single access = low activation)"""
        result = actr_activation(1, 365)
        assert abs(result - 0.13) < 0.05, f"Expected ~0.13, got {result}"

    def test_heavily_used(self):
        """n=100, age_days=365 → ≈ 0.99"""
        result = actr_activation(100, 365)
        assert abs(result - 0.99) < 0.05, f"Expected ~0.99, got {result}"

    def test_n_zero_clamped(self):
        """n=0 should behave same as n=1"""
        result_zero = actr_activation(0, 7)
        result_one = actr_activation(1, 7)
        assert result_zero == result_one

    def test_age_zero_clamped(self):
        """age_days=0 → near-max activation (clamped to 0.01)"""
        result = actr_activation(1, 0)
        assert result > 0.7, f"Expected > 0.7 for age_days=0, got {result}"

    def test_negative_age(self):
        """age_days=-5 → same as age_days=0"""
        result_neg = actr_activation(1, -5)
        result_zero = actr_activation(1, 0)
        assert result_neg == result_zero

    def test_activation_weight_always_positive(self):
        """Various extreme inputs → always > 0"""
        cases = [
            (0, 0), (1, 0.01), (1, 10000), (0, 10000),
            (1000, 0.01), (1000, 10000),
        ]
        for n, age in cases:
            result = actr_activation(n, age)
            assert result > 0, f"actr_activation({n}, {age}) = {result}, expected > 0"


# ── Spread bonus tests ──────────────────────────────────────────────────────

class TestSpreadBonus:
    def test_even_spacing(self):
        """timestamps at [0, 7, 14, 21] days → bonus ≈ 1.2"""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            (base + timedelta(days=d)).isoformat()
            for d in [0, 7, 14, 21]
        ]
        result = actr_activation(4, 21, access_timestamps=timestamps)
        # With even spacing, spread_bonus should be near 1.2
        # Compare with no-bonus version
        result_no_bonus = actr_activation(4, 21, access_timestamps=None)
        assert result > result_no_bonus, "Even spacing should increase activation"

    def test_burst(self):
        """timestamps all at same second → mean_interval=0 → bonus = 1.0"""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # All at the same instant → intervals all 0 → mean=0 → bonus=1.0
        timestamps = [base.isoformat()] * 4
        result = actr_activation(4, 7, access_timestamps=timestamps)
        result_no_ts = actr_activation(4, 7, access_timestamps=None)
        # Same-instant burst gives bonus=1.0, same as no timestamps
        assert abs(result - result_no_ts) < 0.01, \
            f"Burst should give 1.0 bonus, diff={abs(result - result_no_ts)}"

    def test_insufficient_data(self):
        """0 or 1 timestamps → bonus = 1.0"""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result_none = actr_activation(3, 7, access_timestamps=None)
        result_one = actr_activation(3, 7, access_timestamps=[base.isoformat()])
        result_empty = actr_activation(3, 7, access_timestamps=[])
        assert result_none == result_one == result_empty


# ── compute_final_score tests ───────────────────────────────────────────────

class TestComputeFinalScore:
    def test_backward_compat(self):
        """Calling with old signature (no access_count) still works"""
        score = compute_final_score(0.5, "2026-03-01")
        assert score > 0

    def test_with_access_count(self):
        """Higher access count → higher score"""
        score_low = compute_final_score(0.5, "2026-01-01", access_count=1)
        score_high = compute_final_score(0.5, "2026-01-01", access_count=50)
        assert score_high > score_low, \
            f"Expected higher score with more accesses: {score_high} vs {score_low}"


# ── record_access tests ────────────────────────────────────────────────────

class TestRecordAccess:
    def test_increments_count(self, tmp_path):
        """access_count goes from 1 to 2"""
        from src.pipeline.store import VectorStore

        store = VectorStore(agent="bob", base_path=tmp_path)
        # Add a fake segment
        fake_embedding = [0.0] * 768
        store.add_segment(
            segment_id="test_seg_001",
            text="test content",
            embedding=fake_embedding,
            session_id="sess_001",
            date="2026-03-01",
        )

        metadata = {
            "access_count": 1,
            "last_n_access_timestamps": json.dumps([]),
            "session_id": "sess_001",
        }
        record_access(store, "test_seg_001", metadata, "segment")

        # Read back from Chroma
        result = store.segments.get(ids=["test_seg_001"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert int(meta["access_count"]) == 2
        ts_list = json.loads(meta["last_n_access_timestamps"])
        assert len(ts_list) == 1


# ── Store metadata tests ───────────────────────────────────────────────────

class TestStoreMetadata:
    def test_segment_has_cognitive_fields(self, tmp_path):
        """add_segment includes access_count and last_n_access_timestamps"""
        from src.pipeline.store import VectorStore

        store = VectorStore(agent="bob", base_path=tmp_path)
        fake_embedding = [0.0] * 768
        store.add_segment(
            segment_id="seg_cog_001",
            text="cognitive test",
            embedding=fake_embedding,
            session_id="sess_cog",
            date="2026-03-01",
        )
        result = store.segments.get(ids=["seg_cog_001"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert "access_count" in meta
        assert "last_n_access_timestamps" in meta
        assert int(meta["access_count"]) == 1
        assert json.loads(meta["last_n_access_timestamps"]) == []

    def test_summary_has_cognitive_fields(self, tmp_path):
        """add_session_summary includes access_count and last_n_access_timestamps"""
        from src.pipeline.store import VectorStore

        store = VectorStore(agent="bob", base_path=tmp_path)
        fake_embedding = [0.0] * 768
        store.add_session_summary(
            session_id="sess_sum_cog",
            summary_text="summary test",
            embedding=fake_embedding,
            date="2026-03-01",
        )
        result = store.summaries.get(ids=["sess_sum_cog"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert "access_count" in meta
        assert "last_n_access_timestamps" in meta
        assert int(meta["access_count"]) == 1
        assert json.loads(meta["last_n_access_timestamps"]) == []

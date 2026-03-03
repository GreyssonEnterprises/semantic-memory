"""Edge case and negative tests for cognitive primitives — blackbox verification.

Written by verifier agent, NOT the implementers.
"""

import json
import math
import os
import tempfile
import pytest
from datetime import datetime, timezone

# Ensure we use a temp directory for Chroma during tests
os.environ.setdefault("SEMANTIC_MEMORY_STORE", tempfile.mkdtemp())

from src.pipeline.embedder import actr_activation
from src.pipeline.store import VectorStore
from src.retrieval.recall import compute_final_score, record_access
from src.retrieval.confidence import bayesian_update, LIKELIHOOD_RATIOS
from src.graph.coactivation import CoactivationGraph
from src.graph.sequences import log_retrieval_sequence, predict_next_memories
from src.retrieval.push import should_push


# ────────────────────────────────────────────────────────────────────
# Edge Case 1: Cross-phase integration (ACT-R + Hebbian)
# Store a memory, access it multiple times, verify score increases.
# ────────────────────────────────────────────────────────────────────
class TestCrossPhaseIntegration:
    def test_repeated_access_increases_activation(self):
        """ACT-R activation should increase with higher access count."""
        age = 5.0  # 5 days old
        score_1 = actr_activation(n=1, age_days=age)
        score_5 = actr_activation(n=5, age_days=age)
        score_20 = actr_activation(n=20, age_days=age)
        score_100 = actr_activation(n=100, age_days=age)

        assert score_5 > score_1, f"5 accesses ({score_5}) should beat 1 ({score_1})"
        assert score_20 > score_5, f"20 accesses ({score_20}) should beat 5 ({score_5})"
        assert score_100 > score_20, f"100 accesses ({score_100}) should beat 20 ({score_20})"

    def test_hebbian_boost_accumulates_with_co_retrieval(self):
        """Co-retrieval should create edges that boost scores on subsequent searches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "coact.db")
            graph = CoactivationGraph(agent="bob", db_path=db_path)

            ids = ["mem_a", "mem_b", "mem_c"]

            # Record co-retrieval 5 times
            for _ in range(5):
                graph.record_co_retrieval(ids)

            # Check that edge weights accumulated
            assocs = graph.get_associations("mem_a", min_weight=0.0)
            assert len(assocs) >= 2, "Should have edges to mem_b and mem_c"
            for a in assocs:
                assert a["weight"] >= 0.5, f"5 co-retrievals should yield weight >= 0.5, got {a['weight']}"
                assert a["co_retrieval_count"] == 5

            # hebbian_boost should increase scores
            results = [
                {"id": "mem_a", "score": 0.5, "metadata": {}},
                {"id": "mem_b", "score": 0.4, "metadata": {}},
                {"id": "mem_c", "score": 0.3, "metadata": {}},
            ]
            original_total = sum(r["score"] for r in results)
            boosted = graph.hebbian_boost(results)
            boosted_total = sum(r["score"] for r in boosted)
            assert boosted_total > original_total, "Hebbian boost should increase total scores"


# ────────────────────────────────────────────────────────────────────
# Edge Case 2: Backward compatibility — memories without cognitive fields
# ────────────────────────────────────────────────────────────────────
class TestBackwardCompatibility:
    def test_search_works_without_cognitive_fields(self):
        """A memory inserted without access_count/confidence/last_pushed_at should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SEMANTIC_MEMORY_STORE"] = tmpdir
            store = VectorStore(agent="bob", base_path=__import__("pathlib").Path(tmpdir))

            # Manually insert a document WITHOUT cognitive fields
            fake_embedding = [0.1] * 768
            store.segments.add(
                ids=["legacy_mem_1"],
                embeddings=[fake_embedding],
                documents=["This is a legacy memory without cognitive fields"],
                metadatas=[{
                    "session_id": "legacy-session",
                    "date": "2026-01-01",
                    "domain": "general",
                    "source_type": "session_segment",
                    "source_weight": 0.8,
                    "agent": "bob",
                    # NO access_count, confidence, last_pushed_at, last_n_access_timestamps
                }],
            )

            # Search should work with defaults
            results = store.search_segments(fake_embedding, n_results=1)
            assert results["ids"][0][0] == "legacy_mem_1"
            meta = results["metadatas"][0][0]

            # compute_final_score should handle missing fields gracefully
            access_count = int(meta.get("access_count", 1))
            confidence = float(meta.get("confidence", 1.0))
            access_timestamps = json.loads(meta.get("last_n_access_timestamps", "[]"))

            score = compute_final_score(
                distance=results["distances"][0][0],
                date_str=meta.get("date", ""),
                source_weight=float(meta.get("source_weight", 1.0)),
                access_count=access_count,
                access_timestamps=access_timestamps,
                confidence=confidence,
            )
            assert score > 0, f"Score should be positive, got {score}"
            assert not math.isnan(score), "Score should not be NaN"
            assert not math.isinf(score), "Score should not be Inf"


# ────────────────────────────────────────────────────────────────────
# Edge Case 3: Boundary conditions — extreme values
# ────────────────────────────────────────────────────────────────────
class TestBoundaryConditions:
    def test_actr_extreme_high_n(self):
        """actr_activation with n=1,000,000 should not crash or return NaN/Inf."""
        result = actr_activation(n=1_000_000, age_days=10_000)
        assert not math.isnan(result), "Should not be NaN"
        assert not math.isinf(result), "Should not be Inf"
        assert result > 0, "Should be positive"

    def test_actr_extreme_tiny_age(self):
        """actr_activation with age_days=0.0001 should clamp and work."""
        result = actr_activation(n=1, age_days=0.0001)
        assert not math.isnan(result), "Should not be NaN"
        assert not math.isinf(result), "Should not be Inf"
        assert result > 0, "Should be positive"

    def test_actr_zero_n_zero_age(self):
        """actr_activation(n=0, age_days=0) should handle double-zero gracefully."""
        result = actr_activation(n=0, age_days=0)
        assert not math.isnan(result), "Should not be NaN"
        assert not math.isinf(result), "Should not be Inf"
        assert result > 0, "Should be positive"

    def test_actr_negative_inputs(self):
        """actr_activation with negative n and age should clamp and not crash."""
        result = actr_activation(n=-5, age_days=-100)
        assert not math.isnan(result), "Should not be NaN"
        assert not math.isinf(result), "Should not be Inf"
        assert result > 0, "Should be positive"

    def test_bayesian_update_at_floor(self):
        """Repeated strong contradictions should floor at CONFIDENCE_MIN."""
        confidence = 1.0
        for _ in range(100):
            confidence = bayesian_update(confidence, "strong_contradict")
        assert confidence == pytest.approx(0.1, abs=0.01), f"Should floor at 0.1, got {confidence}"

    def test_bayesian_update_at_ceiling(self):
        """Repeated strong confirmations should cap at CONFIDENCE_MAX."""
        confidence = 1.0
        for _ in range(100):
            confidence = bayesian_update(confidence, "strong_confirm")
        assert confidence == pytest.approx(2.0, abs=0.01), f"Should cap at 2.0, got {confidence}"

    def test_should_push_edge_thresholds(self):
        """should_push at exact threshold boundaries."""
        # Exactly at threshold: activation=0.9, confidence=0.8, hours=24
        assert should_push(0.9, 0.8, 24, False) is False, "0.9 is not > 0.9"
        assert should_push(0.91, 0.81, 24, False) is True, "Just above thresholds"
        # Rate limit: exactly 24 hours
        assert should_push(0.95, 0.95, 24, False) is True, "24 hours is not < 24"
        assert should_push(0.95, 0.95, 23.99, False) is False, "23.99 < 24"


# ────────────────────────────────────────────────────────────────────
# Negative Test: bayesian_update with invalid evidence_type
# ────────────────────────────────────────────────────────────────────
class TestNegativeCases:
    def test_bayesian_update_invalid_evidence_raises(self):
        """bayesian_update with an unknown evidence type MUST raise ValueError."""
        with pytest.raises(ValueError, match="nonexistent_evidence_type"):
            bayesian_update(1.0, "nonexistent_evidence_type")

    def test_bayesian_update_empty_string_raises(self):
        """Empty string evidence type should also raise ValueError."""
        with pytest.raises(ValueError):
            bayesian_update(1.0, "")

    def test_coactivation_single_id_no_op(self):
        """record_co_retrieval with a single ID should be a no-op (no edges created)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "coact.db")
            graph = CoactivationGraph(agent="bob", db_path=db_path)
            graph.record_co_retrieval(["only_one"])
            assocs = graph.get_associations("only_one", min_weight=0.0)
            assert len(assocs) == 0, "Single ID should not create any edges"

    def test_predict_next_no_patterns_returns_empty(self):
        """predict_next_memories with no stored sequences should return empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "coact.db")
            result = predict_next_memories(
                current_memory_ids=["mem_a"],
                agent="bob",
                db_path=db_path,
            )
            assert result == [], "No stored sequences should yield no predictions"


# ────────────────────────────────────────────────────────────────────
# Edge Case 4: Spaced repetition bonus with malformed timestamps
# ────────────────────────────────────────────────────────────────────
class TestSpacedRepetitionEdgeCases:
    def test_malformed_timestamps_graceful_fallback(self):
        """actr_activation with garbage timestamps should fall back to spread_bonus=1.0."""
        result = actr_activation(
            n=5, age_days=3.0,
            access_timestamps=["not-a-date", "also-garbage", "nope"]
        )
        baseline = actr_activation(n=5, age_days=3.0, access_timestamps=None)
        assert result == pytest.approx(baseline, rel=0.01), \
            "Malformed timestamps should fall back to no bonus"

    def test_single_timestamp_no_crash(self):
        """actr_activation with a single timestamp (< 2 needed) should not crash."""
        result = actr_activation(
            n=5, age_days=3.0,
            access_timestamps=["2026-03-01T12:00:00"]
        )
        assert not math.isnan(result)
        assert result > 0

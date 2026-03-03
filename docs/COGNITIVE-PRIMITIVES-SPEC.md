# Cognitive Memory Primitives — Implementation Specification

*Author: Spec Writer Agent | Date: 2026-03-04 | Status: DRAFT*
*Reviewed by: Team Lead*

---

## Executive Summary

This specification defines five cognitive memory primitives that transform Circle Zero's semantic memory from a passive similarity-search system into an **active cognitive architecture**. Each primitive is drawn from established cognitive science and neuroscience research, adapted for a distributed AI memory system while maintaining the project's philosophical commitments.

### The Gap

The current retrieval pipeline (`src/retrieval/recall.py:44`) scores memories using three signals:

```
final_score = similarity × time_weight × source_weight
```

Where `time_weight` (`src/pipeline/embedder.py:47`) is a logarithmic decay function:

```
time_weight = 1 / (1 + ln(age_days + 1))
```

This treats all memories as equally important regardless of how often they've been accessed, whether they've been confirmed or contradicted, or what other memories they co-occur with. A procedure used daily and a stale note from six months ago receive identical treatment if their embeddings happen to be equidistant from a query.

### The Solution — Five Primitives

| Phase | Primitive | Cognitive Basis | What It Adds |
|-------|-----------|----------------|--------------|
| 1 | **ACT-R Activation** | Anderson (1993) | Frequently accessed, recently used memories surface first |
| 2 | **Bayesian Confidence** | Bayes (1763), Laplace (1812) | Memories carry confidence scores; contradictions reduce certainty |
| 3 | **Hebbian Co-activation** | Hebb (1949) | Memories retrieved together strengthen their association |
| 4 | **Sequential Patterns** | Temporal learning | Common retrieval sequences enable predictive recall |
| 5 | **Push Triggers** | Proactive retrieval | High-activation memories surface without being queried |

### Philosophy Alignment

Per `docs/PHILOSOPHY.md`, agents are **cognitive extensions** — augmentations of human intelligence. These primitives don't bolt on artificial intelligence features; they replicate mechanisms that biological memory already uses. The human brain activates frequently-used memories faster (ACT-R), downgrades contradicted beliefs (Bayesian), strengthens co-occurring neural pathways (Hebbian), and surfaces relevant memories unprompted (push). We are extending the organism's memory to operate on the same principles as the organism's brain.

The markdown-first, git-tracked philosophy is preserved: all new metadata fields are stored in Chroma (agent web) alongside existing fields, while markdown files (human web) remain human-readable without cognitive metadata noise.

### Scoring Formula Evolution

**Current** (`recall.py:44`):
```
score = similarity × time_weight(age_days) × source_weight
```

**After Phase 1** (ACT-R replaces `time_weight`):
```
score = similarity × activation(n, age_days) × source_weight
```

**After Phase 2** (Bayesian confidence added):
```
score = similarity × activation(n, age_days) × source_weight × confidence
```

**After Phase 3** (Hebbian boost added):
```
score = similarity × activation(n, age_days) × source_weight × confidence + association_boost
```

Each phase is independently deployable and backward-compatible. The scoring formula only grows; it never removes a working signal.

---

## Phase 1: ACT-R Base-Level Activation

### Cognitive Science Foundation

**ACT-R** (Adaptive Control of Thought—Rational) is a cognitive architecture developed by John R. Anderson at Carnegie Mellon University. Its base-level activation equation models how memory accessibility decays over time but is reinforced by repeated access — the same mechanism that makes your phone number easy to recall but your childhood address harder.

> **Citation:** Anderson, J.R. (1993). *Rules of the Mind.* Lawrence Erlbaum Associates. Chapter 4: "Activation of Declarative Knowledge."

The key insight: memory strength is a function of **both** recency and frequency, not recency alone. Our current `time_weight` function captures recency but completely ignores frequency. A procedure consulted daily should be far more accessible than one consulted once six months ago, even if both are semantically similar to a query.

### Formula

The base-level activation *B* for a memory chunk is:

```
B = ln(n + 1) − 0.5 × ln(age_days / (n + 1))
```

Where:
- `n` = number of times the memory has been accessed (retrieved or explicitly referenced)
- `age_days` = days since the memory was first created (not last accessed)
- `ln` = natural logarithm
- `0.5` = decay parameter *d* (Anderson's canonical value; tunable)

**Behavior characteristics:**

| Scenario | n | age_days | B | Interpretation |
|----------|---|----------|---|----------------|
| Brand new memory | 1 | 0.01 | 1.39 | High — just created |
| Used daily for a week | 7 | 7 | 1.95 | Very high — frequent use |
| Used once, month old | 1 | 30 | −0.06 | Low — forgotten |
| Used 20× over 6 months | 20 | 180 | 2.30 | High — well-practiced |
| Used once, year old | 1 | 365 | −0.92 | Very low — deep decay |
| Burst: 5× in 1 day | 5 | 1 | 2.48 | Spike — temporary salience |

**Conversion to weight:** The activation value *B* is unbounded (can be negative). We convert it to a 0–1 weight for multiplication with the existing scoring pipeline:

```
activation_weight = sigmoid(B) = 1 / (1 + e^(−B))
```

This maps *B* ∈ (−∞, +∞) to a smooth 0–1 range, with B=0 → 0.5 (neutral), B>0 → increasingly strong, B<0 → increasingly weak.

### Spaced Repetition Bonus

Memories accessed at **regular intervals** (spaced repetition) should receive a bonus over memories accessed in bursts. A procedure consulted once a week for a month is more durably learned than one consulted five times in one day.

**Temporal spread** measures access regularity:

```
temporal_spread = std_dev(intervals between consecutive accesses)
```

Where `intervals` = `[t₂-t₁, t₃-t₂, ..., tₙ-tₙ₋₁]` computed from the `last_n_access_timestamps` array.

**Spread bonus:**

```
mean_interval = mean(intervals)
spread_bonus = 1 + 0.2 × (1 − min(1, temporal_spread / mean_interval))
```

Where `mean_interval` is the arithmetic mean of the intervals array (average time between consecutive accesses).

- Perfectly even spacing → bonus = 1.2 (20% boost)
- Completely irregular → bonus = 1.0 (no boost)
- The 0.2 multiplier is conservative; tunable via config

**Final activation weight with spread:**

```
activation_weight = sigmoid(B) × spread_bonus
```

### New Metadata Fields

Two new fields are added to the metadata dict in `store.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `access_count` | int | `1` | Number of times this memory has been retrieved or explicitly accessed |
| `last_n_access_timestamps` | string (JSON array) | `"[]"` | ISO timestamps of last N accesses, capped at 20. Stored as JSON string per Chroma's metadata constraints |

**Backward compatibility:** Existing memories without these fields are treated as `access_count=1` and `last_n_access_timestamps=[]`. No migration required — the code handles missing fields with defaults.

### Implementation Details

#### New Function: `actr_activation()` — `src/pipeline/embedder.py`

Replaces `time_weight()` in the scoring pipeline.

```python
def actr_activation(
    n: int,
    age_days: float,
    access_timestamps: list[str] | None = None,
    decay: float = 0.5,
    spread_bonus_weight: float = 0.2,
) -> float:
    """Compute ACT-R base-level activation weight for a memory chunk.

    Implements Anderson's (1993) base-level learning equation adapted for
    semantic memory retrieval, with an optional spaced-repetition bonus.

    Formula:
        B = ln(n + 1) - decay * ln(age_days / (n + 1))
        activation_weight = sigmoid(B) * spread_bonus

    Args:
        n: Access count (times retrieved). Minimum 1.
        age_days: Days since memory creation. Must be >= 0.
        access_timestamps: ISO timestamps of recent accesses (up to 20).
            Used to compute temporal spread for spaced repetition bonus.
            If None or fewer than 2 entries, spread bonus is 1.0 (neutral).
        decay: Decay parameter d from ACT-R. Default 0.5 (Anderson's value).
            Higher values = faster decay for infrequent memories.
        spread_bonus_weight: Maximum bonus for evenly-spaced access (0-1).
            Default 0.2 = up to 20% boost for perfect spacing.

    Returns:
        Float in (0, ~1.2] representing activation-based weight.
        Values > 1.0 are possible due to spread bonus.

    Examples:
        >>> actr_activation(1, 0.01)   # Brand new
        0.80
        >>> actr_activation(7, 7)      # Daily use for a week
        0.88
        >>> actr_activation(1, 30)     # Accessed once, month old
        0.49
        >>> actr_activation(1, 365)    # Accessed once, year old
        0.28

    Reference:
        Anderson, J.R. (1993). Rules of the Mind. Lawrence Erlbaum Associates.
    """
```

**Implementation notes:**
- `n` is clamped to `max(1, n)` — a memory always has at least one access (creation)
- `age_days` is clamped to `max(0.01, age_days)` — avoids `ln(0)`
- The sigmoid conversion ensures the output is always positive and bounded
- `access_timestamps` are parsed with the same `parse_date()` from `recall.py:36`

#### Changes to `compute_final_score()` — `src/retrieval/recall.py:44`

**Before:**
```python
def compute_final_score(
    distance: float,
    date_str: str,
    source_weight: float = 1.0,
) -> float:
    similarity = 1.0 / (1.0 + distance)
    date = parse_date(date_str)
    if date:
        age_days = (datetime.now(timezone.utc) - date).days
        tw = time_weight(max(0, age_days))
    else:
        tw = 0.5
    return similarity * tw * source_weight
```

**After:**
```python
def compute_final_score(
    distance: float,
    date_str: str,
    source_weight: float = 1.0,
    access_count: int = 1,
    access_timestamps: list[str] | None = None,
) -> float:
    """Combine vector similarity, ACT-R activation, and source weight.

    Replaces the previous time_weight-only approach with ACT-R
    base-level activation that accounts for both recency and frequency.

    Args:
        distance: Chroma L2 distance (lower = more similar).
        date_str: ISO date string of memory creation.
        source_weight: Weight from SOURCE_WEIGHTS dict.
        access_count: Number of times this memory has been accessed.
            Default 1 for backward compatibility with pre-ACT-R memories.
        access_timestamps: JSON-decoded list of ISO timestamp strings.
            Used for spaced repetition bonus calculation.

    Returns:
        Float score. Higher = more relevant.
    """
    similarity = 1.0 / (1.0 + distance)

    date = parse_date(date_str)
    if date:
        age_days = (datetime.now(timezone.utc) - date).days
        aw = actr_activation(
            n=max(1, access_count),
            age_days=max(0, age_days),
            access_timestamps=access_timestamps,
        )
    else:
        aw = 0.5  # Unknown date gets neutral activation

    return similarity * aw * source_weight
```

#### Changes to `search()` — `src/retrieval/recall.py:122`

The `search()` function must:

1. **Extract** `access_count` and `last_n_access_timestamps` from each result's metadata (lines ~160, ~190 in the current segment/summary loops)
2. **Pass** them to `compute_final_score()`
3. **Increment** `access_count` and append a timestamp for every returned result

**Access count update** (new helper function in `recall.py`):

```python
def record_access(store: VectorStore, memory_id: str, metadata: dict, source: str) -> None:
    """Record an access event for a retrieved memory.

    Increments access_count and appends current timestamp to
    last_n_access_timestamps (capped at 20 entries, FIFO eviction).

    Args:
        store: VectorStore instance for the current agent.
        memory_id: Chroma document ID.
        metadata: Current metadata dict from the result.
        source: "segment" or "summary" — determines which collection to update.
    """
```

**Important:** Access recording is **fire-and-forget** — it must not block the search response. Use a deferred write or background task. Failed access recording should log a warning but never fail the search.

#### Changes to `add_segment()` — `src/pipeline/store.py:52`

Add the new fields to the metadata dict at line ~70:

```python
metadata = {
    # ... existing fields ...
    "access_count": 1,
    "last_n_access_timestamps": json.dumps([]),
}
```

Same change applies to `add_session_summary()` at `store.py:90`.

#### Deprecation of `time_weight()` — `src/pipeline/embedder.py:47`

The existing `time_weight()` function is **not deleted** — it is marked deprecated with a module-level note. It remains importable for any external code that may reference it, but `compute_final_score()` no longer calls it.

```python
def time_weight(age_days: float) -> float:
    """Logarithmic time decay: recent sessions rank higher, old ones never vanish.

    .. deprecated:: Phase 1 (Cognitive Primitives)
        Replaced by actr_activation() which accounts for both recency AND frequency.
        Retained for backward compatibility. Will be removed in a future version.
    """
    return 1.0 / (1.0 + math.log(age_days + 1))
```

### Edge Cases

| Case | Behavior | Rationale |
|------|----------|-----------|
| `n=0` (impossible — creation counts as 1) | Clamp to `n=1` | A memory always has at least one "access" — its creation |
| `age_days=0` (just created) | Clamp to `age_days=0.01` | Avoids `ln(0)` in the formula; result is near-maximum activation |
| `age_days=0, n=1` (brand new memory) | B ≈ 1.39, weight ≈ 0.80 | New memories start strong but not maximum |
| Very old memory, high n (e.g., 365 days, 100 accesses) | B ≈ 4.23, weight ≈ 0.99 | Heavily used memories resist decay |
| Burst access (5× in 1 hour) | High B but spread_bonus = 1.0 (no bonus) | Bursts don't get spacing credit |
| `last_n_access_timestamps` has >20 entries | Truncate to most recent 20 (FIFO) | Caps metadata size; 20 entries is sufficient for spread calculation |
| Pre-existing memories (no `access_count` field) | Default to `access_count=1, timestamps=[]` | Backward compat — old memories treated as accessed once |

---

## Phase 2: Bayesian Confidence Scoring

### Cognitive Science Foundation

**Bayesian inference** models how rational agents update beliefs when new evidence arrives. In the context of memory, confidence should increase when a memory is confirmed by subsequent information and decrease when contradicted.

> **Citation:** Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances." *Philosophical Transactions of the Royal Society of London,* 53, 370–418.

> **Citation:** Laplace, P.S. (1812). *Th\u00e9orie analytique des probabilit\u00e9s.* Paris: Courcier.

Humans naturally do this: if you learn "the meeting is at 3pm" and then someone says "actually it was moved to 4pm," you don't hold both with equal weight. The first belief's confidence drops. Our memory system should do the same.

### Formula

**Bayes' theorem:**

```
P(H|E) = P(E|H) × P(H) / P(E)
```

**Practical confidence update for memory systems:**

When new evidence `E` arrives that relates to an existing memory `H`:

```
confidence_new = confidence_old × likelihood_ratio
```

Where `likelihood_ratio` captures how the new evidence affects the memory's credibility:

| Evidence Type | Likelihood Ratio | Effect |
|---------------|-----------------|--------|
| Strong confirmation (semantic similarity > 0.90, same conclusion) | 1.15 | +15% confidence |
| Weak confirmation (similarity 0.70–0.90, compatible) | 1.05 | +5% confidence |
| Neutral (similarity < 0.70 or unrelated) | 1.00 | No change |
| Weak contradiction (similarity > 0.70, different conclusion) | 0.85 | −15% confidence |
| Strong contradiction (similarity > 0.85, opposite conclusion) | 0.60 | −40% confidence |

**Confidence bounds:** Confidence is clamped to `[0.1, 2.0]`.
- Floor of 0.1: Even heavily contradicted memories don't fully vanish — they might be re-confirmed later
- Ceiling of 2.0: Prevents runaway confirmation bias
- Default: 1.0 (neutral)

**Integration with scoring pipeline** (`recall.py:44`, post-Phase 2):

```
score = similarity × activation_weight × source_weight × confidence
```

Confidence acts as a direct multiplier. A memory with confidence 0.5 contributes half the score of an identical memory with confidence 1.0.

### Contradiction Detection

The system must detect when a newly stored memory contradicts an existing one. This is the trigger for Bayesian updates.

**Detection algorithm:**

1. When `store_memory` is called, embed the new content
2. Search existing memories with the new embedding (top 10, same agent scope)
3. For each result with **cosine similarity > 0.85** (high topical overlap):
   a. Compare the semantic content for agreement/disagreement
   b. Use an LLM classification call (lightweight, ~100 tokens):
      - Prompt: "Do these two statements agree, disagree, or discuss unrelated aspects? Statement A: {existing}. Statement B: {new}. Answer: agree/disagree/unrelated"
   c. Apply the appropriate likelihood ratio to the existing memory's confidence
4. The new memory starts with confidence 1.0 (or higher if it contradicts a low-confidence memory)

**Cost control:** The LLM call for contradiction detection is only triggered when cosine similarity exceeds 0.85. Based on typical memory distribution, this fires for ~5-15% of new memories. The classification prompt is minimal (~100 input tokens, ~5 output tokens).

### New Metadata Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `confidence` | float | `1.0` | Bayesian confidence score, range [0.1, 2.0] |

Added to the metadata dict in both `add_segment()` (`store.py:52`) and `add_session_summary()` (`store.py:90`).

**Backward compatibility:** Existing memories without `confidence` are treated as `confidence=1.0`. No migration required.

### Implementation Details

#### New Module: `src/retrieval/confidence.py`

```python
"""Bayesian confidence scoring for semantic memories.

Implements confidence updates based on Bayes' theorem, applied when
new memories are stored that may confirm or contradict existing ones.

References:
    Bayes, T. (1763). Philosophical Transactions, 53, 370-418.
    Laplace, P.S. (1812). Theorie analytique des probabilites.
"""

CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 2.0
CONFIDENCE_DEFAULT = 1.0

SIMILARITY_THRESHOLD = 0.85  # Minimum cosine sim to trigger contradiction check

LIKELIHOOD_RATIOS = {
    "strong_confirm": 1.15,
    "weak_confirm": 1.05,
    "neutral": 1.00,
    "weak_contradict": 0.85,
    "strong_contradict": 0.60,
}


def bayesian_update(
    prior_confidence: float,
    evidence_type: str,
) -> float:
    """Update a memory's confidence based on new evidence.

    Applies Bayes' theorem in simplified form: the likelihood ratio
    for the evidence type is multiplied against the prior confidence,
    then clamped to [CONFIDENCE_MIN, CONFIDENCE_MAX].

    Args:
        prior_confidence: Current confidence value of the existing memory.
        evidence_type: One of "strong_confirm", "weak_confirm", "neutral",
            "weak_contradict", "strong_contradict".

    Returns:
        Updated confidence value, clamped to [0.1, 2.0].

    Raises:
        ValueError: If evidence_type is not recognized.

    Examples:
        >>> bayesian_update(1.0, "strong_confirm")
        1.15
        >>> bayesian_update(1.0, "strong_contradict")
        0.60
        >>> bayesian_update(0.5, "strong_contradict")
        0.30
        >>> bayesian_update(0.1, "strong_contradict")
        0.1  # Floor prevents full erasure
    """


def detect_contradictions(
    new_embedding: list[float],
    new_text: str,
    store: "VectorStore",
    agent: str,
    top_k: int = 10,
) -> list[dict]:
    """Find existing memories that may be contradicted by new content.

    Searches for high-similarity existing memories (cosine sim > 0.85),
    then classifies each pair as agreeing, disagreeing, or unrelated.

    Args:
        new_embedding: Vector embedding of the new memory.
        new_text: Raw text of the new memory.
        store: VectorStore instance for searching.
        agent: Agent identifier for namespace scoping.
        top_k: Number of candidates to check. Default 10.

    Returns:
        List of dicts, each containing:
            - memory_id: str — Chroma document ID of the existing memory
            - similarity: float — cosine similarity with new memory
            - evidence_type: str — classification result
            - old_confidence: float — current confidence value
            - new_confidence: float — updated confidence value
    """


def classify_relationship(
    statement_a: str,
    statement_b: str,
) -> str:
    """Classify whether two statements agree, disagree, or are unrelated.

    Uses a lightweight LLM call (~100 input tokens) to determine the
    semantic relationship between two memory statements.

    Args:
        statement_a: Text of the existing memory.
        statement_b: Text of the new memory.

    Returns:
        One of: "strong_confirm", "weak_confirm", "neutral",
        "weak_contradict", "strong_contradict".
    """
```

#### Integration with `store_memory` Pipeline

The contradiction detection integrates into the `store_memory` flow (as defined in `docs/MCP-MEMORY-SERVER-SPEC.md`, Section 2):

**Current pipeline** (parallel):
1. Embed via ollama
2. Classify metadata via LLM
3. Write markdown
4. Index in Chroma

**Updated pipeline** (parallel where possible):
1. Embed via ollama
2. Classify metadata via LLM
3. **Detect contradictions** against existing memories (requires step 1's embedding)
4. **Apply Bayesian updates** to any contradicted memories
5. Write markdown
6. Index in Chroma (new memory with confidence=1.0)

Steps 1 and 2 remain parallel. Step 3 depends on step 1 (needs the embedding). Steps 4, 5, 6 can proceed in parallel after step 3.

#### Changes to `compute_final_score()` — `src/retrieval/recall.py:44`

Add `confidence` parameter:

```python
def compute_final_score(
    distance: float,
    date_str: str,
    source_weight: float = 1.0,
    access_count: int = 1,
    access_timestamps: list[str] | None = None,
    confidence: float = 1.0,
) -> float:
    """Combine similarity, ACT-R activation, source weight, and confidence.

    Args:
        ...existing args...
        confidence: Bayesian confidence score [0.1, 2.0]. Default 1.0.

    Returns:
        Float score. Higher = more relevant.
    """
    similarity = 1.0 / (1.0 + distance)

    date = parse_date(date_str)
    if date:
        age_days = (datetime.now(timezone.utc) - date).days
        aw = actr_activation(
            n=max(1, access_count),
            age_days=max(0, age_days),
            access_timestamps=access_timestamps,
        )
    else:
        aw = 0.5

    return similarity * aw * source_weight * confidence
```

### Edge Cases

| Case | Behavior | Rationale |
|------|----------|-----------|
| New memory contradicts a vault memory (source_weight=2.0) | Confidence update still applies, but vault memories start with implicit high trust | Vault memories are human-curated; they should be hard to downgrade. Consider adding a `vault_floor` config (e.g., 0.8) |
| Self-contradiction (memory contradicts itself) | Similarity check catches this — but classification should return "neutral" for same-session content | Handled by session_id deduplication in detect_contradictions |
| Cascade contradiction (A contradicts B, B contradicts C) | Only direct contradictions are processed; no transitive updates | Keeps complexity bounded. Phase 5+ could add cascade detection |
| LLM classification fails or times out | Default to `evidence_type="neutral"` (no confidence change) | Fail-safe: never degrade confidence on an error |
| Very high confidence (>1.5) being contradicted | Single strong_contradict brings 1.5 → 0.9, still above neutral | Designed: well-confirmed memories require multiple contradictions to erode |
| Pre-existing memories (no `confidence` field) | Default to `confidence=1.0` | Backward compat — same as new memories |

---

## Data Model Changes — Consolidated

### Metadata Field Additions

All changes are to the Chroma metadata dicts in `src/pipeline/store.py`. Chroma stores metadata as flat key-value pairs; complex types are JSON-encoded strings.

| Field | Type (Python) | Chroma Type | Default | Added In | Used By |
|-------|---------------|-------------|---------|----------|---------|
| `access_count` | int | int | `1` | Phase 1 | `actr_activation()` |
| `last_n_access_timestamps` | list[str] (JSON) | string | `"[]"` | Phase 1 | `actr_activation()`, `record_access()` |
| `confidence` | float | float | `1.0` | Phase 2 | `compute_final_score()`, `bayesian_update()` |
| `last_pushed_at` | str (ISO 8601) or None | string | `null` | Phase 5 | `should_push()` |

**Note on `coactivation_ids`:** This field was originally planned as a Chroma metadata cache but is **not implemented**. Phase 3 co-activation data is stored exclusively in SQLite (`coactivation.db`). The field is removed from this table. If future profiling reveals that SQLite lookups during query-time boosting are too slow, a denormalized Chroma cache can be reconsidered.

**Backward compatibility for `last_pushed_at`:** Memories without this field (all pre-Phase 5 memories) default to `null`, meaning "never pushed." The push scan treats these as eligible for evaluation, which is the desired behavior — it allows the system to discover high-activation historical memories.

### Affected Functions

| Function | File:Line | Phase 1 Changes | Phase 2 Changes |
|----------|-----------|-----------------|-----------------|
| `add_segment()` | `store.py:52` | Add `access_count`, `last_n_access_timestamps` to metadata | Add `confidence` to metadata |
| `add_session_summary()` | `store.py:90` | Add `access_count`, `last_n_access_timestamps` to metadata | Add `confidence` to metadata |
| `compute_final_score()` | `recall.py:44` | Replace `time_weight()` call with `actr_activation()` call; add `access_count` and `access_timestamps` params | Add `confidence` param, multiply into score |
| `search()` | `recall.py:122` | Extract `access_count`/`last_n_access_timestamps` from metadata, pass to `compute_final_score()`; call `record_access()` for returned results | Extract `confidence` from metadata, pass to `compute_final_score()` |
| `time_weight()` | `embedder.py:47` | Deprecated (kept for compat) | No change |

### New Functions

| Function | File | Phase | Purpose |
|----------|------|-------|---------|
| `actr_activation()` | `src/pipeline/embedder.py` | 1 | ACT-R base-level activation weight |
| `record_access()` | `src/retrieval/recall.py` | 1 | Increment access count + append timestamp on retrieval |
| `bayesian_update()` | `src/retrieval/confidence.py` | 2 | Update confidence given evidence type |
| `detect_contradictions()` | `src/retrieval/confidence.py` | 2 | Find potential contradictions for new memories |
| `classify_relationship()` | `src/retrieval/confidence.py` | 2 | LLM call to classify agree/disagree/unrelated |

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| `src/retrieval/confidence.py` | 2 | Bayesian confidence module |
| `src/graph/coactivation.py` | 3 | Hebbian co-activation graph |
| `scripts/backfill_cognitive_fields.py` | 1 | Cold-start backfill script |

---

## Phase 3: Hebbian Co-activation Graph

### Cognitive Science Foundation

**Hebbian learning** is the oldest and most fundamental principle of neural plasticity: neurons that fire together wire together. When two memories are consistently retrieved in the same context, the association between them should strengthen — making it more likely that retrieving one will boost the other.

> **Citation:** Hebb, D.O. (1949). *The Organization of Behavior: A Neuropsychological Theory.* New York: Wiley. Chapter 4: "The First Stage of Perception: Growth of the Assembly."

In practice, this means: if every time an agent searches for "silver technical analysis" they also retrieve "BeeBee pattern" and "mean reversion," those three memories should develop strong mutual associations. Future searches for any one should boost the others.

### Weight Update Rules

**Strengthening (co-retrieval):**

When memories *i* and *j* appear in the same search result set:

```
w_ij += α
```

Where:
- `w_ij` = association weight between memories i and j
- `α` = learning rate (default: 0.1)

**Decay (time-based):**

All weights decay on each access cycle:

```
w_ij *= (1 − λ)
```

Where:
- `λ` = decay rate (default: 0.05 per day)

**Weight bounds:** `w_ij ∈ [0, 5.0]` — capped to prevent runaway associations.

### Data Model: SQLite Co-activation Table

Chroma's flat metadata model is insufficient for graph relationships. Co-activation edges are stored in a lightweight SQLite database alongside the Chroma store.

**Database location:** `{vector_store_path}/{agent}/coactivation.db`

```sql
CREATE TABLE IF NOT EXISTS coactivation_edges (
    memory_id_a TEXT NOT NULL,
    memory_id_b TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.1,
    co_retrieval_count INTEGER NOT NULL DEFAULT 1,
    last_co_retrieval TEXT NOT NULL,  -- ISO timestamp
    created_at TEXT NOT NULL,         -- ISO timestamp
    PRIMARY KEY (memory_id_a, memory_id_b)
);

-- Index for fast lookup of all edges for a given memory
CREATE INDEX IF NOT EXISTS idx_edges_a ON coactivation_edges(memory_id_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON coactivation_edges(memory_id_b);

-- Enforce canonical ordering: memory_id_a < memory_id_b (alphabetically)
-- This prevents duplicate edges (a→b and b→a)
```

**Canonical ordering:** Edges are always stored with `memory_id_a < memory_id_b` (string comparison). The application layer normalizes before insert/lookup.

### New Module: `src/graph/coactivation.py`

```python
"""Hebbian co-activation graph for semantic memories.

Tracks which memories are frequently retrieved together and uses
association strength to boost related memories during search.

Storage: SQLite database per agent, co-located with Chroma store.

References:
    Hebb, D.O. (1949). The Organization of Behavior. Wiley.
"""

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DECAY_RATE = 0.05  # Per day
MAX_WEIGHT = 5.0


class CoactivationGraph:
    """Manages co-activation edges between memories.

    Each edge represents a learned association between two memories
    that have been retrieved together. Weights strengthen with
    co-retrieval and decay over time.
    """

    def __init__(self, agent: str, db_path: str | None = None):
        """Initialize the co-activation graph for an agent.

        Args:
            agent: Agent identifier (bob, patterson, dean, shared).
            db_path: Override path for SQLite database.
                Default: {VECTOR_STORE_PATH}/{agent}/coactivation.db
        """

    def record_co_retrieval(self, memory_ids: list[str]) -> None:
        """Record that a set of memories were retrieved together.

        Creates or strengthens edges between all pairs of memories
        in the result set. O(n²) where n = len(memory_ids), but
        n is bounded by top_k (typically 5-10).

        Args:
            memory_ids: List of Chroma document IDs from a search result.
        """

    def get_associations(
        self, memory_id: str, min_weight: float = 0.3, limit: int = 10
    ) -> list[dict]:
        """Get memories associated with a given memory.

        Returns edges sorted by weight (strongest first), filtered
        to those above min_weight threshold.

        Args:
            memory_id: Chroma document ID to find associations for.
            min_weight: Minimum edge weight to include. Default 0.3.
            limit: Maximum associations to return. Default 10.

        Returns:
            List of dicts: {memory_id, weight, co_retrieval_count, last_co_retrieval}
        """

    def apply_decay(self, days_elapsed: float = 1.0) -> int:
        """Apply time-based decay to all edge weights.

        Removes edges that fall below a minimum threshold (0.05)
        to prevent unbounded table growth.

        Args:
            days_elapsed: Number of days of decay to apply. Default 1.0.

        Returns:
            Number of edges pruned (dropped below threshold).
        """

    def hebbian_boost(
        self, query_results: list[dict], boost_factor: float = 0.15
    ) -> list[dict]:
        """Apply association-based score boosting to search results.

        For each result, checks if any other results in the set have
        strong co-activation edges. If so, adds a boost proportional
        to the edge weight.

        boost = sum(w_ij * boost_factor) for all j in results where i≠j

        The boost is ADDITIVE (not multiplicative) to avoid compounding
        with the existing multiplicative factors in compute_final_score.

        Args:
            query_results: List of result dicts from search(), each with
                a "score" key and metadata containing the memory ID.
            boost_factor: Scaling factor for association boost. Default 0.15.

        Returns:
            Same list with updated "score" values.
        """
```

### Integration with `search()` — `src/retrieval/recall.py:122`

After computing scores and before deduplication, apply Hebbian boosting:

```python
# After scoring, before dedup (around line ~200 in current search())
graph = CoactivationGraph(agent=agent)
results = graph.hebbian_boost(results)

# After dedup, record co-retrieval for returned results
memory_ids = [r["metadata"].get("session_id", "") for r in deduped[:top_k]]
graph.record_co_retrieval(memory_ids)
```

### Edge Cases

| Case | Behavior | Rationale |
|------|----------|-----------|
| Single-result search | No co-activation recorded (need ≥2) | No pair to associate |
| Same memory appears as segment + summary | Use segment ID for edge (more specific) | Dedup handles this after scoring |
| Very large result sets (top_k=50) | O(n²) = 2500 edge updates; still fast for SQLite | Consider batching if top_k > 20 |
| Stale edges (not co-retrieved in 90+ days) | `apply_decay()` prunes edges below 0.05 | Called daily by cron or on startup |
| Cross-agent associations | Not supported — each agent has its own graph | Keeps isolation consistent with Chroma model |

---

## Phase 4: Sequential Pattern Detection

### Overview

Sequential patterns capture **temporal ordering** in memory retrieval. If an agent consistently searches for topic A, then B, then C in that order, the system should recognize this sequence and predictively pre-fetch B when A is retrieved.

This is a **lighter implementation** than Phases 1–3 — it logs sequences and enables predictive retrieval but does not require a new cognitive model.

### Data Model

**SQLite table** (added to the same `coactivation.db` per agent):

```sql
CREATE TABLE IF NOT EXISTS retrieval_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence TEXT NOT NULL,      -- JSON array of memory IDs in retrieval order
    query_text TEXT NOT NULL,    -- The search query that produced this sequence
    agent TEXT NOT NULL,
    timestamp TEXT NOT NULL,     -- ISO timestamp
    session_context TEXT         -- Optional: what session/task was active
);

CREATE INDEX IF NOT EXISTS idx_seq_agent ON retrieval_sequences(agent);
CREATE INDEX IF NOT EXISTS idx_seq_timestamp ON retrieval_sequences(timestamp);
```

### Sequence Logging

Every `search()` call logs the ordered result IDs:

```python
def log_retrieval_sequence(
    memory_ids: list[str],
    query_text: str,
    agent: str,
    session_context: str | None = None,
) -> None:
    """Log an ordered retrieval sequence for pattern detection.

    Called after every search() to record what memories were returned
    and in what order. Pattern detection runs asynchronously.

    Args:
        memory_ids: Ordered list of returned memory IDs.
        query_text: The original search query.
        agent: Agent identifier.
        session_context: Optional context about the active task/session.
    """
```

### Predictive Retrieval

```python
def predict_next_memories(
    current_memory_ids: list[str],
    agent: str,
    lookback_days: int = 30,
    min_pattern_count: int = 3,
) -> list[dict]:
    """Predict which memories are likely to be needed next.

    Analyzes retrieval_sequences to find patterns where the current
    set of memory IDs appeared as a prefix. Returns the most common
    "next" memories from those sequences.

    Args:
        current_memory_ids: The memories just retrieved.
        agent: Agent identifier.
        lookback_days: Only consider sequences from this window.
        min_pattern_count: Minimum times a pattern must occur to be considered.

    Returns:
        List of {memory_id, prediction_confidence, pattern_count} dicts,
        sorted by confidence (highest first).
    """
```

### Integration Point

Predictive retrieval is **not** integrated into the core `search()` scoring pipeline. Instead, it's exposed as a separate function that the MCP server can call to provide "you might also want" suggestions alongside search results.

**MCP tool addition** (see MCP Integration section below):
- Add `predicted_next` field to `search_memory` response (optional, off by default)
- New parameter: `predict: bool = False` on `search_memory`

---

## Phase 5: Push Triggers (Proactive Memory Surfacing)

### Overview

Push triggers allow high-activation memories to surface **without being explicitly queried**. This is the "tip of the tongue" phenomenon — when a memory is so strongly activated that it pushes itself into awareness.

### Design: Heartbeat-Driven

Circle Zero already has a heartbeat/cron infrastructure. Push triggers are implemented as a **periodic scan** rather than a real-time event system:

1. **Cron job** (runs every 30 minutes during active hours, hourly during quiet hours)
2. Scans all memories for the agent with `activation_weight > threshold` (default: 0.9)
3. Filters for memories that are:
   - High activation but not recently surfaced (>24h since last push)
   - Related to current context (if available from briefing.json or active tasks)
   - Not already in the agent's active working set
4. Delivers matches via `#cross-communication` or the agent's notification mechanism

### Trigger Conditions

```python
def should_push(
    activation_weight: float,
    confidence: float,
    hours_since_last_push: float,
    has_unresolved_threads: bool,
) -> bool:
    """Determine if a memory should be proactively surfaced.

    Args:
        activation_weight: ACT-R activation weight (from Phase 1).
        confidence: Bayesian confidence (from Phase 2).
        hours_since_last_push: Hours since this memory was last pushed.
        has_unresolved_threads: Whether the memory has open action items.

    Returns:
        True if the memory should be pushed to the agent.
    """
    if hours_since_last_push < 24:
        return False  # Rate limit: max once per day per memory
    if activation_weight > 0.9 and confidence > 0.8:
        return True   # High activation + high confidence = push
    if has_unresolved_threads and activation_weight > 0.7:
        return True   # Open threads with moderate activation = reminder
    return False
```

### New Metadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `last_pushed_at` | string (ISO) | `null` | When this memory was last proactively surfaced |

### Integration

Push triggers are a **read-only consumer** of Phases 1–2 data. They don't modify the scoring pipeline. Implementation is a standalone script or cron job that:

1. Queries Chroma for high-activation memories
2. Filters using `should_push()` logic
3. Formats and delivers notifications
4. Updates `last_pushed_at` metadata

This phase is intentionally lightweight because the existing heartbeat infrastructure already handles 80% of proactive surfacing. Push triggers add the remaining 20%: memories that are contextually important right now based on their cognitive scores.

---

## Cold Start & Backfill

### Problem

Existing memories (pre-cognitive-primitives) have no `access_count`, `last_n_access_timestamps`, or `confidence` fields. While the code defaults these gracefully, all pre-existing memories start with identical ACT-R profiles (`n=1, age_days=actual_age`), which means old, heavily-referenced memories get no activation credit for their historical importance.

### Backfill Script: `scripts/backfill_cognitive_fields.py`

```python
"""Backfill cognitive primitive metadata for pre-existing memories.

Reads all memories from Chroma, adds default cognitive fields,
and optionally infers access_count from heuristic signals.

Usage:
    python scripts/backfill_cognitive_fields.py --agent bob [--dry-run] [--infer-access]

Flags:
    --agent: Which agent's store to backfill (bob, patterson, dean, shared)
    --dry-run: Print changes without writing
    --infer-access: Use heuristic to estimate access counts from:
        - source_weight (vault=high access, segment=low)
        - age_days (older memories that still exist were presumably useful)
        - domain frequency (memories in frequently-searched domains get credit)
"""
```

**Heuristic access count inference:**

| Signal | Inferred `access_count` |
|--------|------------------------|
| `source_type = "vault"` | 10 (vault items are referenced constantly) |
| `source_type = "procedure"` | 5 (procedures are consulted regularly) |
| `source_type = "graph"` | 3 (entity knowledge is looked up occasionally) |
| `source_type = "daily"` | 1 (daily logs are rarely re-read) |
| `source_type = "session_summary"` | 2 |
| `source_type = "session_segment"` | 1 |

These are deliberately conservative. The system self-corrects over time as real access patterns accumulate.

**Idempotency:** The script checks for existing cognitive fields before writing. Running it twice produces no changes.

---

## Test Plan

### Unit Tests

#### Phase 1: ACT-R Activation

| Test | Input | Expected Output | Edge Case |
|------|-------|----------------|-----------|
| `test_actr_new_memory` | `n=1, age_days=0.01` | `≈ 0.80` | Brand new memory |
| `test_actr_daily_use` | `n=7, age_days=7` | `≈ 0.88` | Regular daily use |
| `test_actr_old_single_access` | `n=1, age_days=365` | `≈ 0.28` | Year-old, one access |
| `test_actr_heavily_used` | `n=100, age_days=365` | `≈ 0.99` | Heavily used, old |
| `test_actr_n_zero_clamped` | `n=0, age_days=10` | Same as `n=1` | Zero access clamped |
| `test_actr_age_zero_clamped` | `n=1, age_days=0` | Near-max activation | Zero age clamped to 0.01 |
| `test_actr_negative_age` | `n=1, age_days=-5` | Same as `age_days=0` | Negative age clamped |
| `test_spread_bonus_even_spacing` | timestamps at [0, 7, 14, 21] days | bonus ≈ 1.2 | Perfect weekly spacing |
| `test_spread_bonus_burst` | timestamps all within 1 hour | bonus ≈ 1.0 | No spacing credit |
| `test_spread_bonus_insufficient_data` | 0 or 1 timestamps | bonus = 1.0 | Not enough data |
| `test_activation_weight_always_positive` | Various extreme inputs | `> 0` always | Sigmoid ensures this |

#### Phase 2: Bayesian Confidence

| Test | Input | Expected Output | Edge Case |
|------|-------|----------------|-----------|
| `test_bayesian_strong_confirm` | `prior=1.0, evidence="strong_confirm"` | `1.15` | Basic confirmation |
| `test_bayesian_strong_contradict` | `prior=1.0, evidence="strong_contradict"` | `0.60` | Basic contradiction |
| `test_bayesian_floor` | `prior=0.1, evidence="strong_contradict"` | `0.1` | Floor prevents erasure |
| `test_bayesian_ceiling` | `prior=1.8, evidence="strong_confirm"` | `2.0` | Ceiling prevents runaway |
| `test_bayesian_invalid_evidence` | `evidence="invalid"` | `ValueError` | Bad input rejected |
| `test_detect_contradictions_high_sim` | Two opposing statements, cosine sim > 0.85 | Detected as contradiction | Core functionality |
| `test_detect_contradictions_low_sim` | Unrelated statements, cosine sim < 0.85 | No contradiction detected | Below threshold |
| `test_classify_agree` | "Meeting at 3pm" / "Meeting is at 3pm" | "strong_confirm" | Agreement |
| `test_classify_disagree` | "Meeting at 3pm" / "Meeting moved to 4pm" | "strong_contradict" | Disagreement |
| `test_classify_unrelated` | "Meeting at 3pm" / "Bought groceries" | "neutral" | Unrelated topics |
| `test_llm_failure_fallback` | LLM call times out | "neutral" (no change) | Fail-safe behavior |

#### Phase 3: Hebbian Co-activation

| Test | Input | Expected Output | Edge Case |
|------|-------|----------------|-----------|
| `test_record_co_retrieval_new_edge` | Two new memory IDs | Edge created, weight=0.1 | First co-retrieval |
| `test_record_co_retrieval_strengthen` | Same pair, second time | Weight increases by α | Strengthening |
| `test_weight_cap` | Edge at max weight, co-retrieved again | Weight stays at 5.0 | Upper bound |
| `test_decay_reduces_weight` | Edge at 1.0, 1 day decay | Weight ≈ 0.95 | Basic decay |
| `test_decay_prunes_weak` | Edge at 0.04, decay applied | Edge deleted | Below threshold |
| `test_canonical_ordering` | IDs "b" and "a" | Stored as ("a", "b") | Dedup guarantee |
| `test_hebbian_boost_mutual` | Two results with strong edge | Both scores increase | Mutual boosting |
| `test_hebbian_boost_no_edge` | Two results, no edge | Scores unchanged | No false boost |
| `test_single_result_no_coactivation` | One result | No edge recorded | Need pairs |

#### Phase 4: Sequential Patterns

| Test | Input | Expected Output | Edge Case |
|------|-------|----------------|-----------|
| `test_log_retrieval_sequence` | Retrieve A then B (3× over a week) | Sequence (A→B) recorded with count=3 | Basic logging |
| `test_predict_next_basic` | After retrieving A (known A→B pattern) | Returns B with confidence > 0 | Core prediction |
| `test_predict_next_min_count` | Pattern seen once, `min_pattern_count=3` | No prediction returned | Below threshold |
| `test_predict_next_no_patterns` | Brand new memory, no history | Empty predictions list | Cold start |
| `test_predict_next_lookback_window` | Pattern old (>90 days), `lookback_days=30` | Not returned | Outside window |

#### Phase 5: Push Triggers

| Test | Input | Expected Output | Edge Case |
|------|-------|----------------|-----------|
| `test_should_push_high_activation` | activation=0.95, confidence=0.9, 48h since last push | `True` | Above threshold |
| `test_should_push_low_activation` | activation=0.3, confidence=0.9 | `False` | Below threshold |
| `test_should_push_rate_limited` | activation=0.95, 2h since last push | `False` | Rate limit (24h) |
| `test_should_push_unresolved_threads` | activation=0.85 (below threshold), unresolved=True | `True` | Unresolved lowers threshold |
| `test_should_push_low_confidence` | activation=0.95, confidence=0.2 | `False` | Low confidence blocks push |

### Integration Tests

| Test | Scenario | Validation |
|------|----------|------------|
| `test_search_uses_actr_scoring` | Store memory, access it 5×, search again | Score increases compared to first search |
| `test_search_records_access` | Search for a memory | `access_count` incremented, timestamp added |
| `test_store_detects_contradiction` | Store "meeting at 3pm", then "meeting at 4pm" | First memory's confidence decreases |
| `test_store_detects_confirmation` | Store "meeting at 3pm", then "meeting is indeed at 3pm" | First memory's confidence increases |
| `test_backward_compat_no_fields` | Search against pre-backfill Chroma store | Defaults applied, no errors |
| `test_coactivation_graph_persists` | Record co-retrieval, restart, check edges | Edges survive restart |
| `test_full_pipeline_scoring` | Store 3 memories (varying access, confidence), search | Ranking reflects all cognitive signals |
| `test_backfill_script_idempotent` | Run backfill twice | No changes on second run |

### Performance Tests

| Test | Scenario | Threshold |
|------|----------|-----------|
| `test_actr_computation_speed` | 10,000 activation calculations | < 100ms total |
| `test_contradiction_detection_latency` | Store memory, 1000 existing memories | < 500ms (excluding LLM call) |
| `test_coactivation_sqlite_write` | Record co-retrieval for top_k=10 (45 edges) | < 50ms |
| `test_coactivation_sqlite_read` | Get associations for 1 memory, 10,000 edges in DB | < 20ms |
| `test_search_overhead_with_primitives` | Full search with ACT-R + confidence + Hebbian vs baseline | < 2× baseline latency |

---

## Migration Plan

### Principles

1. **Zero downtime** — memory search must work throughout migration
2. **Backward compatible** — old code can read new data, new code can read old data
3. **Incremental** — each phase can be deployed independently
4. **Reversible** — each phase can be rolled back by reverting the code (data fields are additive, never destructive)

### Phase 1 Migration (ACT-R)

**Step 1: Deploy code changes**
- New `actr_activation()` function in `embedder.py`
- Updated `compute_final_score()` with default parameters
- Updated `search()` to extract new metadata fields
- Updated `add_segment()` and `add_session_summary()` to include new fields

**Deployment note:** Because `compute_final_score()` defaults `access_count=1` and `access_timestamps=None`, the new code works identically to the old code for memories without cognitive fields. Deployment is safe without any data migration.

**Step 2: Run backfill (optional)**
```bash
python scripts/backfill_cognitive_fields.py --agent bob --infer-access
python scripts/backfill_cognitive_fields.py --agent patterson --infer-access
python scripts/backfill_cognitive_fields.py --agent dean --infer-access
python scripts/backfill_cognitive_fields.py --agent shared --infer-access
```

**Step 3: Monitor**
- Log activation values for 48 hours
- Compare search result rankings (old vs new) on a sample query set
- Verify no regressions in `pai-memory-recall` output quality

**Rollback:** Revert code changes. New metadata fields in Chroma are harmless — they're simply ignored by the old code.

### Phase 2 Migration (Bayesian Confidence)

**Step 1: Deploy `confidence.py` module**
**Step 2: Update `compute_final_score()` to include confidence**
**Step 3: Update `store_memory` pipeline to run contradiction detection**

**Deployment note:** All memories default to `confidence=1.0`, which is multiplicatively neutral. No scoring change until contradictions are actively detected.

**Rollback:** Revert code. Confidence values in metadata are ignored by old code.

### Phase 3 Migration (Hebbian)

**Step 1: Deploy `coactivation.py` module**
**Step 2: SQLite databases are created automatically on first use** (no manual setup)
**Step 3: Update `search()` to call `hebbian_boost()` and `record_co_retrieval()`

**Deployment note:** The co-activation graph starts empty. Association edges build up organically through normal search usage. No backfill needed.

**Rollback:** Revert code. SQLite files can be deleted or left in place (harmless).

### Phase 4 Migration (Sequential Patterns)

**Step 1: Deploy `src/graph/sequences.py` module**
**Step 2: SQLite `retrieval_sequences` table is created automatically on first use** (same pattern as Phase 3)
**Step 3: Wire `search()` to call `log_retrieval_sequence()` after returning results**

**Deployment note:** Sequence patterns build up over time through normal search usage. Predictions are only surfaced after `min_pattern_count` occurrences (default: 3), so the system self-calibrates without manual intervention.

**Rollback:** Revert code. Sequence data SQLite file can be deleted or left in place (harmless). Remove `predict` parameter handling from MCP server if deployed.

### Phase 5 Migration (Push Triggers)

**Step 1: Add `last_pushed_at` field to Chroma metadata** — old memories without this field default to `null` (never pushed), meaning they are eligible for push scanning immediately. This is intentional: it allows the system to discover high-activation historical memories on first scan.
**Step 2: Deploy `should_push()` function**
**Step 3: Configure cron job** — schedule push scan every 30 minutes during active hours (7am–11pm), hourly during quiet hours. Uses existing OpenClaw cron infrastructure.

**Deployment note:** First scan may surface a burst of high-activation memories that have never been pushed. This is expected and will stabilize after the first 24-hour cycle as `last_pushed_at` gets populated.

**Rollback:** Disable cron job. Remove `should_push()` calls. `last_pushed_at` fields in Chroma are harmless metadata that can remain.

---

## MCP Integration Points

These changes affect the MCP Memory Server as specified in `docs/MCP-MEMORY-SERVER-SPEC.md`.

### `search_memory` Tool Changes

**New optional parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `predict` | bool | `false` | Include predicted next-memories (Phase 4) |
| `include_cognitive_scores` | bool | `false` | Include ACT-R activation, confidence, and association data in results |

**Updated response schema** (when `include_cognitive_scores=true`):

```json
{
  "content": "...",
  "source_file": "...",
  "score": 0.87,
  "date": "2026-03-01",
  "type": "decision",
  "domain": "redhat",
  "cognitive": {
    "activation_weight": 0.88,
    "access_count": 7,
    "confidence": 1.15,
    "association_boost": 0.05,
    "spread_bonus": 1.12
  },
  "predicted_next": [
    {"memory_id": "...", "confidence": 0.8}
  ]
}
```

The `cognitive` and `predicted_next` fields are **only included** when the corresponding flags are set. Default search responses are unchanged from the current spec.

### `store_memory` Tool Changes

**New behavior** (no parameter changes):

When `store_memory` is called:
1. Existing pipeline (embed, classify, write markdown, index Chroma) runs as before
2. **New:** After embedding, `detect_contradictions()` runs against existing memories
3. **New:** Any contradicted memories have their `confidence` updated via `bayesian_update()`
4. **New:** The response includes a `contradictions` field if any were detected:

```json
{
  "id": "...",
  "type": "decision",
  "contradictions_detected": [
    {
      "existing_memory_id": "...",
      "similarity": 0.91,
      "evidence_type": "strong_contradict",
      "old_confidence": 1.0,
      "new_confidence": 0.60
    }
  ]
}
```

### `memory_stats` Tool Changes

**New fields in response:**

```json
{
  "...existing fields...",
  "cognitive_stats": {
    "avg_activation": 0.62,
    "high_activation_count": 34,
    "low_confidence_count": 8,
    "total_coactivation_edges": 1247,
    "strongest_associations": [
      {"pair": ["memory_a", "memory_b"], "weight": 4.2}
    ],
    "most_contradicted": [
      {"memory_id": "...", "confidence": 0.3, "contradiction_count": 4}
    ]
  }
}
```

### New MCP Tool: `get_cognitive_profile`

Exposes per-memory cognitive data for debugging and transparency.

**Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_id` | string | Yes | Chroma document ID |

**Returns:**

```json
{
  "memory_id": "...",
  "content_preview": "First 100 chars...",
  "activation": {
    "access_count": 12,
    "age_days": 45,
    "base_level_B": 1.82,
    "activation_weight": 0.86,
    "spread_bonus": 1.15,
    "last_accessed": "2026-03-03T14:22:00Z"
  },
  "confidence": {
    "current": 0.85,
    "history": [
      {"date": "2026-03-01", "change": -0.15, "evidence": "weak_contradict", "trigger_memory": "..."}
    ]
  },
  "associations": [
    {"memory_id": "...", "weight": 2.1, "co_retrieval_count": 8}
  ]
}
```

---

## Patent Cleanliness Statement

All cognitive primitives in this specification are based on **published academic research** that is well-established in the public domain. No proprietary algorithms or patented techniques are used.

### Citations and Prior Art

| Primitive | Foundation | Publication | Status |
|-----------|-----------|-------------|--------|
| ACT-R Base-Level Activation | Anderson, J.R. (1993). *Rules of the Mind.* Lawrence Erlbaum Associates. | Academic textbook, widely cited (10,000+ citations per Google Scholar) | **Public domain theory.** ACT-R is an open-source cognitive architecture maintained by Carnegie Mellon University. The base-level learning equation is published, peer-reviewed, and freely implementable. |
| Bayesian Confidence | Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances." *Phil. Trans. Royal Society,* 53, 370–418. Laplace, P.S. (1812). *Théorie analytique des probabilités.* | 18th/19th century mathematics | **Public domain.** Bayes' theorem is foundational probability theory, predating modern patent law by over a century. |
| Hebbian Learning | Hebb, D.O. (1949). *The Organization of Behavior.* Wiley. | Academic monograph, foundational neuroscience | **Public domain.** Hebb's rule is a basic neuroscience principle taught in every introductory cognitive science course. No implementation patent exists or could exist for the general principle. |
| Sequential Pattern Detection | General time-series analysis | Multiple standard CS texts | **Public domain.** Sequence pattern mining is a standard data mining technique (Agrawal & Srikant, 1995; Pei et al., 2001). |
| Sigmoid Function | Verhulst, P.-F. (1838, 1845) | 19th century mathematics | **Public domain.** The logistic function has been in continuous use for nearly 200 years. |

### Implementation Originality

The specific **combination and adaptation** of these primitives for a distributed AI memory system is novel to this project. However, each individual component is a straightforward application of published research. The adaptation involves:

- Mapping ACT-R's activation equation to Chroma metadata fields (engineering, not invention)
- Using Bayesian updates with LLM-classified evidence types (composition of existing techniques)
- Implementing Hebbian learning in SQLite rather than neural weights (data structure choice)
- Integrating these signals into an existing scoring pipeline (software architecture)

None of these adaptations rise to the level of patentable invention. They are competent software engineering applying well-understood cognitive science.

### License Compatibility

- **ACT-R software** is released under LGPL by CMU. We are not using the ACT-R software; we are implementing its published equations independently. No license obligation.
- **Chroma** is Apache 2.0 licensed. Our extensions are metadata additions, not modifications to Chroma.
- **SQLite** is public domain. No license obligation.

---

## Appendix: Configuration Parameters

All tunable parameters are collected here for reference. Default values are conservative and should work well without tuning. Future phases may expose these via an MCP `config` tool.

| Parameter | Default | Range | Phase | Location |
|-----------|---------|-------|-------|----------|
| `actr_decay` | 0.5 | 0.1–1.0 | 1 | `actr_activation()` |
| `spread_bonus_weight` | 0.2 | 0.0–0.5 | 1 | `actr_activation()` |
| `access_timestamps_cap` | 20 | 5–100 | 1 | `record_access()` |
| `confidence_min` | 0.1 | 0.01–0.5 | 2 | `confidence.py` |
| `confidence_max` | 2.0 | 1.5–5.0 | 2 | `confidence.py` |
| `similarity_threshold` | 0.85 | 0.7–0.95 | 2 | `detect_contradictions()` |
| `hebbian_learning_rate` | 0.1 | 0.01–0.5 | 3 | `coactivation.py` |
| `hebbian_decay_rate` | 0.05 | 0.01–0.2 | 3 | `coactivation.py` |
| `hebbian_max_weight` | 5.0 | 1.0–10.0 | 3 | `coactivation.py` |
| `hebbian_boost_factor` | 0.15 | 0.05–0.3 | 3 | `hebbian_boost()` |
| `push_activation_threshold` | 0.9 | 0.7–0.95 | 5 | `should_push()` |
| `push_rate_limit_hours` | 24 | 1–168 | 5 | `should_push()` |

---

*End of specification. This document is a living artifact — update as implementation reveals new edge cases or design decisions.*

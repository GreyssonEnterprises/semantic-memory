# Spec Verification Report — Task 4

**Agent:** code-reviewer
**Date:** 2026-03-04
**Task:** Verify completeness and cross-reference accuracy of COGNITIVE-PRIMITIVES-SPEC.md and MCP-MEMORY-SERVER-SPEC.md

---

## Source File Cross-Reference Results (via jcodemunch)

All verified against `local/semantic-memory` index (indexed 2026-03-04T05:51):

| Spec Reference | Claimed Line | Actual Line | Status |
|----------------|-------------|-------------|--------|
| `src/retrieval/recall.py` — `compute_final_score()` | `:44` | `:44` | ✅ PASS |
| `src/retrieval/recall.py` — `search()` | `:122` | `:122` | ✅ PASS |
| `src/pipeline/embedder.py` — `time_weight()` | `:47` | `:47` | ✅ PASS |
| `src/pipeline/store.py` — `add_segment()` | `:52` | `:52` | ✅ PASS |
| `src/pipeline/store.py` — `add_session_summary()` | `:90` | `:90` | ✅ PASS |
| `src/pipeline/store.py` — `SOURCE_WEIGHTS` | `:18` | `:18` | ✅ PASS |
| `src/retrieval/recall.py` — `parse_date()` | `:30` | `:36` | ❌ FAIL |

**Issue found:** COGNITIVE-PRIMITIVES-SPEC.md line 209 states:
> `access_timestamps` are parsed with the same `parse_date()` from `recall.py:30`

Actual location per jcodemunch: `parse_date` is at **line 36**, not 30.

---

## COGNITIVE-PRIMITIVES-SPEC.md — Detailed Checks

### 1. Function Signatures & Line Numbers — FAILED (one error)
See table above. All references correct except `parse_date()` cited as `:30`, actual `:36`.

### 2. Mathematical Formulas — PASSED (one minor gap)

All formulas have variable definitions:
- ACT-R: `B = ln(n + 1) − 0.5 × ln(age_days / (n + 1))` — n, age_days, ln, decay parameter all defined ✅
- Sigmoid: `1 / (1 + e^(−B))` — B defined ✅
- Spread bonus: `1 + 0.2 × (1 − min(1, temporal_spread / mean_interval))` — `temporal_spread` defined; **`mean_interval` is used but not explicitly defined** (it's the mean of the intervals array, which is implied but should be stated)
- Bayesian: `confidence_new = confidence_old × likelihood_ratio` — all defined ✅
- Hebbian: `w_ij += α` and `w_ij *= (1 − λ)` — all defined ✅
- Hebbian boost: `sum(w_ij * boost_factor)` — all defined ✅

Minor gap: `mean_interval` in spread_bonus formula should have an explicit definition sentence.

### 3. Citations — PASSED

| Citation Required | Present |
|-------------------|---------|
| Anderson (1993) — ACT-R | ✅ Phase 1, full bibliographic detail |
| Hebb (1949) — Hebbian | ✅ Phase 3, full bibliographic detail |
| Bayes (1763) | ✅ Phase 2 + Patent Cleanliness section |
| Laplace (1812) | ✅ Phase 2 + Patent Cleanliness section |

All four required citations present with proper journal/book references.

### 4. Test Plan — FAILED (Phases 4 & 5 missing)

| Phase | Unit Tests | Status |
|-------|-----------|--------|
| Phase 1 (ACT-R) | 12 tests covering all edge cases | ✅ |
| Phase 2 (Bayesian) | 11 tests including failure modes | ✅ |
| Phase 3 (Hebbian) | 9 tests including SQLite persistence | ✅ |
| Phase 4 (Sequential Patterns) | **0 tests** | ❌ |
| Phase 5 (Push Triggers) | **0 tests** | ❌ |

Missing tests needed:
- Phase 4: `test_log_retrieval_sequence`, `test_predict_next_memories_basic`, `test_predict_next_min_pattern_count`, `test_predict_next_no_patterns`
- Phase 5: `test_should_push_rate_limit`, `test_should_push_high_activation`, `test_should_push_unresolved_threads`, `test_should_push_low_activation`

### 5. Migration Plan — FAILED (Phases 4 & 5 absent)

| Phase | Migration Steps | Rollback | Status |
|-------|----------------|---------|--------|
| Phase 1 | 3 steps + rollback | ✅ | ✅ |
| Phase 2 | 3 steps + rollback | ✅ | ✅ |
| Phase 3 | 3 steps + rollback | ✅ | ✅ |
| Phase 4 | **None** | **None** | ❌ |
| Phase 5 | **None** | **None** | ❌ |

Phase 4 migration would need to cover: SQLite `retrieval_sequences` table creation (auto on first use, like Phase 3), monitoring plan. Phase 5 migration would cover: cron job setup, `last_pushed_at` field deployment.

### 6. Backward Compatibility — FAILED (two fields not covered)

| Field | Compat Statement | Status |
|-------|-----------------|--------|
| `access_count` | Defaults to `1` — explicit statement in Phase 1 | ✅ |
| `last_n_access_timestamps` | Defaults to `[]` — explicit statement in Phase 1 | ✅ |
| `confidence` | Defaults to `1.0` — explicit statement in Phase 2 | ✅ |
| `coactivation_ids` | Default `"[]"` in table, but no compat statement in Phase 3 section | ⚠️ |
| `last_pushed_at` | No backward compat statement anywhere | ❌ |

`last_pushed_at` (Phase 5) is defined in the Phase 5 section with `null` default, but no explicit statement about how old memories (without this field) behave when the push scan reads them.

### 7. Edge Cases — PASSED (Phases 1-3 only, Phases 4-5 have no formulas)

Phases 1, 2, 3 all have well-documented edge case tables. Phases 4 and 5 don't have complex formulas requiring edge cases — the edge cases that exist (e.g., `min_pattern_count`, rate limiting) are addressed in the function docs themselves.

### 8. Data Model Section — FAILED (one field missing)

Consolidated Data Model table (line 608+) includes:
- `access_count` ✅
- `last_n_access_timestamps` ✅
- `confidence` ✅
- `coactivation_ids` ✅ (but see Issue B below)

**Missing:** `last_pushed_at` (introduced in Phase 5) is NOT in the consolidated table.

---

## Additional Inconsistency Found

### Issue A: `coactivation_ids` Chroma field vs. SQLite storage

The consolidated Data Model table lists `coactivation_ids | list[str] (JSON) | string | "[]"` as a **Chroma metadata field** (Phase 3). However, Phase 3 explicitly stores co-activation data in **SQLite** (`coactivation.db`), not Chroma. The SQLite schema has no `coactivation_ids` column — edges are identified by `memory_id_a` and `memory_id_b`.

It's unclear why `coactivation_ids` would exist as a Chroma metadata field if the graph is in SQLite. This is either:
- Redundant (pre-computed edges cached in Chroma for fast lookup), or
- An error (field was added to the table but Phase 3 doesn't actually write it to Chroma)

The Phase 3 implementation section has no code showing `coactivation_ids` being written to Chroma. **A developer would be confused about whether to implement this field or not.**

---

## MCP-MEMORY-SERVER-SPEC.md — Checks

### Cross-references to COGNITIVE-PRIMITIVES-SPEC.md — PASSED

All cross-references verified:
- §MCP Integration Points ✅
- §Phase 1, §Phase 2, §Phase 3 ✅
- §Data Model ✅
- §New MCP Tool ✅

All section references resolve to real sections in COGNITIVE-PRIMITIVES-SPEC.md.

### search_memory & store_memory reflect cognitive changes — PASSED

`search_memory`:
- `include_cognitive_scores` param ✅
- `predict` param ✅
- Updated response schema with `cognitive` object ✅
- `predicted_next` field ✅
- Scoring formula evolution table ✅

`store_memory`:
- `confidence` param with range and default ✅
- `vault` param ✅
- Updated pipeline showing contradiction detection (steps 3 & 4) ✅
- `contradictions_detected` response field ✅

### No content accidentally removed — PASSED

MCP spec is 538 lines. All original sections present: Executive Summary, Architecture, MCP Tools (6 tools), Agent Identity & Namespace, Capture Templates, Technical Implementation, Deployment, Implementation Phases, Task Assignments, How This Changes the Chroma Project, Open Questions. New Cognitive Memory Primitives section added (lines 470-492). File is longer than before.

### Architecture diagram includes SQLite — PASSED

SQLite block added to ASCII diagram with label `(Hebbian edges — Phase 3)` ✅

### `get_cognitive_profile` tool — WARNING

The tool is fully specified in COGNITIVE-PRIMITIVES-SPEC.md §New MCP Tool (line 1250+), but in MCP-MEMORY-SERVER-SPEC.md it's only referenced in a single bullet (line 492):
> "New MCP tool: `get_cognitive_profile` exposes per-memory cognitive data..."

It is NOT added to the MCP Tools Specification section alongside `search_memory`, `store_memory`, etc. A developer reading only the MCP spec would not find the tool's parameter list there — they'd have to follow the cross-reference to COGNITIVE-PRIMITIVES-SPEC.md. This is workable but creates a discoverability gap.

---

## Cross-Reference Completeness Check

Every MCP integration point in COGNITIVE-PRIMITIVES-SPEC.md is reflected in MCP-MEMORY-SERVER-SPEC.md:

| Integration Point | In Cogn. Spec | In MCP Spec | Status |
|-------------------|--------------|-------------|--------|
| `search_memory` — `include_cognitive_scores` | ✅ | ✅ | ✅ |
| `search_memory` — `predict` | ✅ | ✅ | ✅ |
| `store_memory` — `contradictions_detected` | ✅ | ✅ | ✅ |
| `memory_stats` — `cognitive_stats` | ✅ | ✅ | ✅ |
| `get_cognitive_profile` (new tool) | ✅ (full spec) | ⚠️ (referenced only) | ⚠️ |
| SQLite in architecture | ✅ | ✅ | ✅ |

---

## Developer Implementability Assessment

**Phase 1 (ACT-R):** ✅ Implementable without questions
**Phase 2 (Bayesian):** ✅ Implementable without questions
**Phase 3 (Hebbian):** ⚠️ Implementable, but `coactivation_ids` Chroma field ambiguity would cause a clarifying question
**Phase 4 (Sequential):** ⚠️ Mostly implementable — function signatures are clear, but no test cases and no migration plan means a developer won't know validation criteria or deployment procedure
**Phase 5 (Push Triggers):** ⚠️ Implementable, but missing test cases and migration plan reduces confidence

---

## Summary: Pass/Fail by Check

| Check | Result |
|-------|--------|
| All file:line references accurate | ❌ FAILED — `parse_date` cited as `:30`, actual `:36` |
| All formula variables defined | ⚠️ MOSTLY PASSED — `mean_interval` undefined in spread_bonus formula |
| All required citations present | ✅ PASSED |
| Test plan covers all 5 phases | ❌ FAILED — Phases 4 & 5 have no unit tests |
| Migration plan complete | ❌ FAILED — Phases 4 & 5 missing migration + rollback |
| Backward compat for all new fields | ❌ FAILED — `last_pushed_at` has no compat statement |
| Edge cases documented | ✅ PASSED (for phases with formulas) |
| Data model has all new fields | ❌ FAILED — `last_pushed_at` missing from consolidated table |
| MCP cross-references accurate | ✅ PASSED |
| search_memory/store_memory updated | ✅ PASSED |
| No content removed from MCP spec | ✅ PASSED |
| Architecture diagram has SQLite | ✅ PASSED |
| MCP integration bidirectionally complete | ⚠️ MOSTLY PASSED — `get_cognitive_profile` not in MCP tools list |
| `coactivation_ids` consistency | ❌ FAILED — listed as Chroma field but Phase 3 uses SQLite; ambiguous |

**Overall: FAILED — 5 issues require fixes before this spec is implementation-ready.**

---

## Required Fixes (Priority Order)

### Fix 1 — CRITICAL: Correct `parse_date()` line number
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, line 209
**Change:** `recall.py:30` → `recall.py:36`

### Fix 2 — HIGH: Add Phase 4 & 5 test cases
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, Test Plan section (after Phase 3 tests)
Add unit tests for `log_retrieval_sequence`, `predict_next_memories`, and `should_push`.

### Fix 3 — HIGH: Add Phase 4 & 5 migration plans
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, Migration Plan section
Phase 4: SQLite `retrieval_sequences` table auto-created; monitor and rollback instructions.
Phase 5: Cron job deployment; `last_pushed_at` field compat note; rollback = disable cron.

### Fix 4 — HIGH: Add `last_pushed_at` to consolidated Data Model table
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, Data Model Changes section, "Metadata Field Additions" table
Add row: `last_pushed_at | string (ISO) | null | Phase 5 | should_push()`
And add backward compat statement: old memories without this field default to "never pushed."

### Fix 5 — MEDIUM: Resolve `coactivation_ids` ambiguity
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, Data Model Changes section
Either:
- Remove `coactivation_ids` from the Chroma metadata table (if it's SQLite-only), or
- Add a sentence in Phase 3 explaining why this field is also stored in Chroma (caching for fast lookup?)

### Fix 6 — MINOR: Define `mean_interval` in spread_bonus formula
**Location:** COGNITIVE-PRIMITIVES-SPEC.md, Phase 1, Spaced Repetition Bonus section
Add: "`mean_interval` = mean(intervals)" after the `temporal_spread` definition.

### Fix 7 — MINOR: Add `get_cognitive_profile` to MCP Tools Specification section
**Location:** MCP-MEMORY-SERVER-SPEC.md, after tool 6 (`get_procedures`)
Add a brief tool spec (params and returns) or explicitly note it's specified in COGNITIVE-PRIMITIVES-SPEC.md §New MCP Tool with a section header.

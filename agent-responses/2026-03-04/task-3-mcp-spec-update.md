# Task 3: MCP-MEMORY-SERVER-SPEC.md Update — Audit Report

**Agent:** mcp-updater (technical-writer)
**Date:** 2026-03-04
**Status:** COMPLETE

## Changes Made

Five targeted edits to `docs/MCP-MEMORY-SERVER-SPEC.md`:

### 1. Architecture Diagram (lines 54-63)
- Added SQLite co-activation graph box alongside Chroma and ollama
- Shows per-agent database path and Phase 3 attribution

### 2. `search_memory` Section (lines 91-126)
- Added `include_cognitive_scores` and `predict` parameters to parameter table
- Updated Returns to include `cognitive` object schema (activation_weight, access_count, confidence, association_boost, spread_bonus) and `predicted_next` field
- Replaced simple time decay description (`cosine_sim * 1/(1+log(days+1))`) with ACT-R activation-based scoring, Bayesian confidence multiplier, and Hebbian co-activation boost
- Added scoring formula evolution showing progression across phases
- Added cross-references to COGNITIVE-PRIMITIVES-SPEC.md

### 3. `store_memory` Section (lines 128-173)
- Added `confidence` parameter (float, default 1.0, range [0.1, 2.0])
- Replaced 4-step parallel pipeline with 6-step pipeline including contradiction detection and Bayesian updates, with dependency annotations
- Updated Returns to include `contradictions_detected` field with JSON schema
- Added cross-reference to COGNITIVE-PRIMITIVES-SPEC.md Phase 2

### 4. Technical Implementation / Stack (lines 369-377)
- Added SQLite co-activation graph entry with path convention and Phase 3 reference
- Added Cognitive Primitives entry summarizing all three integrated primitives

### 5. New Section: Cognitive Memory Primitives (lines 470-493)
- Added after Implementation Phases, before Task Assignments
- Primitives Overview table mapping each phase to affected MCP tools
- Integration Notes covering: backward compatibility, progressive deployment, SQLite dependency, new Chroma metadata fields, new `get_cognitive_profile` MCP tool

## Verification Checklist
- [x] All existing content preserved (no deletions)
- [x] Formatting consistent with existing doc style
- [x] Cross-references to COGNITIVE-PRIMITIVES-SPEC.md use section anchors
- [x] Architecture diagram renders correctly in ASCII
- [x] Parameter tables follow existing column structure
- [x] JSON examples follow existing indentation conventions
- [x] No duplicate information — new section summarizes/references rather than repeating

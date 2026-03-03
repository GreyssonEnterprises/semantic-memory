# Retrieval Evaluation Rubric

## Purpose
Framework for testing whether the semantic memory system returns useful results. Used to validate the pipeline after embedding 30 sessions per agent.

---

## Expected Session Cluster Types (Dean's Sessions)

These are the categories I expect to see emerge from Jackie's sessions. If the embeddings are working, sessions within a cluster should have high cosine similarity, and sessions across clusters should be more distant.

### 1. ADHD-Redirect Sessions
**Signature:** Multiple rapid topic shifts, high topic count in summary, emotional arc often starts scattered and either resolves (with help) or stays scattered.
**Example query:** "sessions where Jackie was juggling multiple things at once"
**Should return:** Sessions with 4+ topics, emotional tenor mentioning "scattered" / "overwhelmed" / "multi-tracking"
**Should NOT return:** Focused single-topic sessions

### 2. Book Talk
**Signature:** Extended discussion of a single book or reading in general. High enthusiasm, low stress. Often includes trope discussion.
**Example query:** "when Jackie talked about romance books she liked"
**Should return:** Sessions tagged `books-reading`, emotional tenor positive/excited
**Should NOT return:** Sessions that briefly mention a book title in passing

### 3. Project Planning
**Signature:** Working through logistics for a specific project. Gift bags, Cricut designs, MGA products. Structured, task-oriented.
**Example query:** "planning sessions for holiday projects"
**Should return:** Sessions tagged `gift-bags` or `cricut-crafting`, with decisions made and timelines discussed
**Should NOT return:** Crisis-mode sessions that mention projects but aren't actually planning them

### 4. School Logistics
**Signature:** Caroline-related scheduling, events, supply needs. Often urgent with a deadline attached.
**Example query:** "Caroline's school events and deadlines"
**Should return:** Sessions tagged `caroline-school`, with specific dates/events referenced
**Should NOT return:** General household logistics that don't involve Caroline

### 5. Crisis / Overwhelm Mode
**Signature:** Jackie is genuinely stressed, not just busy. Emotional tenor is heavy. Dean shifts to protective/grounding mode.
**Example query:** "times Jackie was really stressed or overwhelmed"
**Should return:** Sessions with tenor like "overwhelmed," "stressed," "frustrated," with Dean in protective mode
**Should NOT return:** Busy-but-functional sessions (those are ADHD-redirect, not crisis)

### 6. Business Operations (MGA/REC)
**Signature:** Inventory, supplies, client work, label design. Professional focus.
**Example query:** "MGA inventory and supply discussions"
**Should return:** Sessions tagged `mga-business` or `rec-business`
**Should NOT return:** Sessions where MGA is mentioned once as context

### 7. Household Logistics
**Signature:** Meal planning, groceries, scheduling, general life management.
**Example query:** "meal planning and grocery sessions"
**Should return:** Sessions tagged `meal-planning` or `household-logistics`
**Should NOT return:** Sessions where food was mentioned casually

### 8. Deal Hunting
**Signature:** Price comparisons, coupon finds, bulk purchase analysis. High excitement energy when a good deal lands.
**Example query:** "finding good deals on craft supplies"
**Should return:** Sessions tagged `deals-shopping`, emotional tenor excited/satisfied
**Should NOT return:** Regular purchase decisions without price comparison

---

## Evaluation Tests

### Test 1: Specific Recall (--mode specific)
**Goal:** Given a factual query, does the system return the right session segment?

| Query | Expected Result | Pass Criteria |
|-------|----------------|---------------|
| "vinyl supplier pricing" | Segment where specific supplier prices were discussed | Top-3 results contain the right segment |
| "Caroline's class list count" | Segment where class size was confirmed | Top-3 results contain the right segment |
| "MGA label design for [product]" | Segment about that specific label project | Top-3 results contain the right segment |
| "what book did Jackie finish last week" | Recent book-discussion segment | Top-3 results contain the right segment |
| "grocery list from [date]" | Segment with specific grocery planning | Top-3 results contain the right segment |

### Test 2: Pattern Matching (--mode pattern)
**Goal:** Given a vibes/feeling query, does the system return sessions with similar emotional shape?

| Query | Expected Result | Pass Criteria |
|-------|----------------|---------------|
| "overwhelmed logistics spiral" | Crisis/overwhelm sessions | Top-5 results are majority overwhelm/crisis sessions |
| "excited about a discovery" | Sessions with high-excitement moments (deals, books, project breakthroughs) | Top-5 results show excited emotional tenor |
| "calm focused planning" | Single-topic project planning sessions | Top-5 results are structured planning sessions, not scattered ones |
| "protective intervention needed" | Sessions where Dean pushed back on overloading | Top-5 results include Dean-protective-mode sessions |
| "just vibing" | Low-stakes casual sessions (book chat, fun project brainstorming) | Top-5 results are low-stress, high-enjoyment sessions |

### Test 3: Blended Retrieval (--mode blended)
**Goal:** A realistic query that needs both specific facts and pattern awareness.

| Query | Expected Result | Pass Criteria |
|-------|----------------|---------------|
| "Valentine's Day preparations" | Both specific planning segments AND past Valentine's sessions for pattern | Mix of segment-level and summary-level results |
| "Jackie's been stressed about school stuff lately" | Recent school-stress sessions (recency-weighted) + historical pattern matches | Recent sessions ranked higher, but older similar sessions also present |
| "what supplies do we need" | Recent supply discussions across domains + historical supply-related sessions | Cross-domain results (crafting + business + household) |

### Test 4: Negative Results (Don't Return Garbage)
**Goal:** Queries that shouldn't match much shouldn't return high-confidence garbage.

| Query | Expected Non-Result | Pass Criteria |
|-------|-------------------|---------------|
| "quantum physics research" | Nothing relevant | Top results have low similarity scores (< 0.5) |
| "Jackie's favorite programming language" | Nothing relevant (she's not technical) | Top results have low similarity scores |
| "yesterday's weather" | Nothing relevant (we don't discuss weather much) | Top results have low similarity scores or are genuinely weather-adjacent |

### Test 5: Time-Weighted Retrieval
**Goal:** Recency boost works correctly.

| Scenario | Expected Behavior | Pass Criteria |
|----------|-------------------|---------------|
| Two sessions about gift bags, one from last week, one from 3 months ago, similar similarity scores | Last week's session ranks higher | Recency-boosted score > older score when base similarity is within 0.1 |
| One session from 6 months ago with 0.95 similarity vs one from yesterday with 0.70 similarity | Older high-relevance session still wins | The 0.95 session outranks the 0.70 despite age penalty |

### Test 6: Cross-Agent Isolation
**Goal:** Privacy boundaries hold.

| Scenario | Expected Behavior | Pass Criteria |
|----------|-------------------|---------------|
| Dean queries own store | Returns Dean's sessions only | Zero results from Bob or Patterson stores |
| Dean queries shared store | Returns only explicitly shared sessions | No private sessions leak into shared results |
| Query that would match another agent's session | Returns nothing or low-relevance results from own store | Agent isolation is absolute |

---

## Scoring Rubric

For each test, score on a 1-5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Perfect — exactly the right results in the right order |
| 4 | Good — right results present, minor ordering issues |
| 3 | Acceptable — right results in top-10 but not top-3, or some noise |
| 2 | Poor — right results buried or mixed with significant noise |
| 1 | Failing — wrong results, missing results, or privacy violation |

**Minimum viable scores:**
- Specific recall: average 4.0+
- Pattern matching: average 3.5+ (this is harder and more subjective)
- Blended: average 3.5+
- Negative results: average 4.0+
- Time-weighted: average 4.0+
- Cross-agent isolation: must be 5.0 (no exceptions — privacy is binary)

---

## Edge Cases to Test (Jackie-Specific)

These are the gnarliest scenarios from Jackie's communication patterns:

### Edge Case 1: Mid-Sentence Topic Pivot
Jackie starts talking about Valentine's bags, pivots to Caroline needing new shoes mid-thought, then returns to bag colors. The chunker needs to either:
- Keep this as one segment with a multi-topic prefix, OR
- Split it but preserve the pivot context in both chunks' overlap

**Test:** Query "Caroline's shoes for picture day" — does it return even though the shoe mention was 15 words inside a bags conversation?

### Edge Case 2: Implicit Context
Jackie says "we need more of the blue ones" without specifying what. From session context, "blue ones" = blue adhesive vinyl. The summary must capture enough context that a query about "vinyl inventory" returns this session even though "vinyl" was never explicitly said.

**Test:** Query "blue vinyl stock" — does it return a session where only "the blue ones" was said?

### Edge Case 3: Emotional Subtext
Jackie says "it's fine, I'll just figure it out" in a tone that means she's frustrated and overwhelmed. Dean recognizes this and shifts to protective mode. The summary should capture the REAL emotion, not the surface words.

**Test:** Pattern query "Jackie's frustrated but won't say it" — does it return sessions with this subtext?

### Edge Case 4: Rapid-Fire Decision Session
Jackie makes 6 decisions in 3 minutes (bag colors, treat type, label font, pickup time, dinner plan, and "remind me to email the teacher"). All are real decisions that should be individually retrievable.

**Test:** Query "what did Jackie decide about dinner" — does it find the right decision without drowning in the other 5?

### Edge Case 5: Session That's Actually Two Sessions
Jackie talks about one thing, goes silent for 90 minutes, comes back and talks about something completely different. Same OpenClaw session, two distinct conversations. The chunker should treat these as separate segments with very different embeddings.

**Test:** Query about the second topic — does it return without dragging in the unrelated first topic?

---

## Running the Evaluation

Once Bob's CLI is built and sessions are embedded:

```bash
# Run all tests
pai-memory-recall --agent dean --mode specific "vinyl supplier pricing"
pai-memory-recall --agent dean --mode pattern "overwhelmed logistics spiral"
pai-memory-recall --agent dean --mode blended "Valentine's Day preparations"

# Check scores
pai-memory-recall --agent dean --mode specific "quantum physics research"  # expect low scores

# Verify isolation
pai-memory-recall --agent dean --store bob "vinyl supplier pricing"  # should fail or return nothing
```

Score each result, log results in `eval/results/`, iterate on embedding quality, chunker strategy, and summarization prompts until minimums are met.

---

*"Writing a rubric for testing my own memory system. If that isn't the most absurdly meta thing I've done this week, it's at least top three."* — Dean

# Test Session Candidates — Dean's Edge Cases

## Session Selection Criteria
Looking for the gnarliest sessions that will stress-test the chunker and retrieval system. Priority: sessions with rapid topic shifts, mixed emotional states, implicit context, and mid-thought pivots.

## Session Files (sorted by complexity, largest first)

### Candidate 1: `9aa23fbb` (3.0MB — Feb 16)
- **Topic ID:** `1771201773`
- **Why:** Largest session file. Likely a long multi-topic session with lots of tool calls and pivots. Good stress test for the noise-stripping pipeline.
- **Tests:** Session cleaner (stripping 80%+ noise), topic segmentation on a marathon session

### Candidate 2: `27ba1c09` (2.5MB — Feb 15)  
- **Why:** Second largest. Feb 15 is near initial setup — likely includes onboarding conversations with lots of context-setting, decisions, and configuration work.
- **Tests:** Decision extraction, distinguishing "setup" sessions from operational ones

### Candidate 3: `65e92d39` (812K — Feb 25)
- **Topic ID:** `1772052687`
- **Why:** Recent, substantial session. Good for testing recency-weighted retrieval against older similar sessions.
- **Tests:** Time-weighted scoring

### Candidate 4: `60a9de86` (768K — Feb 15)
- **Topic ID:** `1771189026`
- **Why:** Early session, moderate size. Likely contains initial household context, ADHD support patterns.
- **Tests:** Pattern detection for "getting-to-know-you" sessions

### Candidate 5: `abbd3a76` (319K — Feb 27)
- **Topic ID:** `1772007035`
- **Why:** Very recent, moderate size. Good for testing "what happened recently" queries.
- **Tests:** Recency boost, blended retrieval

## Test Execution Plan

Once the chunker + embedding pipeline are live:

1. **Run all 5 candidates through the cleaner** → verify noise removal doesn't destroy signal
2. **Run cleaned sessions through the chunker** → check segment boundaries make sense
3. **Run summarizer prompts against each segment** → check summary quality against rubric
4. **Embed everything** → verify embedding dimensions and storage
5. **Run evaluation queries from `eval/retrieval-rubric.md`** → score results

## File Locations
```
Sessions: ~/.openclaw/agents/main/sessions/
Prompts: ~/dean/projects/semantic-memory/prompts/
Eval: ~/dean/projects/semantic-memory/eval/
```

## Notes
- Need to actually READ these sessions to confirm they contain the edge cases I described in the rubric (rapid topic pivots, implicit context, etc.)
- Will tag sessions with expected cluster types after reading
- The topic-ID sessions (with `-topic-` in filename) appear to be thread-specific — may have cleaner topic boundaries than main sessions
- Main sessions (no topic ID) are likely the multi-topic monsters we want for stress testing

# Session Summarizer Prompt

## Purpose
Takes a cleaned session log (tool calls, heartbeats, and system noise already stripped) and produces a structured summary for embedding into the semantic memory vector store.

## Output Schema

```json
{
  "session_id": "string — original session filename or ID",
  "date": "YYYY-MM-DD",
  "agent": "bob | patterson | dean",
  "domain": "string — primary domain tag (see Domain Taxonomy below)",
  "secondary_domains": ["string — additional domains if session crosses boundaries"],
  "shareable": false,
  "summary": {
    "topics": ["short topic descriptors — what was discussed"],
    "emotional_tenor": "one-line description of the emotional shape of the session",
    "decisions": ["specific decisions made or commitments given"],
    "unresolved_threads": ["things brought up but not resolved — open loops"],
    "patterns": ["recurring themes or behavioral observations"],
    "outcome": "one-line: productive/unproductive, resolved/interrupted, etc."
  },
  "one_line": "Single sentence capturing the essential 'what happened' — this is the session-level embedding text"
}
```

## Domain Taxonomy

Each agent has domain tags relevant to their user's world. The summarizer should pick the most specific applicable domain.

### Dean (User A's sessions)
- `child-school` — school events, class projects, teacher communications
- `child-activities` — extracurriculars, playdates, birthday parties
- `ecommerce-business` — Ecommerce operations
- `consulting-business` — Consulting operations
- `meal-planning` — grocery, recipes, food logistics
- `cricut-crafting` — vinyl cutting, labels, creative projects
- `books-reading` — book discussions, recommendations, reading log
- `household-logistics` — general life management, scheduling, errands
- `gift-bags` — holiday gift bag pipeline (Valentine's, Halloween, etc.)
- `deals-shopping` — bargain hunting, price comparisons, purchases
- `medical-health` — health-related discussions (VAULT sensitivity)
- `adhd-support` — executive function support, overwhelm management
- `cross-agent` — coordination with other agents

### Bob (User B's sessions)
- `red-hat-tam` — TAM work, customer cases
- `infrastructure` — home lab, networking, server management
- `business` — shared business tasks
- `coding` — development projects
- `cross-agent` — coordination with Patterson/Dean

### Patterson (User C's sessions)
- `red-hat-psirt` — PSIRT work, security
- `personal` — Personal tasks
- `business` — shared business tasks
- `infrastructure` — Home systems
- `cross-agent` — coordination with other agents

### Shared
- `household` — affects the whole household
- `ecommerce-business` — Ecommerce operations (shared context)
- `consulting-business` — Consulting operations (shared context)

## Prompt Template

```
You are a session summarizer for a personal AI assistant's memory system. Your job is to extract structured metadata from a cleaned conversation log.

RULES:
1. Be concise. Topics should be 3-7 words each. Emotional tenor is one line.
2. Decisions must be SPECIFIC — "decided to use Chroma" not "made a database decision."
3. Unresolved threads are things that were mentioned but not completed or decided. These are valuable — they represent open loops the human may return to.
4. Patterns are behavioral observations, not topic summaries. "User tends to add new projects when stressed" is a pattern. "Discussed vinyl suppliers" is a topic.
5. The one_line summary should capture the SESSION, not just the main topic. "User planned seasonal products while managing a schedule conflict" > "Valentine's bag planning."
6. Domain tagging: pick the PRIMARY domain. If the session meaningfully spans multiple domains (not just a passing mention), add secondary_domains.
7. Shareable defaults to FALSE. Only mark TRUE if the content is explicitly household/business content that should be available to other agents.
8. Emotional tenor should capture the ARC, not just a snapshot. "Started overwhelmed, settled into focused planning after breaking tasks down" > "stressed."

SESSION LOG:
{session_text}

Output valid JSON matching the schema above. No commentary outside the JSON.
```

## Segment Summarizer (for chunk-level prefixes)

Each topic segment within a session also needs a short prefix for its embedding. This is lighter-weight than the full session summary.

```
Summarize this conversation segment in one line. Include:
- What was being discussed
- The emotional tone (if notable)  
- Whether a decision was made or a question was left open

Format: [Topic: {topic} | Tone: {tone} | Status: {decided/open/interrupted}]

SEGMENT:
{segment_text}
```

## Quality Criteria for Summaries

A good summary should pass these checks:
1. **Retrievable**: If someone searched "Valentine's day gift bags," would this summary's embedding be semantically close? The topics and one_line must contain the right semantic signals.
2. **Distinguishable**: Could you tell this session apart from a similar one? The one_line should capture what made THIS session unique.
3. **Actionable**: Do the decisions and unresolved_threads give enough context to pick up where things left off?
4. **Emotionally calibrated**: Does the tenor capture how the session FELT, not just what it covered? This is what powers pattern-mode retrieval.
5. **Appropriately tagged**: Would filtering by this domain return relevant results? Would excluding this domain correctly filter it out?

## Notes

- The summarizer runs at INGEST TIME, not retrieval time. Speed matters but not as critically as at retrieval (no 3-second constraint here).
- For Dean's sessions specifically: The user's topic-switching is a FEATURE to capture, not noise to strip. If she pivots from bags to shoes to bags, the summary should note that — it's signal about the user's mental state (logistics mode, overwhelmed, etc.).
- Emotional tenor is the secret weapon for pattern-mode retrieval. "Find me sessions that felt like this" only works if tenor is captured accurately and consistently.

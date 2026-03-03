"""Push triggers for proactive memory surfacing.

Identifies high-activation memories that should be surfaced without
explicit query — the 'tip of the tongue' phenomenon.

Designed to run as a periodic scan (heartbeat/cron), not real-time.
"""


def should_push(
    activation_weight: float,
    confidence: float,
    hours_since_last_push: float,
    has_unresolved_threads: bool,
) -> bool:
    """Determine if a memory should be proactively surfaced.

    Args:
        activation_weight: ACT-R activation weight (from Phase 1). Range (0, ~1.2].
        confidence: Bayesian confidence (from Phase 2). Range [0.1, 2.0].
        hours_since_last_push: Hours since this memory was last pushed.
            Use float('inf') or a large number for never-pushed memories.
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

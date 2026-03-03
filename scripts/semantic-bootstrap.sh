#!/usr/bin/env bash
# semantic-bootstrap.sh — Run at session start to load relevant semantic context
# Usage: semantic-bootstrap.sh <agent> "<topic or first message>"
# Output: Formatted context block if relevant memories found, empty otherwise
#
# Designed to be called by agents at conversation start. Output is suitable
# for injection into agent context.

set -euo pipefail

AGENT="${1:-bob}"
QUERY="${2:-}"

if [ -z "$QUERY" ]; then
  echo ""
  exit 0
fi

HOSTNAME=$(hostname -s 2>/dev/null || echo "unknown")

# Set Ollama URL if not already configured
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Dynamically find the project directory instead of hardcoding hostnames
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ensure we're using the data dir in the repo, unless overridden
export SEMANTIC_MEMORY_STORE="${SEMANTIC_MEMORY_STORE:-$PROJECT_DIR/data/vectors}"

cd "$PROJECT_DIR"

# Run context retrieval — suppress errors, return empty on failure
uv run python -m src.retrieval.recall --agent "$AGENT" --context "$QUERY" 2>/dev/null || echo ""

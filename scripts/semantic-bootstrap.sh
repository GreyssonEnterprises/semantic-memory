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

# Set remote Ollama if not on studio
HOSTNAME=$(hostname -s 2>/dev/null || echo "unknown")
if [ "$HOSTNAME" != "studio" ]; then
  export OLLAMA_URL="${OLLAMA_URL:-http://192.168.1.21:11434}"
fi

# Find project dir based on agent/hostname
if [ "$HOSTNAME" = "shale" ]; then
  PROJECT_DIR="$HOME/clawd/projects/semantic-memory"
elif [ "$HOSTNAME" = "caroline" ]; then
  PROJECT_DIR="$HOME/dean/projects/semantic-memory"
  export SEMANTIC_MEMORY_STORE="$PROJECT_DIR/data/vectors"
else
  PROJECT_DIR="$HOME/clawd/projects/semantic-memory"
fi

cd "$PROJECT_DIR"

# Run context retrieval — suppress errors, return empty on failure
uv run python -m src.retrieval.recall --agent "$AGENT" --context "$QUERY" 2>/dev/null || echo ""

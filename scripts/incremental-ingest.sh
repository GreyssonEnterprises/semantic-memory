#!/usr/bin/env bash
# incremental-ingest.sh — Ingest new sessions for local agent
# Run via OpenClaw cron.
#
# Flow:
#   1. Ingest new sessions for the agent running on this machine
#   2. Backup all stores to NAS (if on bob/studio)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SEMANTIC_MEMORY_STORE="${SEMANTIC_MEMORY_STORE:-$PROJECT_DIR/data/vectors}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-openai}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENAI_API_KEY_EMBEDDINGS:-}}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"

cd "$PROJECT_DIR"

# Set agent via env var, or default to bob.
AGENT="${SEMANTIC_MEMORY_AGENT:-bob}"

LOG_FILE="data/logs/ingest-$(date +%Y-%m-%d).log"
mkdir -p data/logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "=== Incremental ingest starting for $AGENT ==="

# Step 1: Ingest new sessions for agent
# The ingest script skips already-embedded sessions (checks by session_id in Chroma)

log "Ingesting $AGENT sessions..."
SESSION_DIR="${SEMANTIC_SESSION_DIR:-$HOME/.openclaw/agents/$AGENT/sessions}"
if [ ! -d "$SESSION_DIR" ]; then
    log "WARNING: Session directory not found at $SESSION_DIR"
else
    uv run python -m src.pipeline.ingest \
      --agent "$AGENT" \
      --session-dir "$SESSION_DIR" \
      --mode topic \
      --min-messages 4 2>>"$LOG_FILE" || log "WARNING: $AGENT ingest failed"
fi

# Step 2: Backup to NAS (only typically mounted on the primary machine)
if [ "$AGENT" = "bob" ]; then
    NAS_BACKUP="${SEMANTIC_MEMORY_BACKUP:-/Volumes/owc-express/gdrive-personal/areas/office-org/infrastructure/semantic-memory/data/vectors-backup}"
    if [ -d "$(dirname "$NAS_BACKUP")" ]; then
      log "Backing up stores to NAS..."
      mkdir -p "$NAS_BACKUP"
      rsync -az --no-group --no-owner data/vectors/ "$NAS_BACKUP/" 2>>"$LOG_FILE" \
        || log "WARNING: NAS backup failed"
    else
      log "NAS not mounted, skipping backup"
    fi
fi

# Summary
SEGS=$(uv run python -c "
from src.pipeline.store import VectorStore
import sys
try:
    s = VectorStore('$AGENT')
    print(s.segments.count())
except Exception as e:
    print('?', file=sys.stderr)
" 2>/dev/null || echo "?")

log "=== Ingest complete. Segments: $AGENT=$SEGS ==="
echo "Incremental ingest complete. $AGENT=$SEGS segments."

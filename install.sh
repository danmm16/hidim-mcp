#!/usr/bin/env bash
# Creates a virtualenv next to this file, installs hidim-mcp, runs the smoke test,
# and prints a ready-to-paste Claude Desktop config block with absolute paths.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"
EXTRAS="${EXTRAS:-all}"          # EXTRAS=core for the minimal install

if command -v uv >/dev/null 2>&1; then
  uv venv .venv >/dev/null
  if [ "$EXTRAS" = "core" ]; then uv pip install --python .venv/bin/python -e . ; else uv pip install --python .venv/bin/python -e ".[$EXTRAS]" ; fi
else
  "$PY" -m venv .venv
  if [ "$EXTRAS" = "core" ]; then .venv/bin/pip install -q -e . ; else .venv/bin/pip install -q -e ".[$EXTRAS]" ; fi
fi

echo "Running smoke test..."
.venv/bin/python tests/stdio_smoke.py | tail -3

ROOTS="${HIDIM_ROOTS:-$HOME}"
OUT="${HIDIM_OUT:-$HOME/hidim_out}"
cat <<EOF

Add this to Claude Desktop (Settings > Developer > Edit Config), then quit and reopen Claude:

{
  "mcpServers": {
    "hidim": {
      "command": "$HERE/.venv/bin/python",
      "args": ["-m", "hidim_mcp.server"],
      "env": { "HIDIM_ROOTS": "$ROOTS", "HIDIM_OUT": "$OUT" }
    }
  }
}

Or let the script do the merge for you (backs up the file first):
  $HERE/.venv/bin/python $HERE/register.py

Or for Claude Code:
  claude mcp add --transport stdio --scope user hidim -e HIDIM_ROOTS=$ROOTS -e HIDIM_OUT=$OUT -- $HERE/.venv/bin/python -m hidim_mcp.server
EOF

#!/bin/sh
set -eu

mkdir -p /logs/verifier

OUT_FILE="/logs/agent/cursor-cli.txt"

if [ -f "$OUT_FILE" ]; then
  # Normalize whitespace to reduce flakiness from extra newlines.
  TEXT="$(cat "$OUT_FILE" | tr -d '\r' | tr -s '\n' '\n' | sed -e 's/[[:space:]]*$//')"
else
  TEXT=""
fi

# Check that the agent actually acted in the workspace.
PROOF_FILE="/workspace/smoke_proof.txt"
PROOF_OK=0
if [ -f "$PROOF_FILE" ]; then
  PROOF_TEXT="$(cat "$PROOF_FILE" | tr -d '\r' | tr -s '\n' '\n' | sed -e 's/[[:space:]]*$//')"
  if [ "$PROOF_TEXT" = "CURSOR_SMOKE_OK" ]; then
    PROOF_OK=1
  fi
fi

# Check that the final line of output is exactly "OK".
LAST_LINE="$(printf "%s\n" "$TEXT" | tail -n 1 | tr -d '\r')"
OUTPUT_OK=0
if [ "$LAST_LINE" = "OK" ]; then
  OUTPUT_OK=1
fi

if [ "$PROOF_OK" -eq 1 ] && [ "$OUTPUT_OK" -eq 1 ]; then
  printf '{"reward": 1.0}\n' > /logs/verifier/reward.json
else
  printf '{"reward": 0.0, "proof_ok": %s, "output_ok": %s, "last_line": %s}\n' \
    "$PROOF_OK" "$OUTPUT_OK" "$(printf "%s" "$LAST_LINE" | jq -Rs .)" \
    > /logs/verifier/reward.json
fi

exit 0


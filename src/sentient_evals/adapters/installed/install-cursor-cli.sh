#!/bin/bash
set -euo pipefail

apt-get update
apt-get install -y curl ca-certificates

curl -fsS https://cursor.com/install | bash



# Cursor's installer exposes the CLI as `agent` (and sometimes also `cursor-agent`).
export PATH="$HOME/.local/bin:$PATH"
command -v agent >/dev/null 2>&1 || (echo "agent not found after install" >&2 && exit 1)

echo "INSTALL SUCCESS!"

#!/bin/bash
set -e

# If Claude is preinstalled in the environment template, skip reinstall.
if command -v claude >/dev/null 2>&1; then
    claude --version || true
    exit 0
fi

# Install curl if not available
if command -v apk &> /dev/null; then
    apk add --no-cache curl bash
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y curl
fi

# Install Claude Code using the official installer
{% if version %}
curl -fsSL https://claude.ai/install.sh | bash -s -- {{ version }}
{% else %}
curl -fsSL https://claude.ai/install.sh | bash
{% endif %}

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

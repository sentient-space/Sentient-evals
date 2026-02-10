FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsS https://cursor.com/install | bash \
  && export PATH="$HOME/.local/bin:$PATH" \
  && command -v agent >/dev/null 2>&1

ENV PATH="$HOME/.local/bin:$PATH"

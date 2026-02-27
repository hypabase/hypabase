#!/usr/bin/env bash
# Hypabase Memory MCP server — zero-dependency launcher.
# Downloads a pre-built standalone binary on first run, caches it, then executes.
# All status output goes to stderr (stdout is reserved for MCP JSON-RPC).
set -euo pipefail

VERSION="0.2.0"
REPO="hypabase/hypabase"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hypabase/bin"
BINARY="$CACHE_DIR/hypabase-memory-$VERSION"

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) echo "error: unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

case "$OS" in
  darwin|linux) ;;
  *) echo "error: unsupported OS: $OS" >&2; exit 1 ;;
esac

ASSET="hypabase-memory-${OS}-${ARCH}"

if [ ! -x "$BINARY" ]; then
  URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"
  echo "hypabase-memory: downloading v${VERSION} (${OS}/${ARCH})..." >&2
  mkdir -p "$CACHE_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$BINARY"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$BINARY" "$URL"
  else
    echo "error: curl or wget required to download binary" >&2
    exit 1
  fi
  chmod +x "$BINARY"
  echo "hypabase-memory: cached at $BINARY" >&2
fi

exec "$BINARY" "$@"

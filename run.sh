#!/usr/bin/env bash
#
# One command to run everything: create the virtual environment, install
# dependencies (when needed), check optional external tools, and launch the app.
#
#   ./run.sh
#
# Optional environment variables:
#   PYTHON=python3.11 ./run.sh   # interpreter to use
#   SKIP_DEP_CHECK=1 ./run.sh    # don't check/offer to install ffmpeg/swiftc

set -euo pipefail

# Move to the script's directory (so it works from anywhere).
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"
REQ="requirements.txt"
STAMP="$VENV/.requirements.sha"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: '$PYTHON' not found. Install Python 3 or set PYTHON=…" >&2
  exit 1
fi

# Ask a yes/no question (default Yes). Returns non-zero for "no" or when there is
# no interactive terminal (so non-interactive runs never block).
prompt_yes() {
  [ -t 0 ] || return 1
  printf "    %s [Y/n] " "$1"
  local ans
  read -r ans || return 1
  case "${ans:-Y}" in [Nn]*) return 1 ;; *) return 0 ;; esac
}

# 1) Create the virtual environment if missing.
if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment ($VENV)…"
  "$PYTHON" -m venv "$VENV"
fi

VPY="$VENV/bin/python"

# 2) Install Python dependencies only when requirements.txt changed (or first run).
NEWSHA="$(shasum "$REQ" | awk '{print $1}')"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$NEWSHA" ]; then
  echo "==> Installing dependencies (first run downloads large packages like torch; this can take a while)…"
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -r "$REQ"
  echo "$NEWSHA" > "$STAMP"
fi

# 3) Check optional external tools and offer to install missing ones.
#    Both enable optional features; the app runs (degraded) without them.
if [ "${SKIP_DEP_CHECK:-0}" != "1" ]; then
  # ffmpeg — required to import/transcribe existing media files.
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "==> Optional tool 'ffmpeg' is missing (needed to import audio/video files)."
    if command -v brew >/dev/null 2>&1; then
      if prompt_yes "Install ffmpeg with Homebrew now?"; then
        brew install ffmpeg || echo "    Could not install ffmpeg; try 'brew install ffmpeg' later."
      fi
    else
      echo "    Homebrew not found. Install it from https://brew.sh then run: brew install ffmpeg"
    fi
  fi

  # swiftc / Xcode Command Line Tools — required to capture system audio.
  if ! command -v swiftc >/dev/null 2>&1; then
    echo "==> Optional tool 'swiftc' is missing (needed for system-audio capture)."
    if prompt_yes "Install the Xcode Command Line Tools now? (a GUI installer opens)"; then
      xcode-select --install || true
      echo "    Finish the installer, then re-run ./run.sh to use system-audio capture."
    fi
  fi
fi

# 4) Launch the app (pass through any arguments).
exec "$VPY" meeting-transcriptor.py "$@"

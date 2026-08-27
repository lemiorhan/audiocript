#!/usr/bin/env bash
#
# One command to run everything: create the virtual environment, install
# dependencies (when needed), check optional external tools, and launch the app.
#
#   ./run.sh
#   ./run.sh --dictate           # run the dictation daemon instead of the TUI
#
# Optional environment variables:
#   PYTHON=python3.12 ./run.sh   # interpreter to use
#   SKIP_DEP_CHECK=1 ./run.sh    # don't check/offer to install ffmpeg/swiftc

set -euo pipefail

# Move to the script's directory (so it works from anywhere).
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"
REQ="requirements.txt"
AEC_REQ="requirements-aec.txt"
STAMP="$VENV/.requirements.sha"
MIN_PY="3.12"                   # minimum supported Python (see README Requirements)

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: '$PYTHON' not found. Install Python 3 or set PYTHON=…" >&2
  exit 1
fi

# Print an interpreter's version; exit 2 when it is older than MIN_PY. Python
# itself does the comparison, so "3.9" vs "3.10" cannot sort wrong.
python_version() {
  "$1" - "$MIN_PY" 2>/dev/null <<'PY'
import sys
want = tuple(int(p) for p in sys.argv[1].split("."))
print(".".join(str(n) for n in sys.version_info[:3]))
sys.exit(0 if sys.version_info[:len(want)] >= want else 2)
PY
}

# Refuse to go on with too old an interpreter: what it breaks surfaces much
# later, in the middle of a dependency build, as something unrelated.
require_python() {
  local py="$1" hint="$2" ver
  if ver="$(python_version "$py")"; then
    return 0
  fi
  if [ -z "$ver" ]; then
    echo "Error: could not read the Python version from '$py'." >&2
  else
    echo "Error: Python $MIN_PY or newer is required; '$py' is $ver." >&2
  fi
  echo "    $hint" >&2
  exit 1
}

require_python "$PYTHON" \
  "Install a newer one (brew install python@$MIN_PY), or point at one you have: PYTHON=python$MIN_PY ./run.sh"

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

# An environment created earlier by an older interpreter is not covered by the
# check above — that one only looked at $PYTHON.
require_python "$VPY" \
  "The existing $VENV was built with it; remove it (rm -rf $VENV) and re-run."

# 2) Install Python dependencies only when either requirements file changed (or first run).
NEWSHA="$(shasum "$REQ" "$AEC_REQ" | shasum | awk '{print $1}')"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$NEWSHA" ]; then
  echo "==> Installing dependencies (first run downloads large packages like torch; this can take a while)…"
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -r "$REQ"
  if ! "$VPY" -m pip install -r "$AEC_REQ"; then
    echo "Warning: Could not install optional WebRTC echo cancellation; starting without it." >&2
    echo "    Retry later: $VPY -m pip install -r $AEC_REQ" >&2
  fi
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

# 4) Launch the app (pass through any remaining arguments).
#    --dictate runs the dictation daemon (dictate.py) instead of the TUI; it is
#    consumed here so it is never forwarded to audiocript.py, which reads no
#    arguments at all and would otherwise launch the TUI silently.
SCRIPT="audiocript.py"
if [ "${1:-}" = "--dictate" ]; then
  SCRIPT="dictate.py"
  shift
fi
exec "$VPY" "$SCRIPT" "$@"

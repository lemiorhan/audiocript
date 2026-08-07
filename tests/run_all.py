#!/usr/bin/env python3
"""Run every test script and report which ones failed.

    .venv/bin/python tests/run_all.py

Each test file is also runnable on its own, which is usually what you want while
working on one area:

    .venv/bin/python tests/test_finalize.py
"""
import subprocess
import sys
import time
from pathlib import Path

TESTS = sorted(Path(__file__).resolve().parent.glob("test_*.py"))


def main():
    failed = []
    for path in TESTS:
        print(f"\n=== {path.name} " + "=" * (60 - len(path.name)))
        started = time.time()
        result = subprocess.run([sys.executable, str(path)])
        elapsed = time.time() - started
        if result.returncode:
            failed.append(path.name)
            print(f"--- {path.name} FAILED in {elapsed:.0f}s")
        else:
            print(f"--- {path.name} passed in {elapsed:.0f}s")
    print()
    if failed:
        print(f"{len(failed)} of {len(TESTS)} files failed: {', '.join(failed)}")
        return 1
    print(f"all {len(TESTS)} test files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Contributing to Voice Transcriptor

Thanks for your interest in contributing! This document explains how to set up the
project, the conventions we follow, and how to propose changes.

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

This is a macOS app (it uses Core Audio process taps and `termios`).

```bash
git clone <your-fork-url>
cd voice_transcriptor
./run.sh        # creates .venv, installs deps, and launches the app
```

Requirements: macOS 14.4+, Python 3.10+, and Xcode Command Line Tools (`swiftc`)
for the optional system-audio capture.

## Ways to contribute

- **Bug reports** — open an issue using the *Bug report* template. Include your
  macOS and Python versions and exact steps to reproduce.
- **Feature requests** — open an issue using the *Feature request* template.
- **Pull requests** — see below.

## Pull requests

1. Create a branch from `main` (`git switch -c feature/short-description`).
2. Keep changes focused; one logical change per PR.
3. Match the existing code style (standard library + the dependencies already in
   `requirements.txt`; avoid adding heavy dependencies without discussion).
4. If you change behavior, update the `README.md` and the design notes under
   `docs/superpowers/specs/` where relevant.
5. Verify your change:
   - `python -m py_compile meeting-transcriptor.py`
   - Run the app (`./run.sh`) and exercise the affected paths (record, language,
     mic picker, system-audio toggle, transcription).
   - For audio/Swift changes, test on a real terminal — interactive audio capture
     and the full-screen UI cannot be exercised in headless/CI environments.
6. Open the PR using the template and describe what you changed and how you tested.

## Commit messages

Write clear, imperative commit subjects (e.g. "Fix mic picker selection on
reconnect"). Explain the *why* in the body when it isn't obvious.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind and
constructive.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).

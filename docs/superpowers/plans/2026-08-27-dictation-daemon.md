# Dictation Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One global hotkey records the microphone; Whisper transcribes locally, an AI provider corrects only punctuation and fillers, and the corrected text lands on the clipboard ready to paste into any application.

**Architecture:** Two new modules. `dictation.py` holds everything with no threads and no global state — configuration resolution, the clipboard, the status sinks, the correction/delivery pipeline. `dictate.py` holds everything that does — the state machine, the hotkey listener, the capture, the signals, the CLI. Both import `audiocript` and `publish`; neither is imported by them.

**Tech Stack:** Python 3.12+, `pynput` (`keyboard.GlobalHotKeys`), `sounddevice` via `audiocript.DeviceRecorder`, Whisper via `audiocript.transcribe_audio`, OpenAI via `publish.openai_complete`, macOS `pbcopy` / `osascript` / `afplay`.

**Spec:** `docs/superpowers/specs/2026-08-27-dictation-daemon-design.md` — read it alongside this plan; the plan argues from it.

## How to read this plan

**No implementation bodies.** Each task gives the signature, the decisions, and the acceptance tests. Writing the body twice — once here, once in the file — means every defect in the plan's copy becomes a defect in the code, and the plan gets reopened to fix it.

**Test code appears in full only where getting it wrong is likely** — threading, the clipboard-safety rules, the permission check. Where a task's tests are variations on one stated contract, that contract is written out once and the cases are listed. Both are complete instructions; neither is a placeholder.

If a step leaves you genuinely unable to proceed, that is a plan defect. Say so rather than guessing.

## Assumption audit

Run against the code before this plan was dispatched: **32 assumptions checked, 4 wrong.** All four were corrected in the text below rather than noted beside it, because an implementer reads the brief, not the errata.

| # | Assumption | Verdict |
|---|---|---|
| A4 | `_stream_source_to_wav` takes a raw file path | **Wrong** — it takes an `_RawPcmReader` and calls `len()` on it. Corrected in Task 4. |
| A23 | The fake provider must be copied from `test_publish.py` and modified to capture the prompt | **Wrong** — `FakeAPI` already records parsed request bodies, so it is reused unchanged. Corrected in Task 3. |
| A24 | The shared test doubles can live in `test_dictation.py` and be imported by `test_dictate.py` | **Wrong, and it would have passed.** `support.run` ends in `sys.exit`, so the import runs the other file's whole suite and exits 0 first. Doubles moved to `support.py`; every runner guarded. |
| A25 | `run.sh` can grow a `--dictate` switch by following its structure | **Wrong as stated** — its final `exec "$VPY" audiocript.py "$@"` forwards the switch to a program that ignores arguments, silently launching the TUI. Corrected in Task 6. |

The other 28 were confirmed against the code, including every `audiocript` and `publish` symbol this plan calls, `GlobalHotKeys`'s constructor and attributes, `HotKey.parse`'s `ValueError` and its `Key`-membership behaviour, `publish.config()` returning `None` without GitHub credentials, `render_prompt`'s placeholder handling, `os.kill(pid, 0)`'s two exceptions, and that importing `audiocript` neither starts the TUI nor costs more than a second.

## Global Constraints

- **Platform:** macOS 14.4+ only. `pbcopy`, `osascript`, `afplay` and the CoreGraphics event tap are macOS-specific.
- **Python:** 3.12+ (`run.sh` refuses anything older).
- **No new dependencies.** `pynput` is already in `requirements.txt` and already imported by `audiocript.py`.
- **Codebase language is English** — code, comments, commit messages, README. `prompts/*.md` are Turkish; the new prompt is Turkish too.
- **No test may reach a real API, open a real microphone, or leave the real clipboard changed.**
- **Tests are plain scripts.** No framework. Each file ends with `run([...], globals())` from `tests/support.py`; `tests/run_all.py` finds them by `test_*.py`. One file: `.venv/bin/python tests/test_x.py`. All: `.venv/bin/python tests/run_all.py`.
- **Dependency direction is one-way.** `dictate.py` → `dictation.py`, `audiocript.py`, `publish.py`. None of those three imports either new module.
- **`audiocript.py` is not modified.** Outside the new files, only `run.sh`, `README.md` and `CHANGELOG.md` are touched.
- **Clipboard safety rule:** the clipboard is written exactly once per dictation, and not at all when there is nothing to write. Destroying what the user had is the worst available outcome.
- **Doubles shared by both test files live in `tests/support.py`** — which already exists to carry what every test needs. Do **not** import them from one test file into another: `support.run` ends in `sys.exit`, so importing a test file executes its whole suite and exits before the importing file runs anything, with status 0. It would look like a pass.
- **Every test file ends with `if __name__ == "__main__":` before its `run(...)` call**, matching `tests/test_publish.py` and `tests/test_stream_resample.py`. This is what makes a file importable at all.
- **`tests/test_publish.py` is safe to import from** — it guards its own `run(...)`, verified by importing it. Its `FakeAPI` and `completion` helpers are reused rather than reimplemented.
- Python puts a script's own directory at `sys.path[0]` regardless of the working directory, so `from support import …` works whether a test is run directly or through `run_all.py`. Verified.

---

### Task 1: Provider assumption, then configuration

**Files:**
- Create: `dictation.py`
- Test: `tests/test_dictation.py`

**Interfaces:**
- Consumes: `audiocript.LANGUAGES`, `publish.env_value`, `pynput.keyboard.HotKey.parse`
- Produces:

```python
class ConfigError(Exception): ...

HOTKEY_ACTIONS = ("toggle",)
DEFAULT_HOTKEYS = {"toggle": "<cmd>+<alt>+d"}
DEFAULT_MAX_SECONDS = 300
DEFAULT_MODEL = "gpt-4.1-mini"

class DictationConfig:            # plain data holder; a dataclass is fine
    language: str                 # a key of audiocript.LANGUAGES
    hotkeys: dict                 # action -> combination, every action in HOTKEY_ACTIONS
    max_seconds: int
    model: str
    openai_token: str
    openai_base: str

def resolve_config(cfg, env=None) -> DictationConfig
    # cfg: the dict from audiocript.load_config()
    # env: callable(key, default=None); defaults to publish.env_value
    # raises ConfigError whose message contains the offending value
```

- [ ] **Step 1: Settle the provider assumption before anything is built on it**

The spec's first unverified assumption is that `gpt-4.1-mini` exists and answers. One real call, run by hand — no test in this plan reaches a real API.

```bash
cd /Users/lemiorhan/Code/Misc/audiocript
.venv/bin/python - <<'PY'
import publish
cfg = {"openai_base": publish.env_value("OPENAI_API_BASE", "https://api.openai.com"),
       "openai_token": publish.env_value("OPENAI_API_KEY"), "model": "gpt-4.1-mini"}
assert cfg["openai_token"], "OPENAI_API_KEY yok"
print(publish.openai_complete("Sadece noktalama ekle: eee bu pr'i bugun mergeleyemeyiz", cfg))
PY
```

If it answers, `DEFAULT_MODEL` stays. If the model id is rejected, pick the cheapest correction-grade model available, change `DEFAULT_MODEL`, and record what changed and why in this task's commit message. Do not build on an unverified default.

- [ ] **Step 2: Write the failing tests**

First add the three doubles both test files need to `tests/support.py`, beside the helpers already there — `FakeClipboard` and `RecordingSink` are used by Tasks 2–4, `fake_env` by Tasks 1–5. They must not live in a test file; see the Global Constraints.

```python
def fake_env(**values):
    """An `env` callable backed by a dict, matching publish.env_value's signature."""
    def env(key, default=None):
        return values.get(key, default)
    return env


class FakeClipboard:
    def __init__(self, initial="ONCEKI ICERIK"):
        self.text, self.writes = initial, 0

    def copy(self, text):
        self.text = text
        self.writes += 1


class RecordingSink:
    def __init__(self):
        self.calls = []

    def recording(self): self.calls.append(("recording", None))
    def processing(self): self.calls.append(("processing", None))
    def done(self, text): self.calls.append(("done", text))
    def failed(self, reason): self.calls.append(("failed", reason))
```

Then create `tests/test_dictation.py`, its docstring saying nothing here may reach a real API or read a developer's `.env` — every test passes an explicit `env`, so `publish.env_value` is never consulted. Its one local helper:

```python
BASE_ENV = dict(OPENAI_API_KEY="sk-test")


def _refuses(cfg, env, because):
    """resolve_config must raise ConfigError, and the message must name the value."""
    try:
        dictation.resolve_config(cfg, env)
    except dictation.ConfigError as e:
        assert because in str(e), f"message {str(e)!r} does not name {because!r}"
        return
    raise AssertionError(f"accepted {cfg!r}; expected ConfigError naming {because!r}")
```

**Acceptance tests.** Every rejection case is one `_refuses(...)` call; every acceptance case asserts the resolved field. Write one test function per row, named as given.

| Test | Asserts |
|---|---|
| `test_defaults_when_config_is_empty` | `{}` + `BASE_ENV` → `language == "tr"`, `hotkeys == DEFAULT_HOTKEYS`, `max_seconds == DEFAULT_MAX_SECONDS`, `model == DEFAULT_MODEL`, `openai_base == "https://api.openai.com"` |
| `test_config_values_are_used` | `{"language": "en", "dictation_hotkeys": {"toggle": "<cmd>+<shift>+<alt>+k"}, "dictation_max_seconds": 42}` → each value survives |
| `test_env_overrides_model_and_base` | `DICTATION_MODEL` and `OPENAI_API_BASE` reach `model` and `openai_base` |
| `test_a_partial_hotkeys_object_keeps_the_defaults` | `{"dictation_hotkeys": {}}` → `hotkeys == DEFAULT_HOTKEYS` |
| `test_every_known_language_is_accepted` | every key of `A.LANGUAGES` resolves without raising |
| `test_missing_api_key_is_refused` | `_refuses({}, fake_env(), "OPENAI_API_KEY")` |
| `test_unknown_language_is_refused` | `_refuses({"language": "de"}, …, "de")` — a fallback would transcribe with a model the user did not ask for |
| `test_unparseable_hotkey_is_refused` | `_refuses` for each of `"cmd+alt+d"`, `"<zort>+d"`, `""` — `HotKey.parse` raises `ValueError` on all three |
| `test_hotkey_without_a_modifier_is_refused` | `_refuses({"dictation_hotkeys": {"toggle": "d"}}, …, "d")` — `parse` accepts a bare `d`, which would fire on every `d` typed anywhere |
| `test_unknown_hotkey_action_is_refused` | `_refuses({"dictation_hotkeys": {"togle": "<cmd>+<alt>+d"}}, …, "togle")` |
| `test_bad_max_seconds_is_refused` | `_refuses` for `"soon"` and for `0` |
| `test_it_resolves_where_publish_config_would_not` | with only `OPENAI_API_KEY` set and `publish.ROOT` pointed at an empty directory, `publish.config()` is `None` while `resolve_config` succeeds — dictation must not inherit publishing's requirement for GitHub credentials |

The file needs `from support import A, run, fake_env`, `import dictation`, and `import publish` for the last row.

End the file with the guarded runner. The name scan is a deliberate divergence from the explicit lists the existing test files use: an enumerated list lets a newly written test be silently never run.

```python
if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: FAIL, the file erroring on `import dictation`.

- [ ] **Step 4: Implement the configuration half of `dictation.py`**

Module docstring: the parts of dictation holding no threads and no global state, so they can be tested without audio hardware or a running daemon.

Decisions the signature does not carry:

- Read in order: `language` (default `"tr"`), `dictation_hotkeys` merged **over** `DEFAULT_HOTKEYS` so a partial object keeps the rest, `dictation_max_seconds`, then `OPENAI_API_KEY`, `DICTATION_MODEL`, `OPENAI_API_BASE` through `env`.
- `env` defaults to `publish.env_value`. It is a parameter so tests never touch a real `.env`.
- Every `ConfigError` message **contains the offending value** and says what was expected. The tests assert that substring, and it is also what the user reads at startup — write a sentence, not a repr.
- Validate hotkeys with `keyboard.HotKey.parse`, catching `ValueError`; then require at least one `keyboard.Key` member in the parsed result, which is what "has a modifier" means.
- Reject an action outside `HOTKEY_ACTIONS`, naming the ones understood. A silently ignored hotkey is indistinguishable from a missing permission.
- `dictation_max_seconds` must be an `int` greater than zero.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: PASS for every test.

- [ ] **Step 6: Prove the validation is load-bearing**

Use the `mutation-gate` skill. For each of the four rejection paths — language, hotkey parse, missing modifier, unknown action — delete the check, mark it `# MUTANT-n` with the change, run, confirm the matching test fails, restore in the same turn. A check that can be deleted with the suite still green is decoration.

- [ ] **Step 7: Commit**

```bash
git add dictation.py tests/test_dictation.py
git commit -m "feat: resolve and validate the dictation configuration"
```

---

### Task 2: Clipboard and status sinks

**Files:**
- Modify: `dictation.py`
- Test: `tests/test_dictation.py`

**Interfaces:**
- Consumes: Task 1's module
- Produces:

```python
PREVIEW_CHARS = 60

class Clipboard:
    def copy(self, text: str) -> None          # pbcopy; raises on a non-zero exit

class StatusSink:                               # the contract, four calls
    def recording(self) -> None
    def processing(self) -> None
    def done(self, text: str) -> None
    def failed(self, reason: str) -> None

class NotifySink(StatusSink):
    def __init__(self, notify=None, sound=None): ...   # injected for tests
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dictation.py`, above the guarded `run(...)`. `FakeClipboard` and `RecordingSink` already exist in `support.py` from Task 1 — import them, do not redefine them.

The one test worth writing out, because the encoding is the thing under test and a developer's clipboard must survive the suite:

```python
def test_clipboard_round_trips_turkish():
    import subprocess
    before = subprocess.run(["pbpaste"], capture_output=True).stdout
    try:
        text = "Şu PR'ı bugün merge edemeyiz — çünkü İĞÜÖÇ testleri geçmiyor."
        dictation.Clipboard().copy(text)
        back = subprocess.run(["pbpaste"], capture_output=True).stdout.decode("utf-8")
        assert back == text, f"clipboard {back!r} != {text!r}"
    finally:
        subprocess.run(["pbcopy"], input=before)
```

The sink tests all build `NotifySink(notify=<collector>, sound=<collector>)` and assert on what was collected:

| Test | Asserts |
|---|---|
| `test_notify_sink_reports_each_stage` | `recording()`, `processing()`, `done("Bir cümle.")`, `failed("mikrofon açılamadı")` produce four notifications; at least one sound cue; `done`'s text and `failed`'s reason appear in the messages |
| `test_notify_sink_truncates_a_long_preview` | `done("x" * 500)` produces a message under 200 characters |
| `test_notify_sink_survives_a_failing_notifier` | with a `notify` that raises `OSError`, `done(...)` and `failed(...)` return normally — a broken `osascript` must not cost the user a clipboard they already have |

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: the new tests FAIL with `AttributeError` on `dictation.Clipboard` / `dictation.NotifySink`; Task 1's tests still PASS.

- [ ] **Step 3: Implement the clipboard and the sink**

- `Clipboard.copy` pipes UTF-8 through `pbcopy` and checks the exit status. A class, not a function, so the daemon and its tests can substitute a fake.
- `StatusSink` documents the four calls and exists so `dictate.py` never learns `osascript` is involved — the menu bar indicator the spec defers becomes a second implementation, not a change to the daemon.
- `NotifySink.__init__` takes `notify` and `sound` callables, defaulting to helpers that shell out to `osascript -e 'display notification … with title "Audiocript"'` and to `afplay` on a file under `/System/Library/Sounds/`. Both were confirmed present on the target machine.
- Play a cue on `recording` and on `processing`: the start and the stop of speaking are the two moments the user must feel without looking.
- `done` previews the text truncated to `PREVIEW_CHARS` with an ellipsis and says the clipboard is ready. `failed` shows the reason verbatim.
- **Wrap every `notify` and `sound` call so nothing propagates.** The status report is the least important thing happening; losing it must never cost a dictation that otherwise succeeded. Send the swallowed exception to `audiocript._log_problem` so it is not lost entirely.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: PASS for every test.

- [ ] **Step 5: Commit**

```bash
git add dictation.py tests/test_dictation.py
git commit -m "feat: add the dictation clipboard and status sink"
```

---

### Task 3: The correction prompt and the delivery pipeline

**Files:**
- Create: `prompts/dictation_prompt.md`
- Modify: `dictation.py`
- Test: `tests/test_dictation.py`

**Interfaces:**
- Consumes: Tasks 1–2; `publish.render_prompt`, `publish.openai_complete`
- Produces:

```python
CORRECTED   = "corrected"      # the provider's text is on the clipboard
UNCORRECTED = "uncorrected"    # the provider failed; the raw transcript is on the clipboard
SKIPPED     = "skipped"        # the reply broke the length contract; raw transcript
EMPTY       = "empty"          # nothing was said; the clipboard was not touched

MAX_GROWTH = 2.0               # a reply longer than this multiple of the input is refused
MIN_SHRINK = 0.4               # a reply shorter than this fraction of the input is refused

PROMPT_PATH = <repo root> / "prompts" / "dictation_prompt.md"

class Delivery:
    status: str                # one of the four above
    text: str                  # what reached the clipboard; "" for EMPTY
    detail: str                # the reason, for the notification

def deliver(transcript, config, clipboard, sink, complete=None) -> Delivery
    # complete: callable(prompt, cfg_dict) -> str; defaults to publish.openai_complete
```

- [ ] **Step 1: Write the prompt**

Create `prompts/dictation_prompt.md` in Turkish, in the register of the two existing prompts. It **must end with a bracketed placeholder line**: `publish.render_prompt` replaces the last line matching `^\[[^\]\n]*\]\s*$`, and silently appends to the end of the file when there is none.

The contract, stated plainly and then as a do/don't list:

**Yap:** noktalama ve büyük harf ekle · anlam taşımayan dolguları at (eee, ııı, yani, hani, işte) · kekemelik ve tekrarı temizle ("bu bu PR" → "bu PR") · Whisper'ın yanlış duyduğu teknik terimleri düzelt (`mergeleyemeyiz` → `merge edemeyiz`, `pr` → `PR`, repo, commit, branch, deploy).

**Yapma:** kelime seçimini değiştirme · cümle sırasını değiştirme · özetleme · kısaltma · bir şey ekleme · başlık, madde işareti, markdown ya da tırnak koyma · açıklama veya yorum yazma · anlamadıysan yorum yapma, girdiyi olduğu gibi döndür.

Close with: the reply is the corrected text and nothing else.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_dictation.py`, above the final `run(...)`.

The provider double already exists and needs no changes: `from test_publish import FakeAPI, completion`. `FakeAPI(routes)` takes a dict keyed by `(method, path_prefix)` whose values are lists of `(status, payload)` replies, serves on `127.0.0.1` on a free port, exposes `.url` and `.close()`, and **records every request in `.requests` as `{"method", "path", "body", "headers"}` with `body` already parsed from JSON**. So the prompt that was sent is `api.requests[0]["body"]["messages"][0]["content"]` — no callable reply is needed. `completion(text)` builds the response payload.

One local helper: `_config(base)` returning `resolve_config({}, fake_env(OPENAI_API_KEY="sk-test", OPENAI_API_BASE=base))`.

A provider failure is `FakeAPI` with a `(500, {})` reply; an unreachable provider is `_config("http://127.0.0.1:1")` with no server at all. Close every `FakeAPI` in a `finally`.

Fixtures, shared by the tests below:

```python
RAW = "eee bu pr'i bugun mergeleyemeyiz cunku yani testler gecmiyor hani o repodaki"
FIXED = "Bu PR'ı bugün merge edemeyiz, çünkü o repo'daki testler geçmiyor."
```

The three tests written out, because each pins a rule the spec states and prose cannot enforce:

```python
ROUTE = ("POST", "/v1/chat/completions")


def _bases():
    """Both ways the provider can fail: a 500, and nothing listening at all."""
    api = FakeAPI({ROUTE: [(500, {})]})
    yield api.url, api
    api.close()
    yield "http://127.0.0.1:1", None


def test_a_failing_provider_still_delivers_the_raw_transcript():
    # The user has already spoken and is waiting. Unpunctuated Turkish beats nothing.
    for base, api in _bases():
        try:
            clip, sink = FakeClipboard(), RecordingSink()
            d = dictation.deliver(RAW, _config(base), clip, sink)
            assert d.status == dictation.UNCORRECTED, f"{base}: status {d.status!r}"
            assert clip.text == RAW, f"{base}: clipboard {clip.text!r} != the raw transcript"
            assert clip.writes == 1, f"{base}: clipboard written {clip.writes} times"
            assert d.detail, f"{base}: no reason was recorded"
            assert any(k == "done" for k, _ in sink.calls), f"{base}: {sink.calls!r}"
        finally:
            if api:
                api.close()


def test_an_empty_transcript_leaves_the_clipboard_alone():
    for nothing in ("", "   ", "\n"):
        clip, sink = FakeClipboard(), RecordingSink()
        d = dictation.deliver(nothing, _config("http://127.0.0.1:1"), clip, sink)
        assert d.status == dictation.EMPTY, f"{nothing!r}: status {d.status!r}"
        assert clip.writes == 0, f"{nothing!r}: the clipboard was written to"
        assert clip.text == "ONCEKI ICERIK", f"{nothing!r}: clipboard {clip.text!r}"
        assert any(k == "failed" for k, _ in sink.calls), f"{nothing!r}: {sink.calls!r}"


def test_the_transcript_is_inlined_into_the_prompt():
    # A prompt file edited into a shape with no placeholder would append the
    # transcript instead, and the model would read the rules as the text.
    api = FakeAPI({ROUTE: [(200, completion(FIXED))]})
    try:
        dictation.deliver(RAW, _config(api.url), FakeClipboard(), RecordingSink())
    finally:
        api.close()
    sent = api.requests[0]["body"]["messages"][0]["content"]
    assert RAW in sent, "the transcript did not reach the prompt"
    assert not sent.rstrip().endswith(RAW), (
        "the transcript landed at the end — the placeholder was not found")
```

The remaining tests:

| Test | Asserts |
|---|---|
| `test_corrected_text_reaches_the_clipboard` | `completion(FIXED)` → `status == CORRECTED`, `clip.text == FIXED`, `clip.writes == 1`, `("done", FIXED)` in the sink's calls |
| `test_a_reply_that_grows_too_much_is_refused` | `completion(RAW * 5)` → `SKIPPED`, `clip.text == RAW` |
| `test_a_reply_that_shrinks_too_much_is_refused` | `completion("Testler geçmiyor.")` → `SKIPPED`, `clip.text == RAW` |
| `test_an_empty_reply_is_refused` | `completion("   ")` → `SKIPPED`, `clip.text == RAW` |
| `test_the_fixture_exercises_the_guards_middle` | `MIN_SHRINK < len(FIXED)/len(RAW) < MAX_GROWTH` — otherwise the happy-path test is silently testing a guard breach |
| `test_the_prompt_file_carries_a_placeholder` | `re.search(r"^\[[^\]\n]*\]\s*$", PROMPT_PATH.read_text(), re.MULTILINE)` matches |

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: the new tests FAIL with `AttributeError` on `dictation.deliver`; earlier tests still PASS.

- [ ] **Step 4: Implement `deliver`**

Order of operations, and the decision behind each:

1. Strip the transcript. Nothing left → `EMPTY`, `sink.failed(...)` saying no speech was detected, **return without touching the clipboard**.
2. Build the prompt with `publish.render_prompt(PROMPT_PATH, transcript)`.
3. Call `complete(prompt, cfg_dict)` where `cfg_dict` carries the three keys `publish.openai_complete` reads — `openai_base`, `openai_token`, `model`. `complete` is a parameter so the daemon's tests need no socket.
4. Any exception → `UNCORRECTED`: the **raw transcript** to the clipboard, the reason into `detail`, `sink.done(...)`. Send the exception to `audiocript._log_problem` so it outlives the notification.
5. Apply the length guard against the raw transcript: longer than `MAX_GROWTH ×`, shorter than `MIN_SHRINK ×`, or empty after stripping → `SKIPPED`, raw transcript to the clipboard. The prompt asks for a minimal correction; this is that contract enforced where prose cannot reach.
6. Otherwise `CORRECTED`: the reply to the clipboard.
7. Exactly one `clipboard.copy` on every path but `EMPTY`, which has none.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python tests/test_dictation.py
```

Expected: PASS for every test.

- [ ] **Step 6: Prove the guards are load-bearing**

Use the `mutation-gate` skill on three lines, marking each `# MUTANT-n` and restoring in the same turn:

1. The empty-transcript early return → `test_an_empty_transcript_leaves_the_clipboard_alone` must fail.
2. The `except` that falls back to the raw transcript → `test_a_failing_provider_still_delivers_the_raw_transcript` must fail.
3. The length guard → both refusal tests must fail.

- [ ] **Step 7: Commit**

```bash
git add dictation.py prompts/dictation_prompt.md tests/test_dictation.py
git commit -m "feat: correct a dictation and deliver it to the clipboard"
```

---

### Task 4: The state machine and the capture

**Files:**
- Create: `dictate.py`
- Test: `tests/test_dictate.py`

**Interfaces:**
- Consumes: Tasks 1–3; `audiocript._start_mic`, `audiocript._stream_source_to_wav`, `audiocript._resolve_mic_index`, `audiocript.transcribe_audio`
- Produces:

```python
IDLE, RECORDING, PROCESSING = "idle", "recording", "processing"

class MicCapture:
    """Owns one recording's temporary directory and microphone."""
    def __init__(self, cfg): ...       # cfg: the dict from audiocript.load_config()
    def start(self) -> None            # temp dir + audiocript._start_mic
    def stop(self) -> pathlib.Path     # stop the recorder, return a 16 kHz mono WAV
    def discard(self) -> None          # stop if running, delete the temp dir; never raises

class Daemon:
    def __init__(self, config, clipboard, sink, cfg=None,
                 capture_factory=MicCapture, transcribe=None, deliver=None): ...
    # capture_factory is called as capture_factory(cfg) and must return an object
    # with start() / stop() / discard(); transcribe as transcribe(wav_path, language);
    # deliver as deliver(transcript, config, clipboard, sink).
    state: str                         # IDLE | RECORDING | PROCESSING
    def toggle(self) -> None           # the ONLY way in; hotkey and signal both call it
    def join_worker(self, timeout=None) -> None    # for tests; not how the daemon sequences
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dictate.py`. Docstring: no test here opens a microphone, loads a model, or reaches a provider — the daemon takes its capture, transcription and delivery as parameters for exactly that reason.

It needs `from support import run, fake_env, FakeClipboard, RecordingSink`, plus `pathlib`, `threading`, `time`, `dictate` and `dictation`. **Import the doubles from `support`, never from `test_dictation`** — see the Global Constraints for why that would silently pass.

Two pieces of scaffolding the tests share:

```python
class FakeCapture:
    """Stands in for MicCapture. Records what the daemon asked it to do."""
    instances = []

    def __init__(self, cfg=None, wav="/tmp/never-read.wav", fail=None):
        self.events, self.wav, self.fail = [], pathlib.Path(wav), fail
        FakeCapture.instances.append(self)

    def start(self):
        self.events.append("start")
        if self.fail:
            raise self.fail

    def stop(self):
        self.events.append("stop")
        return self.wav

    def discard(self):
        self.events.append("discard")


def build(transcript="bir cumle", deliver=None, capture=None, max_seconds=300):
    """A daemon with every edge faked. Returns (daemon, clipboard, sink, delivered)."""
    FakeCapture.instances = []
    clip, sink, delivered = FakeClipboard(), RecordingSink(), []
    config = dictation.resolve_config(
        {"dictation_max_seconds": max_seconds},
        fake_env(OPENAI_API_KEY="sk-test", OPENAI_API_BASE="http://127.0.0.1:1"))

    def default_deliver(text, cfg, clipboard, status_sink, **kw):
        delivered.append(text)
        clipboard.copy(text)
        status_sink.done(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    daemon = dictate.Daemon(config, clip, sink, cfg={},
                            capture_factory=capture or FakeCapture,
                            transcribe=lambda wav, lang: transcript,
                            deliver=deliver or default_deliver)
    return daemon, clip, sink, delivered
```

The concurrency test, written out because a wrong version of it passes against a broken daemon:

```python
def test_a_toggle_while_processing_is_ignored():
    # A second dictation must not start on top of one still being corrected.
    gate = threading.Event()

    def slow_deliver(text, cfg, clipboard, status_sink, **kw):
        gate.wait(5)
        clipboard.copy(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    d, _, sink, _ = build(deliver=slow_deliver)
    d.toggle()
    d.toggle()
    deadline = time.time() + 5
    while d.state != dictate.PROCESSING and time.time() < deadline:
        time.sleep(0.01)
    assert d.state == dictate.PROCESSING, f"state {d.state!r}"
    before = len(FakeCapture.instances)
    d.toggle()
    assert d.state == dictate.PROCESSING, f"the toggle was not ignored: {d.state!r}"
    assert len(FakeCapture.instances) == before, "a second capture was started"
    assert any(k == "failed" for k, _ in sink.calls), (
        f"the user was not told it was still working: {sink.calls!r}")
    gate.set()
    d.join_worker(timeout=5)
```

The rest. Each drives `build(...)`, calls `toggle()` as described, and waits with `join_worker(timeout=5)` where a worker runs:

| Test | Drives | Asserts |
|---|---|---|
| `test_a_full_cycle_returns_to_idle` | toggle, toggle, join | `IDLE` → `RECORDING` → `IDLE`; `delivered == ["bir cumle"]`; `clip.text == "bir cumle"` |
| `test_the_sink_hears_recording_then_processing` | toggle, toggle, join | first sink call is `recording`; `recording` precedes `processing` |
| `test_a_microphone_that_cannot_open_returns_to_idle` | `capture=` a factory returning `FakeCapture(fail=OSError("device refused"))`; one toggle | state stays `IDLE`; a `failed` call reached the sink; `clip.writes == 0`; `"discard"` in the capture's events |
| `test_a_crashing_worker_returns_to_idle` | `deliver=` one that raises `RuntimeError`; toggle, toggle, join | state is `IDLE`; a `failed` call reached the sink |
| `test_the_capture_is_always_discarded` | toggle, toggle, join | `"discard"` in the last capture's events |
| `test_the_duration_bound_stops_the_recording_by_itself` | `max_seconds=1`; one toggle; poll for `IDLE` up to 10s | reaches `IDLE` on its own; something was delivered; a sink message contains `sınır` |

End with the same guarded runner Task 1 used:

```python
if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python tests/test_dictate.py
```

Expected: FAIL on `import dictate`.

- [ ] **Step 3: Implement `MicCapture`**

Module docstring: this module holds the threads and the state; `dictation.py` holds what can be tested without them.

- `start` makes a directory with `tempfile.mkdtemp`, then opens the microphone through `audiocript._start_mic(shim, raw_path, announce)`. The shim exposes only `.mic_index` — from `audiocript._resolve_mic_index(cfg)` — and `.cfg`; that is all `_start_mic` reads, and building a `_TuiState` would drag the whole TUI into the daemon. Going through `_start_mic` rather than `DeviceRecorder` directly is deliberate: its retry loop exists because refused microphone opens cost two real recordings.
- `announce` routes to `audiocript._log_problem` or drops the message — the daemon has no status line.
- `stop` stops the recorder, then converts the raw capture to a 16 kHz mono WAV inside the temporary directory. **`_stream_source_to_wav` does not take a path** — its signature is `_stream_source_to_wav(reader, rate, out_path, target_fs=16000, on_progress=None)`, and the reader is an `audiocript._RawPcmReader(path, dtype, channels)` used as a context manager (`__len__`, `__enter__`, `__exit__`, `read(start, stop)`). The chain is: build the reader from the raw file with `DeviceRecorder.RAW_DTYPE` and the recorder's `.channels`, enter it, and pass it with the recorder's native `.rate`. `DeviceRecorder.manifest()` already returns exactly `{"kind", "file", "rate", "channels", "dtype"}`, so read the three values from there rather than assembling them by hand. Return the WAV's path.
- `discard` stops a running recorder and removes the directory with `shutil.rmtree(..., ignore_errors=True)`. **It must never raise** — every failure path calls it.

- [ ] **Step 4: Implement `Daemon`**

- One `threading.Lock` guards `state`. `toggle` is the only public way in, so the hotkey thread and the signal handler cannot interleave.
- `IDLE`: build a capture via `capture_factory`, `start()`, set `RECORDING`, `sink.recording()`. If `start()` raises → `discard()`, `sink.failed(reason)`, stay `IDLE`, do not touch the clipboard.
- `RECORDING`: `stop()`, set `PROCESSING`, `sink.processing()`, hand the WAV to a worker thread, cancel the duration timer.
- `PROCESSING`: `sink.failed(...)` saying it is still working, and return. Nothing else.
- On entering `RECORDING`, start a `threading.Timer(config.max_seconds, self.toggle)`. Its notification must say the limit was reached rather than implying the user stopped it — one test asserts the Turkish word `sınır`, so keep it. Cancel the timer on every route out of `RECORDING`.
- The worker calls `transcribe(wav, config.language)` then `deliver(transcript, config, clipboard, sink)`. Defaults are `audiocript.transcribe_audio` and `dictation.deliver`; both are parameters so tests need neither a model nor a socket.
- **The worker's body is wrapped in `try`/`finally`.** The `finally` discards the capture and returns the state to `IDLE` whatever happened; an exception inside reaches `sink.failed` and `audiocript._log_problem`. A daemon stuck in `PROCESSING` is dead with no way for the user to tell — two tests pin this.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python tests/test_dictate.py
```

Expected: PASS for every test.

- [ ] **Step 6: Prove the state machine's guards are load-bearing**

Use the `mutation-gate` skill, marking each `# MUTANT-n` and restoring in the same turn:

1. Remove `toggle`'s `PROCESSING` branch → `test_a_toggle_while_processing_is_ignored` must fail.
2. Remove the worker's `finally` → `test_a_crashing_worker_returns_to_idle` and `test_the_capture_is_always_discarded` must fail.
3. Remove the timer → `test_the_duration_bound_stops_the_recording_by_itself` must fail.

- [ ] **Step 7: Commit**

```bash
git add dictate.py tests/test_dictate.py
git commit -m "feat: drive one dictation through a state machine"
```

---

### Task 5: Hotkeys, permission detection, lifecycle and CLI

**Files:**
- Modify: `dictate.py`
- Test: `tests/test_dictate.py`

**Interfaces:**
- Consumes: Task 4's `Daemon`; `pynput.keyboard.GlobalHotKeys`
- Produces:

```python
PID_PATH = pathlib.Path("~/.audiocript/dictate.pid").expanduser()

def build_listener(config, on_toggle, hotkeys_class=keyboard.GlobalHotKeys)
    # builds {combination: callback} from config.hotkeys; returns it unstarted

def check_listener(listener) -> str | None
    # None when input monitoring is granted and the tap is up; otherwise the warning

def write_pid(path=PID_PATH) -> None        # raises when a live daemon already owns it
def read_pid(path=PID_PATH) -> int | None   # None when absent, unparseable or stale
def signal_daemon(action, path=PID_PATH) -> int   # "toggle" | "stop"; a process exit status
def main(argv=None) -> int                  # the CLI; returns a process exit status
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dictate.py`, above the final `run(...)`. Two doubles:

```python
class FakeListener:
    def __init__(self, trusted=True, alive=True):
        self.IS_TRUSTED, self._alive = trusted, alive

    def wait(self): pass
    def is_alive(self): return self._alive


class SpyHotKeys:
    """Captures the mapping build_listener registered, without creating an event tap."""
    registered = {}

    def __init__(self, mapping):
        SpyHotKeys.registered = dict(mapping)
        self.IS_TRUSTED = True

    def start(self): pass
    def wait(self): pass
    def is_alive(self): return True
    def stop(self): pass
```

| Test | Asserts |
|---|---|
| `test_a_healthy_listener_raises_no_warning` | `check_listener(FakeListener()) is None` |
| `test_a_listener_without_permission_warns_and_names_the_fallback` | `check_listener(FakeListener(trusted=False))` returns a message containing `"Input Monitoring"` **and** `"--toggle"`. Without this the daemon looks like it is working while the hotkey silently does nothing — indistinguishable from a broken install |
| `test_a_dead_listener_warns_too` | `check_listener(FakeListener(alive=False))` is truthy. `_create_event_tap` returning `None` marks the listener ready and then exits its thread, so `IS_TRUSTED` alone does not cover every failure |
| `test_the_configured_hotkey_is_the_one_registered` | with `{"dictation_hotkeys": {"toggle": "<cmd>+<shift>+<alt>+k"}}`, that combination is a key of `SpyHotKeys.registered` |
| `test_the_default_hotkey_is_registered_when_config_is_silent` | with `{}`, `DEFAULT_HOTKEYS["toggle"]` is a key of `SpyHotKeys.registered` |

The pid-file tests each run inside `workdir("pid")` — add it to the file's `support` import — against `d / "dictate.pid"`, and need `os`:

| Test | Asserts |
|---|---|
| `test_the_pid_file_round_trips` | after `write_pid(path)`, `read_pid(path) == os.getpid()` |
| `test_a_stale_pid_file_reads_as_absent_and_does_not_block` | a file holding `"999999"` → `read_pid` is `None`, and `write_pid` succeeds and then reads back as this process |
| `test_a_live_pid_file_blocks_a_second_daemon` | after `write_pid`, a second `write_pid` on the same path raises |
| `test_toggle_with_no_daemon_exits_non_zero` | `signal_daemon("toggle", d / "absent.pid") != 0` — exiting zero would make a Shortcuts binding look like it worked |

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python tests/test_dictate.py
```

Expected: the new tests FAIL with `AttributeError` on `dictate.check_listener` and friends; Task 4's tests still PASS.

- [ ] **Step 3: Implement the listener and the permission check**

- `build_listener` maps each of `config.hotkeys` to its callback and returns the constructed listener **without starting it**. `hotkeys_class` is a parameter purely so the tests above can watch what was registered; no real event tap is created in a test.
- `check_listener` is called after `start()` and `wait()`. It returns `None` when `IS_TRUSTED` and `is_alive()` are both true, otherwise the warning. Both signals are needed: `IS_TRUSTED` comes from `AXIsProcessTrusted()` at the top of the darwin backend's `_run`, while a tap that fails for another reason marks itself ready and exits its thread.
- The warning names **Input Monitoring**, says where to grant it (System Settings → Privacy & Security), and says `dictate.py --toggle` works without it. A user who cannot grant the permission still has a working feature; a user who is not told has neither.

- [ ] **Step 4: Implement the lifecycle and the CLI**

- `write_pid` refuses when the file names a live process and overwrites when it does not. `read_pid` returns `None` for absent, unparseable or stale. Probe with `os.kill(pid, 0)`: `ProcessLookupError` means stale, `PermissionError` means alive.
- `signal_daemon` sends `SIGUSR1` for `"toggle"`, `SIGTERM` for `"stop"`. With no live daemon it prints that and returns non-zero.
- `main(argv)` handles no arguments (run the daemon), `--toggle`, `--stop`. The daemon path:
  1. `audiocript.load_config()`, then `dictation.resolve_config`. A `ConfigError` prints its message — which already names the offending value — and returns non-zero.
  2. `write_pid`; a refusal prints why and returns non-zero.
  3. `audiocript._ensure_model(config.language)`, so the first dictation is not the one paying for the model load. Say so on the way in; it takes a while.
  4. Install `SIGUSR1` → `daemon.toggle`, and `SIGTERM`/`SIGINT` → shutdown. `signal.signal` only works on the main thread, which is where `main` runs.
  5. `build_listener`, `start()`, `wait()`, `check_listener` — print the warning and **keep running**. The signal path still works, so an unusable hotkey is not a reason to exit.
  6. Block the main thread until shutdown, otherwise idle: the spec reserves it for the menu bar indicator.
  7. On the way out: stop the listener, discard any capture in flight, remove the pid file.
- `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python tests/test_dictate.py && .venv/bin/python tests/test_dictation.py
```

Expected: PASS for every test in both files.

- [ ] **Step 6: Prove the permission warning is load-bearing**

Use the `mutation-gate` skill: delete the `IS_TRUSTED` half of `check_listener`, mark `# MUTANT-1`, confirm `test_a_listener_without_permission_warns_and_names_the_fallback` fails, restore. Repeat for the `is_alive` half against `test_a_dead_listener_warns_too`.

- [ ] **Step 7: Commit**

```bash
git add dictate.py tests/test_dictate.py
git commit -m "feat: bind the dictation hotkey and manage the daemon's lifecycle"
```

---

### Task 6: Launcher and documentation

**Files:** Modify `run.sh`, `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Add the `--dictate` switch to `run.sh`**

Read `run.sh` first and follow its structure. The switch reuses the dependency and virtualenv setup the script already performs, then runs `dictate.py` instead of `audiocript.py` as the final step. Do not duplicate the setup.

**The trap:** the script's last line is `exec "$VPY" audiocript.py "$@"`, so an unhandled `--dictate` is forwarded to `audiocript.py` — which reads no arguments at all and would launch the TUI as if nothing had been asked for. Consume the switch before that line: read it off `$1`, `shift`, and swap the script the `exec` targets, leaving the remaining arguments to pass through as they do today.

- [ ] **Step 2: Verify the switch reaches the daemon**

```bash
./run.sh --dictate
```

Expected: the usual setup, then the daemon's startup output. `Ctrl-C` exits cleanly; then:

```bash
ls ~/.audiocript/dictate.pid
```

Expected: no such file.

- [ ] **Step 3: Document the feature in `README.md`**

A section shaped like the existing "Publishing transcripts (optional, off by default)", plus one line in the feature list. Cover: the hotkey and what it does; that the language follows the existing Language setting and is one or the other; the `config.json` keys and their defaults (`dictation_hotkeys`, `dictation_max_seconds`); `DICTATION_MODEL`; the Input Monitoring permission and where to grant it; the `--toggle` fallback for macOS Shortcuts or `skhd`; and that nothing is saved.

**The privacy point is not optional.** The README carries a "runs offline" badge earned by local transcription. Dictation sends every utterance to an AI provider. Say so plainly here, the way the publishing section already declares its two paid calls. A reader who trusts the badge and dictates a password was misled by our documentation.

- [ ] **Step 4: Add a `CHANGELOG.md` entry**

Follow the file's existing format exactly: the feature, the new `config.json` keys, the new environment variable, the permission requirement.

- [ ] **Step 5: Commit**

```bash
git add run.sh README.md CHANGELOG.md
git commit -m "docs: explain dictation and add the run.sh switch"
```

---

### Task 7: End-to-end verification

**Files:** none — this task produces evidence, not code.

The spec's remaining unverified assumptions are settled here. Composition errors are what task-scoped tests structurally cannot see, so this is neither optional nor a formality.

- [ ] **Step 1: Confirm the whole suite is green**

```bash
.venv/bin/python tests/run_all.py
```

Expected: no failures. A pre-existing failure is reported as pre-existing, with its output — not quietly absorbed.

- [ ] **Step 2: Settle the Input Monitoring question**

Start the daemon with `./run.sh --dictate` and record which happened:

- the hotkey works outright;
- macOS prompted for Input Monitoring and it worked after granting — note **which binary** the permission attached to, since that is what the spec could not predict;
- it could not be granted and `check_listener` printed the warning.

Replace that assumption in the spec with what was observed. In the third case, verify `--toggle` end to end and say in the README that it is the supported path on this setup.

- [ ] **Step 3: Dictate for real, in another application**

With the daemon running, focus Slack or any text field. Press the hotkey, say a sentence with fillers and a technical term, press it again, wait for the notification, paste.

Check against the spec's acceptance criteria: the text is punctuated; it says what you said, with no summary and no added sentences; and you were told when recording started, when work began, and when the clipboard was ready.

- [ ] **Step 4: Measure the latency**

Time the gap between the second keypress and the notification, for a short utterance and a longer one. Record the numbers in the spec, replacing that assumption. If it is too slow to feel like dictation, say so with the measurements — the model choice and the warm-up strategy are where to look, and that is a separate change.

- [ ] **Step 5: Exercise the failure paths by hand**

- **Provider unreachable:** start with `OPENAI_API_BASE=http://127.0.0.1:1`, dictate, confirm the clipboard holds the raw transcript and the notification says it was not corrected.
- **Silence:** put a known string on the clipboard, dictate without speaking, confirm the string is still there.
- **The duration bound:** set `dictation_max_seconds` small, start a recording, do not stop it, confirm it stops itself and delivers what it captured.
- **A bad hotkey:** put `"cmd+alt+d"` in `config.json`, confirm the daemon refuses to start and names the value.
- **A reconfigured hotkey:** set `dictation_hotkeys.toggle` to a different combination, restart, and confirm the new one drives a dictation and the old one does nothing. This is the whole point of the key being in `config.json`, and only the registration is covered by a test.

- [ ] **Step 6: Answer the shared-microphone question**

Start the TUI, begin a recording there, and press the dictation hotkey. Record what happened. The spec assumes `_start_mic`'s retry handles a refusal and no lock is needed; if that is wrong, write down what actually happened and raise a lock as a separate change rather than patching one in here.

- [ ] **Step 7: Confirm nothing was left behind**

```bash
ls ~/.audiocript/ ; ls /tmp | grep -i -E 'audiocript|dictat' ; git status --short
```

Expected: no pid file once the daemon has exited, no temporary directories, and a working tree holding only the intended changes.

- [ ] **Step 8: Commit the settled assumptions**

```bash
git add docs/superpowers/specs/2026-08-27-dictation-daemon-design.md
git commit -m "docs: record what dictation's open questions turned out to be"
```

---

## Notes for the executor

- **Re-resolve every coordinate.** This plan cites symbols, not line numbers, because line numbers drift. `grep` for a symbol before calling it; if it is not what this plan describes, stop and say so rather than adapting silently.
- **Report what happened.** A failing test is reported with its output. A skipped step is named. A blocked step does not stop the rest of the plan.
- **Every mutation is marked and reverted in the same turn.** A `# MUTANT-n` left behind reads as real code, and recovery has only ever worked when the mutation was marked.

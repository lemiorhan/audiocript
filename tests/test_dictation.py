"""dictation: the configuration, the output sinks and the correction pipeline.

Nothing here may reach a real API or read a developer's .env. Every test but
one passes an explicit `env`, so publish.env_value is never consulted.
test_env_defaults_to_publish_env_value is the exception: it monkeypatches
publish.env_value itself, so resolve_config's fallback to it is pinned without
ever touching a real .env. Every provider call goes to a FakeAPI on localhost or
to an address with nothing listening, so no test can spend a real API key.
"""
import os
import re
import tempfile
from pathlib import Path

from support import A, run, fake_env, FakeClipboard, RecordingSink
from test_publish import FakeAPI, completion

import dictation
import publish

BASE_ENV = dict(OPENAI_API_KEY="sk-test")


def _refuses(cfg, env, because):
    """resolve_config must raise ConfigError, and the message must name the value."""
    try:
        dictation.resolve_config(cfg, env)
    except dictation.ConfigError as e:
        assert because in str(e), f"message {str(e)!r} does not name {because!r}"
        return
    raise AssertionError(f"accepted {cfg!r}; expected ConfigError naming {because!r}")


def test_defaults_when_config_is_empty():
    cfg = dictation.resolve_config({}, fake_env(**BASE_ENV))
    print(f"  {cfg}")
    assert cfg.language == "tr"
    assert cfg.hotkeys == dictation.DEFAULT_HOTKEYS
    assert cfg.max_seconds == dictation.DEFAULT_MAX_SECONDS
    assert cfg.model == dictation.DEFAULT_MODEL
    assert cfg.openai_base == "https://api.openai.com"


def test_config_values_are_used():
    given = {"language": "en",
             "dictation_hotkeys": {"toggle": "<cmd>+<shift>+<alt>+k"},
             "dictation_max_seconds": 42}
    cfg = dictation.resolve_config(given, fake_env(**BASE_ENV))
    print(f"  {cfg}")
    assert cfg.language == "en"
    assert cfg.hotkeys == {"toggle": "<cmd>+<shift>+<alt>+k"}
    assert cfg.max_seconds == 42


def test_env_overrides_model_and_base():
    env = fake_env(OPENAI_API_KEY="sk-test", DICTATION_MODEL="gpt-9-mini",
                   OPENAI_API_BASE="https://proxy.example")
    cfg = dictation.resolve_config({}, env)
    print(f"  model={cfg.model} openai_base={cfg.openai_base}")
    assert cfg.model == "gpt-9-mini"
    assert cfg.openai_base == "https://proxy.example"


def test_a_partial_hotkeys_object_keeps_the_defaults():
    cfg = dictation.resolve_config({"dictation_hotkeys": {}}, fake_env(**BASE_ENV))
    print(f"  hotkeys={cfg.hotkeys}")
    assert cfg.hotkeys == dictation.DEFAULT_HOTKEYS


def test_every_known_language_is_accepted():
    for code in A.LANGUAGES:
        cfg = dictation.resolve_config({"language": code}, fake_env(**BASE_ENV))
        assert cfg.language == code
    print(f"  accepted {sorted(A.LANGUAGES)}")


def test_missing_api_key_is_refused():
    _refuses({}, fake_env(), "OPENAI_API_KEY")
    print("  refused with no OPENAI_API_KEY")


def test_unknown_language_is_refused():
    _refuses({"language": "de"}, fake_env(**BASE_ENV), "de")
    print("  refused unknown language")


def test_unparseable_hotkey_is_refused():
    for bad in ("cmd+alt+d", "<zort>+d"):
        cfg = {"dictation_hotkeys": {"toggle": bad}}
        _refuses(cfg, fake_env(**BASE_ENV), bad)

    # "" in message is vacuously true for any message, so the empty-string case
    # needs a real assertion: the message must be non-empty and must name either
    # the action or the expected syntax.
    try:
        dictation.resolve_config({"dictation_hotkeys": {"toggle": ""}},
                                 fake_env(**BASE_ENV))
    except dictation.ConfigError as e:
        message = str(e)
        assert message, "an empty hotkey produced an empty message"
        assert "toggle" in message or "key combination" in message, \
            f"message {message!r} names neither the action nor the expected syntax"
    else:
        raise AssertionError("accepted an empty hotkey combination")
    print("  refused 3 unparseable combinations")


def test_hotkey_without_a_modifier_is_refused():
    _refuses({"dictation_hotkeys": {"toggle": "d"}}, fake_env(**BASE_ENV), "d")
    print("  refused a bare key with no modifier")


def test_unknown_hotkey_action_is_refused():
    cfg = {"dictation_hotkeys": {"togle": "<cmd>+<alt>+d"}}
    _refuses(cfg, fake_env(**BASE_ENV), "togle")
    print("  refused an unknown hotkey action")


def test_bad_max_seconds_is_refused():
    _refuses({"dictation_max_seconds": "soon"}, fake_env(**BASE_ENV), "soon")
    _refuses({"dictation_max_seconds": 0}, fake_env(**BASE_ENV), "0")
    print("  refused a non-integer and a non-positive max_seconds")


def test_env_defaults_to_publish_env_value():
    """resolve_config's `env` parameter defaults to publish.env_value. Pinned by
    monkeypatching that function rather than touching a real .env, so dropping
    or inverting `env = env or publish.env_value` in dictation.py would be
    caught here even though every other test passes an explicit `env`."""
    saved = publish.env_value
    publish.env_value = fake_env(OPENAI_API_KEY="sk-from-publish")
    try:
        cfg = dictation.resolve_config({})
        assert cfg.openai_token == "sk-from-publish", \
            "resolve_config's default did not consult publish.env_value"
        print("  default env consulted publish.env_value")
    finally:
        publish.env_value = saved


def test_openai_token_does_not_appear_in_the_rendered_config():
    """A traceback, a log line, or a careless print/f-string must not be able to
    reconstruct the token from a rendered DictationConfig — so a resolved config
    carrying a recognisable sentinel must not show it in repr() or str()."""
    cfg = dictation.resolve_config({}, fake_env(OPENAI_API_KEY="sk-SECRET-CANARY"))
    assert "SECRET-CANARY" not in repr(cfg), repr(cfg)
    assert "SECRET-CANARY" not in str(cfg), str(cfg)
    assert cfg.openai_token == "sk-SECRET-CANARY", "the field itself must still work"
    print("  sentinel absent from repr() and str(); still readable as a field")


def test_it_resolves_where_publish_config_would_not():
    """dictation must not inherit publish's requirement for GitHub credentials."""
    saved_root = publish.ROOT
    saved = {k: os.environ.pop(k, None) for k in
             ("GITHUB_REPO_FOR_TRANSCRIPTS", "GITHUB_TOKEN", "OPENAI_API_KEY")}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            publish.ROOT = Path(tmp)
            os.environ["OPENAI_API_KEY"] = "sk-test"
            assert publish.config() is None, \
                "publish.config() configured itself without GitHub credentials"
            cfg = dictation.resolve_config({}, fake_env(**BASE_ENV))
            print(f"  publish.config()=None, dictation model={cfg.model}")
            assert cfg.model == dictation.DEFAULT_MODEL
    finally:
        publish.ROOT = saved_root
        os.environ.pop("OPENAI_API_KEY", None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_clipboard_round_trips_turkish():
    """pbcopy is real here — the encoding is the thing under test — so the
    developer's actual clipboard is saved and restored around it."""
    import subprocess
    before = subprocess.run(["pbpaste"], capture_output=True).stdout
    try:
        text = "Şu PR'ı bugün merge edemeyiz — çünkü İĞÜÖÇ testleri geçmiyor."
        dictation.Clipboard().copy(text)
        back = subprocess.run(["pbpaste"], capture_output=True).stdout.decode("utf-8")
        assert back == text, f"clipboard {back!r} != {text!r}"
    finally:
        subprocess.run(["pbcopy"], input=before)
    print("  round-tripped Turkish text through the real clipboard")


def test_notify_sink_reports_each_stage():
    """recording() and processing() are the start and the stop of speaking —
    each must cue on its own, so the cue count is checked right after each
    call rather than once at the end (where one missing cue would hide behind
    the other)."""
    notified, sounds = [], []
    sink = dictation.NotifySink(notify=notified.append,
                                sound=lambda: sounds.append(None))
    sink.recording()
    assert len(sounds) == 1, f"recording() did not cue: {sounds}"
    sink.processing()
    assert len(sounds) == 2, f"processing() did not cue: {sounds}"
    sink.done("Bir cümle.")
    sink.failed("mikrofon açılamadı")
    assert len(sounds) == 2, f"done/failed must not cue: {sounds}"
    assert len(notified) == 4, notified
    assert any("Bir cümle." in m for m in notified), notified
    assert any("mikrofon açılamadı" in m for m in notified), notified
    print(f"  notified={notified}  sounds={len(sounds)}")


def test_notify_sink_truncates_a_long_preview():
    notified = []
    sink = dictation.NotifySink(notify=notified.append, sound=lambda: None)
    sink.done("x" * 500)
    assert len(notified) == 1, notified
    message = notified[0]
    assert len(message) < 200, f"message is {len(message)} chars: {message!r}"
    print(f"  message length={len(message)}")


def test_notify_sink_survives_a_failing_notifier():
    """A broken osascript must not cost the user a dictation whose text is
    already on the clipboard: notify raising must not propagate. The swallowed
    exception must not be lost entirely either — it goes to
    audiocript._log_problem, pinned here by monkeypatching it."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        def broken_notify(message):
            raise OSError("osascript is not available")
        sink = dictation.NotifySink(notify=broken_notify, sound=lambda: None)
        sink.done("Bir cümle.")
        sink.failed("mikrofon açılamadı")
    finally:
        A._log_problem = saved
    assert len(logged) == 2, logged
    print(f"  survived a failing notifier; logged {len(logged)} problems")


def test_default_notify_and_sound_pass_a_timeout():
    """NotifySink's try/except only catches a raise, not a hang. A stuck
    osascript or afplay (no timeout=) would block the caller indefinitely —
    Task 4's state machine, on the hotkey thread — so the defaults must bound
    the subprocess themselves. Pinned by stubbing subprocess.run and reading
    back the kwargs it was called with."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)

    saved = dictation.subprocess.run
    dictation.subprocess.run = fake_run
    try:
        dictation._osascript_notify("hello")
        dictation._afplay_sound()
    finally:
        dictation.subprocess.run = saved

    assert len(calls) == 2, calls
    for kwargs in calls:
        assert "timeout" in kwargs, kwargs
        assert isinstance(kwargs["timeout"], (int, float)) and kwargs["timeout"] > 0, \
            f"timeout must be a positive number: {kwargs}"
    print(f"  timeouts={[c['timeout'] for c in calls]}")


# =========================== the correction pipeline ===========================

# What Whisper hands over: no punctuation, no Turkish letters, fillers in the
# middle of it — and what one minimal correction of it looks like.
RAW = "eee bu pr'i bugun mergeleyemeyiz cunku yani testler gecmiyor hani o repodaki"
FIXED = "Bu PR'ı bugün merge edemeyiz, çünkü o repo'daki testler geçmiyor."

ROUTE = ("POST", "/v1/chat/completions")


def _config(base):
    return dictation.resolve_config(
        {}, fake_env(OPENAI_API_KEY="sk-test", OPENAI_API_BASE=base))


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
    print("  a 500 and an unreachable provider both delivered the raw transcript")


def test_an_empty_transcript_leaves_the_clipboard_alone():
    for nothing in ("", "   ", "\n"):
        clip, sink = FakeClipboard(), RecordingSink()
        d = dictation.deliver(nothing, _config("http://127.0.0.1:1"), clip, sink)
        assert d.status == dictation.EMPTY, f"{nothing!r}: status {d.status!r}"
        assert clip.writes == 0, f"{nothing!r}: the clipboard was written to"
        assert clip.text == "ONCEKI ICERIK", f"{nothing!r}: clipboard {clip.text!r}"
        assert any(k == "failed" for k, _ in sink.calls), f"{nothing!r}: {sink.calls!r}"
    print("  3 empty transcripts left the clipboard untouched")


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
    print(f"  the transcript was inlined into {len(sent)} chars of prompt")


def test_corrected_text_reaches_the_clipboard():
    api = FakeAPI({ROUTE: [(200, completion(FIXED))]})
    clip, sink = FakeClipboard(), RecordingSink()
    try:
        d = dictation.deliver(RAW, _config(api.url), clip, sink)
    finally:
        api.close()
    assert d.status == dictation.CORRECTED, f"status {d.status!r} ({d.detail})"
    assert d.text == FIXED, f"delivered {d.text!r}"
    assert clip.text == FIXED, f"clipboard {clip.text!r}"
    assert clip.writes == 1, f"clipboard written {clip.writes} times"
    assert ("done", FIXED) in sink.calls, sink.calls
    assert api.requests[0]["body"]["model"] == dictation.DEFAULT_MODEL, \
        api.requests[0]["body"]["model"]
    print(f"  {FIXED!r} reached the clipboard")


def _refused(reply, complete=None):
    """deliver must refuse `reply` and put the raw transcript on the clipboard."""
    clip, sink = FakeClipboard(), RecordingSink()
    api = None
    try:
        if complete is None:
            api = FakeAPI({ROUTE: [(200, completion(reply))]})
            base = api.url
        else:
            base = "http://127.0.0.1:1"      # never reached: complete is injected
        d = dictation.deliver(RAW, _config(base), clip, sink, complete=complete)
    finally:
        if api:
            api.close()
    assert d.status == dictation.SKIPPED, \
        f"status {d.status!r} for a {len(reply)}-char reply"
    assert clip.text == RAW, f"clipboard {clip.text!r} != the raw transcript"
    assert clip.writes == 1, f"clipboard written {clip.writes} times"
    assert d.detail, "no reason was recorded"
    return d


def test_a_reply_that_grows_too_much_is_refused():
    d = _refused(RAW * 5)
    print(f"  refused a 5x reply: {d.detail}")


def test_a_reply_that_shrinks_too_much_is_refused():
    d = _refused("Testler geçmiyor.")
    print(f"  refused a summary: {d.detail}")


def test_an_empty_reply_is_refused():
    """The reply is injected rather than served: publish.openai_complete turns an
    empty message into a RuntimeError of its own, so a FakeAPI serving one would
    exercise the provider-failure path instead of deliver's length guard."""
    d = _refused("   ", complete=lambda prompt, cfg: "   ")
    print(f"  refused a blank reply: {d.detail}")


def test_a_provider_failure_is_logged():
    """A notification is gone the moment the user dismisses it, and the dictation
    still succeeded from the user's side — so the reason the correction was lost
    has to outlive it. Pinned by monkeypatching audiocript._log_problem."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        def broken(prompt, cfg_dict):
            raise RuntimeError("the provider is down")
        d = dictation.deliver(RAW, _config("http://127.0.0.1:1"), FakeClipboard(),
                              RecordingSink(), complete=broken)
    finally:
        A._log_problem = saved
    assert d.status == dictation.UNCORRECTED, f"status {d.status!r}"
    assert len(logged) == 1, f"logged {logged!r}"
    what, exc = logged[0]
    assert "the provider is down" in str(exc), f"the exception was lost: {logged!r}"
    assert what, "nothing was said about what failed"
    print(f"  logged {what!r} with the exception")


def test_the_api_key_reaches_nothing_but_the_provider():
    """The accident this guards actually happened once: a config rendered into a
    message leaked a real key. The key must reach the provider call and nowhere
    else — not the prompt, the Delivery, the clipboard, the sink or the log."""
    canary = "sk-SECRET-CANARY"
    cfg = dictation.resolve_config(
        {}, fake_env(OPENAI_API_KEY=canary, OPENAI_API_BASE="http://127.0.0.1:1"))
    seen, logged = [], []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append(f"{what} {exc}")
    try:
        def complete(prompt, cfg_dict):
            seen.append((prompt, cfg_dict["openai_token"]))
            raise RuntimeError("no provider today")
        clip, sink = FakeClipboard(), RecordingSink()
        d = dictation.deliver(RAW, cfg, clip, sink, complete=complete)
    finally:
        A._log_problem = saved
    assert len(seen) == 1, f"complete was called {len(seen)} times with a token"
    prompt, token = seen[0]
    assert token == canary, "the provider call did not get the key it needs"
    for where, text in (("the prompt", prompt), ("the detail", d.detail),
                        ("the delivered text", d.text), ("the clipboard", clip.text),
                        ("the sink", repr(sink.calls)), ("the log", " ".join(logged))):
        assert "SECRET-CANARY" not in text, f"the API key reached {where}: {text!r}"
    print("  the key reached the provider call and nothing else")


def test_the_fixture_exercises_the_guards_middle():
    """Otherwise test_corrected_text_reaches_the_clipboard would be silently
    asserting on a reply the length guard should have refused."""
    ratio = len(FIXED) / len(RAW)
    assert dictation.MIN_SHRINK < ratio < dictation.MAX_GROWTH, \
        f"the happy-path fixture is at {ratio:.2f}x, outside the guard's bounds"
    print(f"  the fixture pair is {ratio:.2f}x, inside "
          f"[{dictation.MIN_SHRINK}, {dictation.MAX_GROWTH}]")


def test_the_prompt_file_carries_a_placeholder():
    """publish.render_prompt appends the transcript to the end of a prompt with no
    bracketed trailing line, so losing the placeholder silently changes what the
    model is asked."""
    text = dictation.PROMPT_PATH.read_text(encoding="utf-8")
    assert re.search(r"^\[[^\]\n]*\]\s*$", text, re.MULTILINE), \
        f"{dictation.PROMPT_PATH} has no bracketed placeholder line"
    print(f"  {dictation.PROMPT_PATH.name} carries a placeholder")


def test_the_prompt_path_does_not_depend_on_the_working_directory():
    """The daemon runs from wherever the user launched it, so a cwd-relative
    prompt path would work in the repo and fail everywhere else."""
    saved = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        assert dictation.PROMPT_PATH.is_file(), \
            f"{dictation.PROMPT_PATH} is not readable from {os.getcwd()}"
        body = publish.render_prompt(dictation.PROMPT_PATH, RAW)
    finally:
        os.chdir(saved)
    assert RAW in body, "the prompt did not render from another directory"
    print(f"  the prompt rendered from {tempfile.gettempdir()}")


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())

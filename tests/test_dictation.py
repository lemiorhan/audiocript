"""dictation: the configuration, the output sinks and the correction pipeline.

Nothing here may reach a real API or read a developer's .env. Every test but
one passes an explicit `env`, so publish.env_value is never consulted.
test_env_defaults_to_publish_env_value is the exception: it monkeypatches
publish.env_value itself, so resolve_config's fallback to it is pinned without
ever touching a real .env. Every provider call goes to a FakeAPI on localhost or
to an address with nothing listening, so no test can spend a real API key.
"""
import inspect
import json
import os
import re
import tempfile
import threading
from pathlib import Path

from support import A, run, fake_env, FakeClipboard, RecordingSink, workdir
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


# One sample per parameter name the contract uses. A moment whose parameter has no
# sample here fails _drive_every_moment loudly rather than being skipped, which is
# what keeps the scans below honest as the contract grows.
MOMENT_SAMPLES = {"text": "Bir cümle.", "reason": "mikrofon açılamadı",
                  "power": "on"}


def _drive_every_moment(sink):
    """Call every moment StatusSink declares on `sink`, and return their names.

    Discovered by scanning the contract, never by listing it. Enumerating the moments
    is how a test keeps passing while a newly added one goes unimplemented and
    unguarded — which is exactly what happened when the fifth, power_changed, was
    added to a contract whose tests named done() and failed()."""
    called = []
    for name, declared in vars(dictation.StatusSink).items():
        if name.startswith("_") or not callable(declared):
            continue
        args = []
        for parameter in list(inspect.signature(declared).parameters.values())[1:]:
            if parameter.default is not inspect.Parameter.empty:
                continue                      # optional: the default is the contract
            assert parameter.name in MOMENT_SAMPLES, (
                f"{name}() requires {parameter.name!r} and this file has no sample "
                "for it, so the moment would go untested — add one to MOMENT_SAMPLES")
            args.append(MOMENT_SAMPLES[parameter.name])
        getattr(sink, name)(*args)
        called.append(name)
    assert called, "no moments were discovered on StatusSink"
    return sorted(called)


def test_the_sink_contract_is_fully_implemented():
    """StatusSink is a documented contract, not an abstract base class: nothing makes
    a subclass implement it, and a moment left out inherits the contract's own empty
    body — a silent no-op that reports nothing and raises nothing.

    Scanned rather than listed, for the reason in _drive_every_moment. MenuBarSink is
    not checked here because it lives in menubar.py, which this file may not import;
    its own check belongs with it."""
    moments = [name for name, value in vars(dictation.StatusSink).items()
               if not name.startswith("_") and callable(value)]
    assert moments, "no moments were discovered on StatusSink"
    for implementation in (dictation.NotifySink, dictation.FanoutSink):
        for moment in moments:
            assert (getattr(implementation, moment)
                    is not getattr(dictation.StatusSink, moment)), \
                f"{implementation.__name__} inherits {moment}() as a silent no-op"
    print(f"  {len(moments)} moments implemented by both sinks: {sorted(moments)}")


def test_notify_sink_reports_each_power_change():
    """The fifth moment exists because enable() returns before the model is loaded:
    LOADING becoming ON is something only a sink call can tell the app about.

    All three have to read differently, or "still loading the model" and "ready" look
    the same to the user. No cue: the two that cue are the start and the stop of
    speaking, where the user is not looking at the screen."""
    notified, sounds = [], []
    sink = dictation.NotifySink(notify=notified.append,
                                sound=lambda: sounds.append(None))
    powers = (dictation.POWER_OFF, dictation.POWER_LOADING, dictation.POWER_ON)
    for power in powers:
        sink.power_changed(power)
    assert len(notified) == len(powers), notified
    assert not sounds, f"a power change must not cue: {sounds}"
    assert len(set(notified)) == len(powers), \
        f"two power changes read the same to the user: {notified}"
    print(f"  {notified}")


def test_every_moment_survives_a_failing_notifier():
    """A broken osascript must not cost the user a dictation whose text is
    already on the clipboard: notify raising must not propagate. The swallowed
    exception must not be lost entirely either — it goes to
    audiocript._log_problem, pinned here by monkeypatching it.

    This test named done() and failed() until the contract grew a fifth moment; it
    scans now, so the sixth cannot arrive unguarded."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        def broken_notify(_message):
            raise OSError("osascript is not available")
        sink = dictation.NotifySink(notify=broken_notify, sound=lambda: None)
        called = _drive_every_moment(sink)
    finally:
        A._log_problem = saved
    assert len(logged) == len(called), (
        f"{len(called)} moments were driven ({called}) but {len(logged)} problems "
        "were logged")
    print(f"  survived a failing notifier on every moment: {called}")


def test_the_fan_out_forwards_every_moment():
    """The menu bar's icons are added to the notifications, not substituted for them:
    an icon is visible only while the user looks at the menu bar, and NotifySink's
    message with its cue is what says a dictation reached the clipboard. So the daemon
    is handed both sinks behind a fan-out, and every moment has to reach both with the
    same arguments."""
    first, second = RecordingSink(), RecordingSink()
    called = _drive_every_moment(dictation.FanoutSink(first, second))
    for sink, which in ((first, "first"), (second, "second")):
        kinds = sorted(kind for kind, _ in sink.calls)
        assert kinds == called, f"the {which} sink got {kinds}, not {called}"
    assert first.calls == second.calls, \
        f"the two sinks got different arguments:\n{first.calls}\n{second.calls}"
    print(f"  forwarded to both sinks: {called}")


def test_one_failing_sink_does_not_silence_another():
    """Every forward is guarded on its own. Without that, the second sink's
    reliability would depend on the first one's: a raising MenuBarSink would take the
    notification with it, on the one path where the user has already spoken and is
    waiting to be told the text is on the clipboard."""
    class Broken:
        """Raises on every moment, whatever the contract grows to."""
        def __getattr__(self, name):
            def raise_it(*_args):
                raise OSError(f"{name} is broken")
            return raise_it

    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        working = RecordingSink()
        try:
            called = _drive_every_moment(dictation.FanoutSink(Broken(), working))
        except AssertionError:
            raise
        except Exception as e:
            # Caught and re-raised as an assertion so the failure names the defect.
            # Unguarded, the first raising sink escapes the fan-out and support.run
            # — which catches only AssertionError — reports it as a crashed script
            # rather than as this test failing.
            raise AssertionError(
                f"a raising sink propagated out of the fan-out: {e!r}") from None
    finally:
        A._log_problem = saved
    kinds = sorted(kind for kind, _ in working.calls)
    assert kinds == called, f"the working sink lost moments: {kinds} != {called}"
    assert len(logged) == len(called), (
        f"{len(called)} forwards failed but {len(logged)} were logged")
    print(f"  the working sink still got every moment: {kinds}")


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


def test_a_failing_provider_still_delivers_the_raw_transcript():
    """Both ways the provider can fail: a 500, and nothing listening at all. The
    user has already spoken and is waiting, so unpunctuated Turkish beats nothing.

    publish.RETRIES is 0 for the duration: the retry backoff sleeps ~3s per base
    and belongs to publish, which tests/test_publish.py already covers. The seam
    under test here is deliver's except clause, and the call still goes through
    the real publish.openai_complete."""
    api = FakeAPI({ROUTE: [(500, {})]})
    saved_retries, publish.RETRIES = publish.RETRIES, 0
    try:
        for base in (api.url, "http://127.0.0.1:1"):
            clip, sink = FakeClipboard(), RecordingSink()
            d = dictation.deliver(RAW, _config(base), clip, sink)
            assert d.status == dictation.UNCORRECTED, f"{base}: status {d.status!r}"
            assert clip.text == RAW, f"{base}: clipboard {clip.text!r} != the raw text"
            assert clip.writes == 1, f"{base}: clipboard written {clip.writes} times"
            assert d.detail, f"{base}: no reason was recorded"
            assert any(k == "done" for k, _ in sink.calls), f"{base}: {sink.calls!r}"
            assert sink.notes == [d.detail], f"{base}: notes {sink.notes!r}"
    finally:
        publish.RETRIES = saved_retries
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


def test_a_whitespace_reply_is_refused():
    """The reply is as long as the transcript on purpose. A shorter one would be
    refused by the length guard whether or not deliver strips the reply, leaving
    the .strip() unpinned — and a whitespace reply that passes the guard means the
    clipboard is overwritten with nothing, the worst outcome this feature has.

    The reply is injected rather than served because publish.openai_complete turns
    a blank message into a RuntimeError of its own, which would exercise the
    provider-failure path instead of deliver's own handling."""
    blanks = " " * len(RAW)
    d = _refused(blanks, complete=lambda prompt, cfg: blanks)
    print(f"  refused {len(blanks)} spaces against {len(RAW)} chars: {d.detail}")


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


def test_a_refused_correction_says_why_in_the_notification():
    """Without this the user sees a plain success notification, pastes
    unpunctuated Turkish, and has nothing anywhere saying the correction was
    refused. Driven end to end — deliver -> sink.done(text, note) -> the
    message NotifySink builds — because the gap can open at either step."""
    notified = []
    sink = dictation.NotifySink(notify=notified.append, sound=lambda: None)
    api = FakeAPI({ROUTE: [(200, completion(RAW * 5))]})
    clip = FakeClipboard()
    try:
        d = dictation.deliver(RAW, _config(api.url), clip, sink)
    finally:
        api.close()
    assert d.status == dictation.SKIPPED, f"status {d.status!r}"
    assert clip.text == RAW, f"clipboard {clip.text!r}"
    assert len(notified) == 1, f"notified {notified!r}"
    assert d.detail in notified[0], \
        f"the reason {d.detail!r} did not reach the notification {notified[0]!r}"
    print(f"  {notified[0]}")


def test_a_correction_that_went_through_says_nothing_extra():
    """The note is for the paths that need explaining. A plain correction must
    not acquire one, or every successful dictation reads like a warning."""
    api = FakeAPI({ROUTE: [(200, completion(FIXED))]})
    clip, sink = FakeClipboard(), RecordingSink()
    try:
        dictation.deliver(RAW, _config(api.url), clip, sink)
    finally:
        api.close()
    assert sink.notes == [""], f"a corrected dictation carried a note: {sink.notes!r}"
    print("  no note on the corrected path")


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


# ================================ the history ================================


def _lines(path):
    """Every line of the log, parsed. Individually, so one bad line is visible as
    that line rather than as the whole file being unreadable."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_one_line_per_append():
    with workdir("history-append") as home:
        path = home / "dictations.jsonl"
        history = dictation.History(path)
        history.append("Bir.", dictation.CORRECTED)
        history.append("İki.", dictation.UNCORRECTED)
        history.append("Üç.", dictation.SKIPPED)
        entries = _lines(path)
        assert len(entries) == 3, f"three appends left {len(entries)} lines"
        for entry in entries:
            assert set(entry) == {"at", "status", "text"}, \
                f"unexpected shape: {sorted(entry)}"
        assert [e["status"] for e in entries] == [
            dictation.CORRECTED, dictation.UNCORRECTED, dictation.SKIPPED], entries
        print(f"  {[e['text'] for e in entries]}")


def test_turkish_survives_the_round_trip():
    """A log full of \\u011f is a log nobody greps. ensure_ascii=False is the point,
    and the raw bytes are what proves it — a json.loads round trip would pass either
    way."""
    spoken = "Toplantıyı yarına alalım, şimdi çıkıyorum. ĞÜŞİÖÇ ığüşöç"
    with workdir("history-turkish") as home:
        path = home / "dictations.jsonl"
        dictation.History(path).append(spoken, dictation.CORRECTED)
        assert _lines(path)[0]["text"] == spoken, "the text did not round trip"
        raw = path.read_bytes()
        assert "ığüşöç".encode("utf-8") in raw, \
            "the file escaped its non-ASCII characters instead of writing them"
        assert b"\\u" not in raw, f"the file holds escape sequences: {raw!r}"
        print(f"  {len(raw)} bytes, readable as written")


def test_the_parent_directory_is_created():
    """~/.audiocript exists only because the daemon made it; a fresh machine has
    neither it nor the log."""
    with workdir("history-mkdir") as home:
        path = home / "not" / "there" / "dictations.jsonl"
        dictation.History(path).append("Bir.", dictation.CORRECTED)
        assert path.exists(), f"{path} was not created"


def test_recent_returns_the_newest_first():
    """The menu lists the newest dictation at the top: it is the one the user is most
    likely to want back."""
    with workdir("history-order") as home:
        history = dictation.History(home / "dictations.jsonl")
        for i in range(5):
            history.append(f"dictation {i}", dictation.CORRECTED)
        texts = [e.text for e in history.recent()]
        assert texts == [f"dictation {i}" for i in reversed(range(5))], texts
        print(f"  {texts}")


def test_recent_honours_the_limit():
    with workdir("history-limit") as home:
        history = dictation.History(home / "dictations.jsonl")
        for i in range(25):
            history.append(f"dictation {i:02d}", dictation.CORRECTED)
        entries = history.recent(3)
        assert [e.text for e in entries] == ["dictation 24", "dictation 23",
                                             "dictation 22"], [e.text for e in entries]
        assert len(history.recent()) == dictation.HISTORY_ITEMS, \
            f"the default limit is not HISTORY_ITEMS: {len(history.recent())}"


def test_an_unparseable_line_is_skipped():
    """Anything can end up in a file a user can open. One bad line must cost that
    line, not the whole history."""
    with workdir("history-garbage") as home:
        path = home / "dictations.jsonl"
        history = dictation.History(path)
        history.append("Bir.", dictation.CORRECTED)
        with open(path, "a", encoding="utf-8") as f:
            f.write("this is not json at all\n")
        history.append("İki.", dictation.CORRECTED)
        texts = [e.text for e in history.recent()]
        assert texts == ["İki.", "Bir."], f"the garbage line was not skipped: {texts}"
        print(f"  survived a garbage line: {texts}")


def test_a_truncated_final_line_is_survived():
    """The normal thing to find in a file being appended to when the process died
    mid-write. The complete entries before it are still the user's history."""
    with workdir("history-truncated") as home:
        path = home / "dictations.jsonl"
        history = dictation.History(path)
        for i in range(3):
            history.append(f"dictation {i}", dictation.CORRECTED)
        raw = path.read_bytes()
        last = raw.rfind(b"\n")                  # newline ending the third line
        previous = raw.rfind(b"\n", 0, last)     # newline ending the second
        path.write_bytes(raw[:previous + 21])    # two whole lines + 20 B of the third
        texts = [e.text for e in history.recent()]
        assert texts == ["dictation 1", "dictation 0"], \
            f"a truncated tail cost more than its own line: {texts}"
        print(f"  {texts} survived a truncated last line")


def test_a_missing_file_reads_empty():
    """The menu asks for the history before the first dictation exists. Reading must
    not create the file either — an empty log on disk is a lie about what happened."""
    with workdir("history-missing") as home:
        path = home / "dictations.jsonl"
        assert dictation.History(path).recent() == [], "a missing log read non-empty"
        assert not path.exists(), "reading the history created the file"


def test_only_the_tail_is_read():
    """The log is unbounded and the menu reads it every time it opens, so recent()
    must not scale with the whole file.

    Pinned by shrinking the window rather than by timing anything: with a window that
    holds only the last few lines, the older ones are provably not read, because they
    do not come back."""
    with workdir("history-tail") as home:
        path = home / "dictations.jsonl"
        history = dictation.History(path, tail_bytes=300)
        for i in range(40):
            history.append(f"dictation number {i:03d}", dictation.CORRECTED)
        size = path.stat().st_size
        assert size > 300 * 4, f"the fixture is too small to prove anything: {size} B"
        entries = history.recent(40)
        assert entries, "the tail read returned nothing at all"
        assert len(entries) < 40, (
            f"all {len(entries)} entries came back from a {size} B file with a 300 B "
            "window, so the whole file was read")
        assert entries[0].text == "dictation number 039", \
            f"the newest entry is not first: {entries[0].text!r}"
        print(f"  {len(entries)} of 40 entries, from the last 300 B of {size} B")


def test_a_write_failure_is_swallowed():
    """The text is already on the clipboard by the time the log is written. Losing the
    log entry must not cost the dictation, and must not be silent either."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        with workdir("history-unwritable") as home:
            blocker = home / "blocker"
            blocker.write_text("not a directory")
            dictation.History(blocker / "dictations.jsonl").append(
                "Bir.", dictation.CORRECTED)
    finally:
        A._log_problem = saved
    assert len(logged) == 1, f"the failure was not logged exactly once: {logged}"
    print(f"  swallowed and logged: {logged[0][0]}")


def test_a_read_failure_is_not_swallowed():
    """The counterpart decision. An unreadable log returning [] would look exactly
    like "no dictations yet" in the menu, which is the one thing the menu must not
    say wrongly — so the menu layer decides what to show, not History."""
    with workdir("history-unreadable") as home:
        path = home / "dictations.jsonl"
        path.mkdir()                     # a directory where a file belongs
        try:
            dictation.History(path).recent()
        except OSError as e:
            print(f"  raised, as it should: {type(e).__name__}")
            return
        raise AssertionError("an unreadable log read as an empty history")


def test_concurrent_appends_do_not_interleave():
    """The worker thread appends while the main thread reads the menu. Two appends
    landing in the middle of each other would leave a line no parser can read — and
    the entry it destroyed would be a dictation the user had already spoken."""
    with workdir("history-threads") as home:
        path = home / "dictations.jsonl"
        history = dictation.History(path)
        threads = [threading.Thread(
            target=lambda n=n: [history.append(f"thread {n} line {i} " + "x" * 200,
                                               dictation.CORRECTED)
                                for i in range(20)])
                   for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        entries = _lines(path)           # raises if any single line is unparseable
        assert len(entries) == 8 * 20, f"{len(entries)} of 160 appends survived"
        print(f"  {len(entries)} lines from 8 threads, every one parseable")


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())

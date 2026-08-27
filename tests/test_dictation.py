"""dictation.resolve_config: the configuration half of the dictation daemon.

Nothing here may reach a real API or read a developer's .env. Every test passes
an explicit `env`, so publish.env_value is never consulted — except the last
test, which proves resolve_config's default falls back to publish.env_value
while keeping publish.ROOT pointed at an empty directory so it still cannot
reach a real .env.
"""
import os
import tempfile
from pathlib import Path

from support import A, run, fake_env

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
    for bad in ("cmd+alt+d", "<zort>+d", ""):
        cfg = {"dictation_hotkeys": {"toggle": bad}}
        _refuses(cfg, fake_env(**BASE_ENV), bad)
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


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())

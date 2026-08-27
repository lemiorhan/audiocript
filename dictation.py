"""The parts of dictation holding no threads and no global state: configuration
today, and — as later tasks add them — the clipboard, the status sinks, and the
correction pipeline. Kept apart from dictate.py's state machine and hotkey
listener so all of it can be tested without audio hardware or a running daemon.
"""
from dataclasses import dataclass

from pynput.keyboard import HotKey, Key

import audiocript as A
import publish

HOTKEY_ACTIONS = ("toggle",)
DEFAULT_HOTKEYS = {"toggle": "<cmd>+<alt>+d"}
DEFAULT_MAX_SECONDS = 300
DEFAULT_MODEL = "gpt-4.1-mini"


class ConfigError(Exception):
    """Raised by resolve_config when a configured value cannot be used as given."""


@dataclass
class DictationConfig:
    language: str
    hotkeys: dict
    max_seconds: int
    model: str
    openai_token: str
    openai_base: str


def _validated_hotkeys(cfg):
    """DEFAULT_HOTKEYS with any configured combinations laid over it, so a partial
    `dictation_hotkeys` object keeps the actions it did not mention."""
    hotkeys = {**DEFAULT_HOTKEYS, **(cfg.get("dictation_hotkeys") or {})}
    for action, combination in hotkeys.items():
        if action not in HOTKEY_ACTIONS:
            raise ConfigError(
                f"unknown hotkey action {action!r}: expected one of {HOTKEY_ACTIONS}")
        try:
            parsed = HotKey.parse(combination)
        except ValueError:
            raise ConfigError(
                f"hotkey {combination!r} for the {action!r} action is not a "
                "valid key combination") from None
        if not any(isinstance(key, Key) for key in parsed):
            raise ConfigError(
                f"hotkey {combination!r} for the {action!r} action needs a "
                "modifier such as <cmd>, <alt> or <shift> — otherwise it would "
                "fire on every matching key typed anywhere")
    return hotkeys


def _validated_max_seconds(cfg):
    value = cfg.get("dictation_max_seconds", DEFAULT_MAX_SECONDS)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(
            f"dictation_max_seconds must be a positive integer, got {value!r}")
    return value


def resolve_config(cfg, env=None):
    """Turn the dict from audiocript.load_config() plus the environment into a
    validated DictationConfig, or raise ConfigError naming what is wrong.

    `env` defaults to publish.env_value; tests pass a fake one so nothing here
    ever touches a developer's real .env."""
    env = env or publish.env_value

    language = cfg.get("language", "tr")
    if language not in A.LANGUAGES:
        raise ConfigError(
            f"unknown language {language!r}: expected one of {sorted(A.LANGUAGES)}")

    hotkeys = _validated_hotkeys(cfg)
    max_seconds = _validated_max_seconds(cfg)

    openai_token = env("OPENAI_API_KEY")
    if not openai_token:
        raise ConfigError("OPENAI_API_KEY is required (set it in the environment "
                          "or in .env)")

    return DictationConfig(
        language=language,
        hotkeys=hotkeys,
        max_seconds=max_seconds,
        model=env("DICTATION_MODEL", DEFAULT_MODEL),
        openai_token=openai_token,
        openai_base=env("OPENAI_API_BASE", "https://api.openai.com"))

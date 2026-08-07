"""Turn a finished transcript into an edited transcript and a document, and file all
three in a GitHub repository.

Kept out of audiocript.py because none of it touches the UI: it is configuration, two
OpenAI calls and the GitHub Git Data API, which makes the whole pipeline testable on
its own against a local server.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MIN_CHARS = 2000
OPENAI_TIMEOUT = 600          # a long transcript genuinely takes minutes to generate
GITHUB_TIMEOUT = 30
RETRIES = 2


def _dotenv():
    """Values from the .env next to the app: KEY=value, '#' comments, optional quotes.

    Not cached — the file is a handful of lines read a few times per publish, and a
    cache would only make the setting stale after the user finally fills it in."""
    values = {}
    try:
        text = (ROOT / ".env").read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def env_value(key, default=None):
    """An exported environment variable wins over .env, matching how _hf_token already
    resolves its token."""
    return os.environ.get(key) or _dotenv().get(key) or default


_REPO_URL = re.compile(
    r"(?:https?://|git@)[^/:]+[/:]+(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?$")


def normalize_repo(value):
    """Reduce whatever was configured to `owner/repo`.

    The README asks for `owner/repo`, but the natural thing is to paste the address
    out of the browser — and a full URL used to go straight into the API path, giving
    /repos/https://github.com/owner/repo, which can never work. Browser URLs, SSH
    remotes and a bare owner/repo all land in the same place now."""
    value = (value or "").strip().rstrip("/")
    match = _REPO_URL.match(value)
    if match:
        return f"{match['owner']}/{match['name']}"
    return value.removesuffix(".git")


def config():
    """Everything the publish job needs, or None when it is not set up.

    Missing keys mean the feature is off rather than broken: someone who never
    configured it must not get a failure message after every recording."""
    repo = env_value("GITHUB_REPO_FOR_TRANSCRIPTS")
    github_token = env_value("GITHUB_TOKEN")
    openai_token = env_value("OPENAI_API_KEY")
    if not (repo and github_token and openai_token):
        return None
    try:
        min_chars = int(env_value("TRANSCRIPT_MIN_CHARS", DEFAULT_MIN_CHARS))
    except ValueError:
        min_chars = DEFAULT_MIN_CHARS
    return {
        "repo": normalize_repo(repo),
        "github_token": github_token,
        "openai_token": openai_token,
        "model": env_value("OPENAI_MODEL", DEFAULT_MODEL),
        "min_chars": min_chars,
        "openai_base": env_value("OPENAI_API_BASE", "https://api.openai.com"),
        "github_base": env_value("GITHUB_API_BASE", "https://api.github.com"),
    }


# =============================== HTTP ===============================

_PLACEHOLDER = re.compile(r"^\[[^\]\n]*\]\s*$", re.MULTILINE)


def _json_or_text(raw):
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {"error": {"message": raw[:200]}}


def _detail(obj):
    """The human-readable part of an API error body, whatever shape it arrived in."""
    if isinstance(obj, dict):
        error = obj.get("error")
        if isinstance(error, dict):
            return error.get("message") or json.dumps(error)[:200]
        if error:
            return str(error)
        if obj.get("message"):
            return str(obj["message"])
    return json.dumps(obj)[:200] if obj else ""


def _request(method, url, token, payload=None, timeout=GITHUB_TIMEOUT,
             accept="application/json"):
    """One API call, returning (status, parsed body).

    Retries 429, 5xx and connection failures; every other status comes back for the
    caller to judge, because a 404 on a branch ref is an expected answer rather than a
    failure. Raises once the retries are spent."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    last = None
    for attempt in range(RETRIES + 1):
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Accept", accept)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, _json_or_text(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = _json_or_text(e.read().decode("utf-8", "replace"))
            if e.code != 429 and e.code < 500:
                return e.code, body
            last = RuntimeError(f"{method} {url} -> {e.code}: {_detail(body)}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = RuntimeError(f"{method} {url}: {e}")
        if attempt < RETRIES:
            time.sleep(2 ** attempt)
    raise last


# ============================ OpenAI stages ============================

def render_prompt(path, body):
    """Put `body` where the prompt file's trailing placeholder line is.

    Both prompts end with a bracketed line ([TRANSCRIPT BURAYA]) — they are written to
    be read as one instruction with the text inlined, so that is how they are sent."""
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_PLACEHOLDER.finditer(text))
    if not matches:
        return text.rstrip() + "\n\n" + body
    last = matches[-1]
    return text[:last.start()] + body + text[last.end():]


def openai_complete(prompt, cfg):
    """One chat completion, returned as text.

    A response the model had to cut short is treated as a failure: half a document
    filed in a repository is worse than none."""
    status, obj = _request("POST", f"{cfg['openai_base']}/v1/chat/completions",
                           cfg["openai_token"],
                           {"model": cfg["model"],
                            "messages": [{"role": "user", "content": prompt}]},
                           timeout=OPENAI_TIMEOUT)
    if not 200 <= status < 300:
        raise RuntimeError(f"OpenAI {status}: {_detail(obj)}")
    choice = (obj.get("choices") or [{}])[0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"the transcript is past {cfg['model']}'s output limit "
                           "(the response was cut short)")
    text = ((choice.get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response")
    return text


# ============================ GitHub publishing ============================

_UNSAFE_IN_PATH = re.compile(r"[\s/\\:]+")


def folder_name(dir_name, title):
    """The recording's own folder name plus its title, so the repository sorts
    chronologically and lines up with what is on disk. Characters that fight with
    paths become '-'; Turkish letters are left alone."""
    title = _UNSAFE_IN_PATH.sub("-", (title or "").strip()).strip("-")
    return f"{dir_name}-{title}" if title else dir_name


def _gh(method, path, cfg, payload=None):
    return _request(method, f"{cfg['github_base']}{path}", cfg["github_token"],
                    payload, timeout=GITHUB_TIMEOUT,
                    accept="application/vnd.github+json")


def _ok(status, obj, what):
    if not 200 <= status < 300:
        raise RuntimeError(f"GitHub {status} on {what}: {_detail(obj)}")
    return obj


def publish_files(cfg, folder, files, message):
    """Commit `files` ({name: text}) under `folder` in a single commit; return its URL.

    The Contents API would be shorter but makes one commit per file, and the commit is
    meant to name the recording once. This walks the Git Data API instead, which also
    means nothing has to be cloned."""
    repo = cfg["repo"]
    info = _ok(*_gh("GET", f"/repos/{repo}", cfg), f"repo {repo}")
    branch = info.get("default_branch") or "main"

    status, ref = _gh("GET", f"/repos/{repo}/git/ref/heads/{branch}", cfg)
    empty = status == 404                     # a repository with no commits yet
    parents, base_tree = [], None
    if not empty:
        _ok(status, ref, f"branch {branch}")
        head = ref["object"]["sha"]
        parents = [head]
        commit = _ok(*_gh("GET", f"/repos/{repo}/git/commits/{head}", cfg),
                     "the current commit")
        base_tree = commit["tree"]["sha"]

    entries = []
    for name, text in files.items():
        blob = _ok(*_gh("POST", f"/repos/{repo}/git/blobs", cfg,
                        {"content": text, "encoding": "utf-8"}), f"the blob for {name}")
        entries.append({"path": f"{folder}/{name}", "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})

    tree_payload = {"tree": entries}
    if base_tree:
        tree_payload["base_tree"] = base_tree
    tree = _ok(*_gh("POST", f"/repos/{repo}/git/trees", cfg, tree_payload), "the tree")

    commit = _ok(*_gh("POST", f"/repos/{repo}/git/commits", cfg,
                      {"message": message, "tree": tree["sha"], "parents": parents}),
                 "the commit")

    if empty:
        _ok(*_gh("POST", f"/repos/{repo}/git/refs", cfg,
                 {"ref": f"refs/heads/{branch}", "sha": commit["sha"]}),
            f"creating {branch}")
    else:
        _ok(*_gh("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", cfg,
                 {"sha": commit["sha"]}), f"moving {branch}")
    return commit.get("html_url") or f"https://github.com/{repo}/commit/{commit['sha']}"


# ============================ The whole pipeline ============================

RAW_NAME = "transcript.raw.md"
EDITED_NAME = "transcript.edited.md"
DOC_NAME = "documentation.md"


def run(cfg, project_dir, text, title, on_step=None):
    """Edit the transcript, write it up, and file all three forms in one commit.
    Returns the commit's URL.

    Each stage is written into the recording folder before the next one starts, and a
    stage whose output is already there is reused rather than run again. That matters
    on a retry: these responses take minutes and cost real money, so a run that died
    at the documentation stage — or was cut short by quitting the app — must not pay
    for the edit a second time."""
    d = Path(project_dir)

    def step(name):
        if on_step:
            on_step(name)

    def finished(path):
        """The output of a stage that already completed, or None."""
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    edited = finished(d / EDITED_NAME)
    if edited is None:
        step("edit")
        edited = openai_complete(
            render_prompt(PROMPTS / "transcript_prompt.md", text), cfg)
        (d / EDITED_NAME).write_text(edited, encoding="utf-8")

    document = finished(d / DOC_NAME)
    if document is None:
        step("document")
        document = openai_complete(
            render_prompt(PROMPTS / "documentation_prompt.md", edited), cfg)
        (d / DOC_NAME).write_text(document, encoding="utf-8")

    step("publish")
    folder = folder_name(d.name, title)
    return publish_files(cfg, folder,
                         {RAW_NAME: text, EDITED_NAME: edited, DOC_NAME: document},
                         folder)

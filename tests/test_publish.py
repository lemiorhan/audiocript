"""The publish pipeline: configuration, the OpenAI stages, and the GitHub commit.

Nothing here may reach the real APIs. Every test either passes an explicit config
pointed at the local fake server, or goes through `isolated` — which clears the
publishing variables and points publish.ROOT at an empty directory, so a developer's
real .env can never be picked up and spent.
"""
import base64
import contextlib
import json
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

from support import A, REPO_ROOT, run, workdir

import publish

PUBLISH_VARS = ("GITHUB_REPO_FOR_TRANSCRIPTS", "GITHUB_TOKEN", "OPENAI_API_KEY",
                "OPENAI_MODEL", "TRANSCRIPT_MIN_CHARS", "OPENAI_API_BASE",
                "GITHUB_API_BASE")


@contextlib.contextmanager
def isolated(root):
    """Clear the publishing environment and read .env from `root` instead of the repo."""
    saved = {k: os.environ.pop(k, None) for k in PUBLISH_VARS}
    publish.ROOT = root
    try:
        yield
    finally:
        publish.ROOT = REPO_ROOT
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_config_is_none_without_keys():
    with workdir("noenv") as d, isolated(d):
        assert publish.config() is None, "configured itself out of nothing"
    print("  no keys -> feature off")


def test_config_reads_dotenv():
    with workdir("dotenv") as d, isolated(d):
        (d / ".env").write_text(
            "# a comment\n"
            "\n"
            "GITHUB_REPO_FOR_TRANSCRIPTS=owner/repo\n"
            'GITHUB_TOKEN="gh-token"\n'
            "OPENAI_API_KEY='oa-token'\n"
            "TRANSCRIPT_MIN_CHARS=50\n")
        cfg = publish.config()
        print(f"  repo={cfg['repo']} model={cfg['model']} min_chars={cfg['min_chars']}")
        assert cfg["repo"] == "owner/repo"
        assert cfg["github_token"] == "gh-token", "double quotes not stripped"
        assert cfg["openai_token"] == "oa-token", "single quotes not stripped"
        assert cfg["min_chars"] == 50
        assert cfg["model"] == publish.DEFAULT_MODEL
        assert cfg["github_base"] == "https://api.github.com"


def test_exported_variable_beats_dotenv():
    with workdir("override") as d, isolated(d):
        (d / ".env").write_text("GITHUB_REPO_FOR_TRANSCRIPTS=from/file\n"
                                "GITHUB_TOKEN=g\nOPENAI_API_KEY=o\n")
        os.environ["GITHUB_REPO_FOR_TRANSCRIPTS"] = "from/env"
        assert publish.config()["repo"] == "from/env", ".env overrode the export"
    print("  exported variable wins")


def test_bad_min_chars_falls_back():
    with workdir("badmin") as d, isolated(d):
        (d / ".env").write_text("GITHUB_REPO_FOR_TRANSCRIPTS=o/r\nGITHUB_TOKEN=g\n"
                                "OPENAI_API_KEY=o\nTRANSCRIPT_MIN_CHARS=lots\n")
        assert publish.config()["min_chars"] == publish.DEFAULT_MIN_CHARS
    print("  unparsable threshold -> default")


class FakeAPI:
    """A stand-in for api.openai.com and api.github.com that records what it was sent.

    Routes are (method, path-prefix) -> list of (status, body) replies, consumed in
    order; a single reply repeats forever. `requests` holds every call, so a test can
    assert on what the code actually sent rather than only on what came back."""

    def __init__(self, routes):
        self.routes, self.requests = routes, []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                api.requests.append({"method": self.command, "path": self.path,
                                     "body": json.loads(raw) if raw else None,
                                     "headers": dict(self.headers)})
                for (method, prefix), replies in api.routes.items():
                    if self.command != method or not self.path.startswith(prefix):
                        continue
                    status, payload = replies[0] if len(replies) == 1 else replies.pop(0)
                    out = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(599)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = do_PATCH = do_PUT = _reply

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def paths(self, method=None):
        return [r["path"] for r in self.requests if method in (None, r["method"])]


def completion(text, finish_reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}


def configured(api, **overrides):
    """A config dict pointed at the fake server, bypassing config() entirely."""
    cfg = {"repo": "owner/repo", "github_token": "gh", "openai_token": "oa",
           "model": "test-model", "min_chars": 10,
           "openai_base": api.url, "github_base": api.url}
    cfg.update(overrides)
    return cfg


def test_render_prompt_substitutes_the_placeholder():
    body = publish.render_prompt(publish.PROMPTS / "transcript_prompt.md", "AAA")
    assert "[TRANSCRIPT BURAYA]" not in body, "placeholder left in place"
    assert body.rstrip().endswith("AAA"), "transcript not substituted at the end"
    assert "toplantı editörü" in body, "prompt body lost"
    print(f"  transcript prompt rendered, {len(body)} chars")


def test_render_prompt_appends_when_there_is_no_placeholder():
    with workdir("prompt") as d:
        (d / "p.md").write_text("Do the thing.\n")
        body = publish.render_prompt(d / "p.md", "AAA")
        assert body == "Do the thing.\n\nAAA", f"got {body!r}"
    print("  appended when no placeholder")


def test_openai_returns_the_message_content():
    api = FakeAPI({("POST", "/v1/chat/completions"): [(200, completion("edited text"))]})
    try:
        out = publish.openai_complete("prompt here", configured(api))
        sent = api.requests[0]
        print(f"  got {out!r}, model {sent['body']['model']}")
        assert out == "edited text"
        assert sent["body"]["messages"][0]["content"] == "prompt here"
        assert sent["headers"]["Authorization"] == "Bearer oa"
    finally:
        api.close()


def test_openai_truncation_raises():
    api = FakeAPI({("POST", "/v1/chat/completions"):
                   [(200, completion("half a doc", finish_reason="length"))]})
    try:
        publish.openai_complete("prompt", configured(api))
    except RuntimeError as e:
        print(f"  raised: {e}")
        assert "output limit" in str(e), f"unhelpful message: {e}"
    else:
        raise AssertionError("a truncated response was accepted")
    finally:
        api.close()


def test_openai_error_status_raises():
    api = FakeAPI({("POST", "/v1/chat/completions"):
                   [(401, {"error": {"message": "bad key"}})]})
    try:
        publish.openai_complete("prompt", configured(api))
    except RuntimeError as e:
        print(f"  raised: {e}")
        assert "401" in str(e) and "bad key" in str(e)
    else:
        raise AssertionError("a 401 was accepted")
    finally:
        api.close()


def test_server_errors_are_retried():
    api = FakeAPI({("POST", "/v1/chat/completions"):
                   [(500, {"error": "boom"}), (200, completion("ok"))]})
    try:
        assert publish.openai_complete("prompt", configured(api)) == "ok"
        print(f"  {len(api.requests)} attempts, the second succeeded")
        assert len(api.requests) == 2, f"expected a retry, made {len(api.requests)}"
    finally:
        api.close()


def github_routes(ref_status=200, tree_status=201):
    """The Git Data API call sequence a successful publish makes."""
    return {
        ("GET", "/repos/owner/repo/git/ref/"): [
            (ref_status, {"object": {"sha": "head-sha"}} if ref_status == 200 else {})],
        ("GET", "/repos/owner/repo/git/commits/"): [(200, {"tree": {"sha": "base-tree"}})],
        ("GET", "/repos/owner/repo"): [(200, {"default_branch": "trunk"})],
        ("POST", "/repos/owner/repo/git/blobs"): [(201, {"sha": "blob"})],
        ("POST", "/repos/owner/repo/git/trees"): [(tree_status, {"sha": "tree"})],
        ("POST", "/repos/owner/repo/git/commits"): [
            (201, {"sha": "new-sha",
                   "html_url": "https://github.com/owner/repo/commit/new-sha"})],
        ("PATCH", "/repos/owner/repo/git/refs/"): [(200, {})],
        ("POST", "/repos/owner/repo/git/refs"): [(201, {})],
    }


def empty_repo_routes(ref_status):
    """A repo with no commits: the branch ref lookup answers 404 or 409, the Git Data
    API is unavailable, and files are created one at a time with the Contents API."""
    return {  # specific prefixes first: "/repos/owner/repo" is a prefix of them all
        ("GET", "/repos/owner/repo/git/ref/"): [(ref_status, {})],
        ("PUT", "/repos/owner/repo/contents/"): [
            (201, {"commit": {
                "sha": "seed-sha",
                "html_url": "https://github.com/owner/repo/commit/seed-sha"}})],
        ("GET", "/repos/owner/repo"): [(200, {"default_branch": "trunk"})],
    }


def test_folder_name():
    assert publish.folder_name("2026-08-07_13-47-05", "fatih11-1") == \
        "2026-08-07_13-47-05-fatih11-1"
    assert publish.folder_name("2026-08-07_13-47-05", "") == "2026-08-07_13-47-05", \
        "an untitled recording should not get a trailing separator"
    assert publish.folder_name("d", "a b/c") == "d-a-b-c", "unsafe characters kept"
    assert publish.folder_name("d", "  Toplantı  ") == "d-Toplantı", \
        "Turkish letters should survive"
    print("  folder names built")


def test_publish_makes_one_commit_with_every_file():
    api = FakeAPI(github_routes())
    try:
        url = publish.publish_files(configured(api), "2026-08-07-demo",
                                    {"a.md": "A", "b.md": "B", "c.md": "C"},
                                    "2026-08-07-demo")
        blobs = [r for r in api.requests if r["path"].endswith("/git/blobs")]
        commits = [r for r in api.requests
                   if r["method"] == "POST" and r["path"].endswith("/git/commits")]
        tree = next(r for r in api.requests if r["path"].endswith("/git/trees"))
        print(f"  {len(blobs)} blobs, {len(commits)} commit, url {url}")
        assert url.endswith("/commit/new-sha")
        assert len(blobs) == 3, f"expected one blob per file, got {len(blobs)}"
        assert len(commits) == 1, f"expected a single commit, got {len(commits)}"
        assert commits[0]["body"]["message"] == "2026-08-07-demo"
        assert commits[0]["body"]["parents"] == ["head-sha"]
        assert tree["body"]["base_tree"] == "base-tree", "commit did not build on HEAD"
        assert {e["path"] for e in tree["body"]["tree"]} == {
            "2026-08-07-demo/a.md", "2026-08-07-demo/b.md", "2026-08-07-demo/c.md"}
        assert any("/git/ref/heads/trunk" in p for p in api.paths("GET")), \
            "did not use the repository's default branch"
    finally:
        api.close()


def _assert_contents_bootstrap(ref_status):
    """A repo whose branch ref answers `ref_status` has no commits: the Git Data API is
    unreachable, so each file is written with the Contents API, one commit apiece."""
    api = FakeAPI(empty_repo_routes(ref_status))
    try:
        url = publish.publish_files(configured(api), "f",
                                    {"a.md": "A", "b.md": "BB"}, "msg")
        puts = [r for r in api.requests if r["method"] == "PUT"]
        print(f"  ref {ref_status} -> {len(puts)} Contents PUT(s), url {url}")
        assert len(puts) == 2, f"one Contents PUT per file, got {len(puts)}"
        assert not any(p.endswith(("/git/blobs", "/git/trees", "/git/commits",
                                   "/git/refs")) for p in api.paths()), \
            "the Git Data API must not be touched on an empty repository"
        assert base64.b64decode(puts[0]["body"]["content"]).decode() == "A", \
            "the file content is sent base64-encoded"
        assert puts[0]["path"].startswith("/repos/owner/repo/contents/f/"), \
            "the file lands under the recording folder"
        assert puts[0]["body"]["branch"] == "trunk", "wrote to the default branch"
        assert url.endswith("/commit/seed-sha")
    finally:
        api.close()


def test_publish_into_an_empty_repository():
    """A repo whose branch ref is a plain 404 has no commit to build on."""
    _assert_contents_bootstrap(ref_status=404)


def test_publish_into_a_never_committed_repository():
    """GitHub answers a repo that has no commits at all with 409 'Git Repository is
    empty.' on the branch ref — not 404. Both take the Contents-API path, or a
    brand-new transcripts repo could never receive its first files."""
    _assert_contents_bootstrap(ref_status=409)


def test_publish_reports_a_failed_step():
    api = FakeAPI(github_routes(tree_status=422))
    try:
        publish.publish_files(configured(api), "f", {"a.md": "A"}, "msg")
    except RuntimeError as e:
        print(f"  raised: {e}")
        assert "422" in str(e)
    else:
        raise AssertionError("a 422 on the tree was ignored")
    finally:
        api.close()


@contextlib.contextmanager
def configured_env(api, **extra):
    """Point the app's own config() at the fake server for the duration."""
    saved = {k: os.environ.get(k) for k in PUBLISH_VARS}
    os.environ.update({"GITHUB_REPO_FOR_TRANSCRIPTS": "owner/repo",
                       "GITHUB_TOKEN": "gh", "OPENAI_API_KEY": "oa",
                       "OPENAI_MODEL": "test-model", "TRANSCRIPT_MIN_CHARS": "20",
                       "OPENAI_API_BASE": api.url, "GITHUB_API_BASE": api.url})
    os.environ.update(extra)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def recording(d, text, name="demo", published=None):
    """A recording folder with a saved transcript, ready to publish."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcription.txt").write_text(text, encoding="utf-8")
    meta = {"name": name, "language": "tr"}
    if published:
        meta["published"] = published
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def fake_state(base):
    """A stand-in for _TuiState carrying only what publishing touches. `jobs` and
    `jobs_lock` are the list a publish registers itself on and the lock every mutation
    of it goes through."""
    return types.SimpleNamespace(status="", base_path=base, recordings=[],
                                 jobs=[], jobs_lock=threading.Lock())


def pipeline_routes():
    routes = github_routes()
    routes[("POST", "/v1/chat/completions")] = [(200, completion("EDITED")),
                                                (200, completion("DOCUMENT"))]
    return routes


def test_run_writes_both_files_and_publishes():
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("run") as base:
            d = recording(base / "2026-08-07_13-47-05", "x" * 100)
            url = publish.run(configured(api), d, "x" * 100, "demo")
            tree = next(r for r in api.requests if r["path"].endswith("/git/trees"))
            names = sorted(e["path"].split("/")[-1] for e in tree["body"]["tree"])
            print(f"  published {names} -> {url}")
            assert (d / "transcript.edited.md").read_text() == "EDITED"
            assert (d / "documentation.md").read_text() == "DOCUMENT"
            assert names == ["documentation.md", "transcript.edited.md",
                             "transcript.raw.md"]
            assert all(e["path"].startswith("2026-08-07_13-47-05-demo/")
                       for e in tree["body"]["tree"])
            # The second call must be fed the first call's output, not the raw text.
            second = [r for r in api.requests if "chat/completions" in r["path"]][1]
            assert "EDITED" in second["body"]["messages"][0]["content"]
    finally:
        api.close()


def test_truncation_writes_nothing_and_publishes_nothing():
    routes = pipeline_routes()
    routes[("POST", "/v1/chat/completions")] = [
        (200, completion("half", finish_reason="length"))]
    api = FakeAPI(routes)
    try:
        with workdir("trunc") as base:
            d = recording(base / "2026-08-07_10-00-00", "x" * 100)
            try:
                publish.run(configured(api), d, "x" * 100, "demo")
            except RuntimeError as e:
                print(f"  raised: {e}")
            else:
                raise AssertionError("truncation was accepted")
            assert not (d / "transcript.edited.md").exists(), "wrote a truncated stage"
            assert not any("/git/" in p for p in api.paths()), "published anyway"
    finally:
        api.close()


def test_push_failure_keeps_the_model_output():
    routes = pipeline_routes()
    routes[("POST", "/repos/owner/repo/git/trees")] = [(422, {"message": "nope"})]
    api = FakeAPI(routes)
    try:
        with workdir("pushfail") as base:
            d = recording(base / "2026-08-07_11-00-00", "x" * 100)
            try:
                publish.run(configured(api), d, "x" * 100, "demo")
            except RuntimeError as e:
                print(f"  raised: {e}")
            else:
                raise AssertionError("a failed push looked successful")
            assert (d / "transcript.edited.md").read_text() == "EDITED", \
                "the paid-for output was thrown away"
            assert (d / "documentation.md").read_text() == "DOCUMENT"
    finally:
        api.close()


def test_gate_skips_short_transcripts_and_republishing():
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("gate") as base, configured_env(api):
            short = recording(base / "2026-08-07_09-00-00", "too short", name="s")
            assert A._publish_async(fake_state(base), short, "too short", "s") is None
            done = recording(base / "2026-08-07_09-30-00", "x" * 100, name="d",
                             published={"url": "u", "at": "t"})
            assert A._publish_async(fake_state(base), done, "x" * 100, "d") is None
            print(f"  {len(api.requests)} requests made")
            assert api.requests == [], "the gate let a request through"
    finally:
        api.close()


def test_gate_reports_missing_configuration():
    """Nothing publishes by itself any more, so every refusal answers a keypress and
    has to say something — silence here reads as a broken [u]."""
    with workdir("unconfigured") as base, isolated(base):
        d = recording(base / "2026-08-07_08-00-00", "x" * 5000)
        state = fake_state(base)
        assert A._publish_async(state, d, "x" * 5000, "demo") is None
        print(f"  status={state.status!r}")
        assert "not configured" in state.status, state.status
        assert state.jobs == [], "registered a job for work that never started"


def test_end_to_end_updates_status_and_meta():
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("e2e") as base, configured_env(api):
            d = recording(base / "2026-08-07_12-00-00", "x" * 100, name="stand-up")
            state = fake_state(base)
            thread = A._publish_async(state, d, "x" * 100, "stand-up")
            assert thread is not None, "the gate refused a publishable transcript"
            thread.join(60)
            meta = json.loads((d / "meta.json").read_text())
            print(f"  status={state.status!r}")
            assert A._active_jobs(state) == [], (
                "the publish job never reached a terminal state — it would block its "
                "recording and sit in the quit confirmation for the rest of the session")
            assert state.jobs[0].state == A._JOB_DONE, state.jobs[0].state
            assert "Published" in state.status and "new-sha" in state.status
            assert meta["published"]["url"].endswith("/commit/new-sha")
            assert meta["published"]["at"], "no timestamp recorded"
    finally:
        api.close()


def test_a_failure_is_reported_on_the_status_line():
    routes = pipeline_routes()
    routes[("POST", "/v1/chat/completions")] = [(401, {"error": {"message": "bad key"}})]
    api = FakeAPI(routes)
    try:
        with workdir("failmsg") as base, configured_env(api):
            d = recording(base / "2026-08-07_07-00-00", "x" * 100)
            state = fake_state(base)
            A._publish_async(state, d, "x" * 100, "demo").join(60)
            print(f"  status={state.status!r}")
            assert state.status.startswith("Publish failed:"), state.status
            assert "bad key" in state.status
            assert A._active_jobs(state) == [], "a failed publish left its job running"
            assert state.jobs[0].state == A._JOB_FAILED, state.jobs[0].state
            assert "bad key" in state.jobs[0].message, state.jobs[0].message
            assert "published" not in json.loads((d / "meta.json").read_text())
    finally:
        api.close()


def test_repo_accepts_what_people_actually_paste():
    """The README says owner/repo, but the natural thing is to paste the browser URL —
    which used to be dropped into the API path verbatim and could never work."""
    cases = {
        "lemiorhan/meeting_transcripts": "lemiorhan/meeting_transcripts",
        "https://github.com/lemiorhan/meeting_transcripts":
            "lemiorhan/meeting_transcripts",
        "https://github.com/lemiorhan/meeting_transcripts/":
            "lemiorhan/meeting_transcripts",
        "https://github.com/lemiorhan/meeting_transcripts.git":
            "lemiorhan/meeting_transcripts",
        "git@github.com:lemiorhan/meeting_transcripts.git":
            "lemiorhan/meeting_transcripts",
    }
    for given, want in cases.items():
        with workdir("repo") as d, isolated(d):
            (d / ".env").write_text(f"GITHUB_REPO_FOR_TRANSCRIPTS={given}\n"
                                    "GITHUB_TOKEN=g\nOPENAI_API_KEY=o\n")
            got = publish.config()["repo"]
            assert got == want, f"{given!r} -> {got!r}, wanted {want!r}"
    print(f"  {len(cases)} forms all normalise to owner/repo")


def test_run_reuses_stages_that_already_finished():
    """A retry must not pay for output that is already on disk."""
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("reuse") as base:
            d = recording(base / "2026-08-08_00-08-15", "x" * 100)
            (d / "transcript.edited.md").write_text("ALREADY EDITED", encoding="utf-8")
            publish.run(configured(api), d, "x" * 100, "demo")
            calls = [r for r in api.requests if "chat/completions" in r["path"]]
            print(f"  {len(calls)} OpenAI call(s) after a completed stage 1")
            assert len(calls) == 1, f"re-ran a finished stage ({len(calls)} calls)"
            assert "ALREADY EDITED" in calls[0]["body"]["messages"][0]["content"], \
                "the documentation stage was not fed the existing edit"
            assert (d / "transcript.edited.md").read_text() == "ALREADY EDITED", \
                "overwrote the earlier output"
    finally:
        api.close()


def test_failure_is_recorded_in_meta():
    """A background job whose only report is the status line leaves nothing behind
    when the app is closed — which is exactly how the first real failure was lost."""
    routes = pipeline_routes()
    routes[("POST", "/v1/chat/completions")] = [(401, {"error": {"message": "bad key"}})]
    api = FakeAPI(routes)
    try:
        with workdir("trace") as base, configured_env(api):
            d = recording(base / "2026-08-08_01-00-00", "x" * 100)
            A._publish_async(fake_state(base), d, "x" * 100, "demo").join(60)
            meta = json.loads((d / "meta.json").read_text())
            print(f"  publish_error={meta.get('publish_error')!r}")
            assert "bad key" in (meta.get("publish_error") or ""), \
                "the failure left no trace on disk"
            assert "published" not in meta
    finally:
        api.close()


def test_manual_publish_says_why_it_did_nothing():
    """[u] must explain itself: a keypress that does nothing and says nothing reads as
    a broken app."""
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("manual") as base, configured_env(api):
            state = fake_state(base)

            short = recording(base / "2026-08-08_02-00-00", "tiny", name="s")
            A._publish_existing(state, {"dir": short, "name": "s"})
            print(f"  short: {state.status!r}")
            assert "shorter" in state.status.lower(), state.status

            done = recording(base / "2026-08-08_03-00-00", "x" * 100, name="d",
                             published={"url": "https://example/commit/abc", "at": "t"})
            A._publish_existing(state, {"dir": done, "name": "d"})
            print(f"  published: {state.status!r}")
            assert "already published" in state.status.lower(), state.status

            missing = base / "2026-08-08_04-00-00"
            missing.mkdir()
            A._publish_existing(state, {"dir": missing, "name": "m"})
            print(f"  no transcript: {state.status!r}")
            assert "no transcript" in state.status.lower(), state.status

            assert api.requests == [], "a skipped publish still called out"
    finally:
        api.close()


def test_manual_publish_runs_the_pipeline():
    api = FakeAPI(pipeline_routes())
    try:
        with workdir("manualrun") as base, configured_env(api):
            d = recording(base / "2026-08-08_05-00-00", "x" * 100, name="stand-up")
            state = fake_state(base)
            thread = A._publish_existing(state, {"dir": d, "name": "stand-up"})
            assert thread is not None, "[u] refused a publishable recording"
            thread.join(60)
            meta = json.loads((d / "meta.json").read_text())
            print(f"  status={state.status!r}")
            assert "Published" in state.status
            assert meta["published"]["url"].endswith("/commit/new-sha")
    finally:
        api.close()


def test_saving_a_transcript_does_not_publish():
    """Saving a transcript used to start a publish on its own — two paid model calls
    nobody asked for. [u] is now the only trigger, so this call site must stay empty."""
    api = FakeAPI(pipeline_routes())
    calls = []
    real = A._publish_async
    A._publish_async = lambda *a, **kw: calls.append(a) or None
    try:
        with workdir("nopublish") as base, configured_env(api):
            d = base / "2026-08-08_06-00-00"
            d.mkdir()
            state = types.SimpleNamespace(status="", base_path=base, recordings=[],
                                          jobs=[], jobs_lock=threading.Lock())
            # Everything _save_and_open needs is on the job, settled when the user
            # asked for the work: it no longer falls back to whatever is on the state
            # by the time it runs.
            job = A.Job("transcribe", d, "stand-up", language="tr", name="stand-up",
                        open_app=None, base_path=base, audio_path=d / "audio.wav")
            A._save_and_open(state, job, "x" * 5000)
            print(f"  status={state.status!r}, publish calls={len(calls)}")
            assert (d / "transcription.txt").exists(), "did not even save"
            assert calls == [], "saving a transcript started a publish"
            assert api.requests == [], "reached the API while only saving"
            assert "published" not in json.loads((d / "meta.json").read_text())
    finally:
        A._publish_async = real
        api.close()


TESTS = ["test_config_is_none_without_keys", "test_config_reads_dotenv",
         "test_exported_variable_beats_dotenv", "test_bad_min_chars_falls_back",
         "test_render_prompt_substitutes_the_placeholder",
         "test_render_prompt_appends_when_there_is_no_placeholder",
         "test_openai_returns_the_message_content", "test_openai_truncation_raises",
         "test_openai_error_status_raises", "test_server_errors_are_retried",
         "test_folder_name", "test_publish_makes_one_commit_with_every_file",
         "test_publish_into_an_empty_repository",
         "test_publish_into_a_never_committed_repository",
         "test_publish_reports_a_failed_step",
         "test_run_writes_both_files_and_publishes",
         "test_truncation_writes_nothing_and_publishes_nothing",
         "test_push_failure_keeps_the_model_output",
         "test_gate_skips_short_transcripts_and_republishing",
         "test_gate_reports_missing_configuration",
         "test_saving_a_transcript_does_not_publish",
         "test_end_to_end_updates_status_and_meta",
         "test_a_failure_is_reported_on_the_status_line",
         "test_repo_accepts_what_people_actually_paste",
         "test_run_reuses_stages_that_already_finished",
         "test_failure_is_recorded_in_meta",
         "test_manual_publish_says_why_it_did_nothing",
         "test_manual_publish_runs_the_pipeline"]

if __name__ == "__main__":
    run(TESTS, globals())

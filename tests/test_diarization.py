"""The diarizer against the pyannote 4 interface. The return type is checked with
pyannote's own classes, so a double cannot keep an older shape passing.
"""
import contextlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from support import REPO_ROOT, A, run, workdir, write_wav


@contextlib.contextmanager
def fake_pyannote(pipeline_class):
    saved_modules = {k: sys.modules.get(k) for k in ("pyannote", "pyannote.audio")}
    saved_diar = A._diar_pipe
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = pipeline_class
    sys.modules["pyannote"] = types.ModuleType("pyannote")
    sys.modules["pyannote.audio"] = fake_audio
    A._diar_pipe = None
    try:
        yield
    finally:
        A._diar_pipe = saved_diar
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@contextlib.contextmanager
def diarizer_unavailable(home, reason):
    """Every diarize() raises `reason`; problems are logged to a file under `home`."""
    def no_diarizer(*a, **kw):
        raise RuntimeError(reason)

    saved = A.diarize, A.LOG_PATH
    A.diarize = no_diarizer
    A.LOG_PATH = home / "audiocript.log"
    try:
        yield A.LOG_PATH
    finally:
        A.diarize, A.LOG_PATH = saved


def labeling_state(home):
    return A._TuiState({"base_path": str(home / "recordings"), "language": "tr",
                        "open_app": None, "capture_system_audio": True, "diarize": True})


def test_diarize_reads_the_turns_of_a_pyannote_4_output():
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
    from pyannote.core import Annotation, Segment

    overlapping = Annotation()
    overlapping[Segment(1.0, 3.0)] = "SPEAKER_01"
    overlapping[Segment(0.0, 1.5)] = "SPEAKER_00"
    exclusive = Annotation()
    exclusive[Segment(0.0, 1.0)] = "SPEAKER_00"
    exclusive[Segment(1.0, 3.0)] = "SPEAKER_01"

    saved = A._diar_pipe
    A._diar_pipe = lambda path: DiarizeOutput(
        speaker_diarization=overlapping, exclusive_speaker_diarization=exclusive)
    try:
        turns = A.diarize("x.wav")
    finally:
        A._diar_pipe = saved

    print(f"  turns: {turns}")
    assert turns == [(0.0, 1.5, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")], turns


def test_the_configured_token_reaches_pyannote():
    received = []

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(name, token=None):
            received.append((name, token))
            return types.SimpleNamespace(to=lambda device: None)

    with fake_pyannote(FakePipelineClass):
        A._ensure_diarizer({"hf_token": "hf_from_config"})

    print(f"  from_pretrained received: {received}")
    assert received == [("pyannote/speaker-diarization-3.1", "hf_from_config")], received


def test_a_gated_repo_becomes_an_error_that_says_what_to_do():
    """pyannote 4 raises the hub's own error for a missing token or unaccepted terms
    and never returns None, so the hint has to come from catching it."""
    import httpx
    from huggingface_hub.errors import GatedRepoError

    url = "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin"

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(name, token=None):
            raise GatedRepoError(
                "401 Client Error",
                response=httpx.Response(401, request=httpx.Request("GET", url)))

    with fake_pyannote(FakePipelineClass):
        try:
            A._ensure_diarizer({})
        except RuntimeError as e:
            message = str(e)
        else:
            raise AssertionError("a gated repo did not raise")

    print(f"  message: {message}")
    for expected in ("hf_token", "HF_TOKEN",
                     "huggingface.co/pyannote/speaker-diarization-3.1",
                     "huggingface.co/pyannote/segmentation-3.0",
                     "huggingface.co/pyannote/speaker-diarization-community-1"):
        assert expected in message, f"{expected!r} missing from: {message}"


def test_a_recording_without_speaker_labels_says_why():
    reason = "no Hugging Face token in this test"
    real_plain = A._plain_transcript
    A._plain_transcript = lambda job: "the transcript"
    try:
        with workdir("labels-off-recording") as home, \
                diarizer_unavailable(home, reason) as log:
            state = labeling_state(home)
            d = home / "recordings" / "2026-10-07_10-59-16"
            d.mkdir(parents=True)
            write_wav(d / "audio.wav", np.zeros(16000, dtype=np.int16))
            write_wav(d / "system.wav", np.zeros(16000, dtype=np.int16))
            job = A.Job("transcribe", d, "techexecs", language="tr", name="techexecs",
                        open_app=None, base_path=state.base_path,
                        audio_path=d / "audio.wav", cfg=state.cfg,
                        steps=["diarize", "transcribe"])
            A._job_add(state, job)
            A._transcribe_and_save(state, job, diarize_system=True)
            logged = log.read_text(encoding="utf-8") if log.exists() else ""
    finally:
        A._plain_transcript = real_plain

    print(f"  job={job.state!r} message={job.message!r}")
    assert job.state == A._JOB_DONE, f"the transcript was not saved: {job.message!r}"
    assert job.message == ("Saved “techexecs” · Speaker labels off "
                           "(diarization unavailable: no Hugging Face token in this test)"), (
        f"the finished job does not say it saved and why it has no labels: "
        f"{job.message!r}")
    assert reason in logged, f"the reason never reached the log: {logged!r}"


def test_an_import_without_speaker_labels_says_why():
    reason = "no Hugging Face token in this test"
    real = A._extract_audio, A._media_duration, A.transcribe_segments
    A._extract_audio = lambda src, dst, duration=None, on_pct=None: Path(dst).touch()
    A._media_duration = lambda p: 12.0
    A.transcribe_segments = lambda *a, **kw: [{"start": 0.0, "end": 1.0,
                                                "text": "merhaba"}]
    try:
        with workdir("labels-off-import") as home, \
                diarizer_unavailable(home, reason) as log:
            state = labeling_state(home)
            d = home / "recordings" / "2026-10-07_14-00-00"
            d.mkdir(parents=True)
            job = A.Job("import", d, "clip.mp4", language="tr", name="clip",
                        open_app=None, base_path=state.base_path,
                        audio_path=d / "audio.wav", cfg=state.cfg,
                        steps=["extract", "transcribe", "diarize"])
            A._job_add(state, job)
            A._import_worker(state, job, str(home / "clip.mp4"), True)
            logged = log.read_text(encoding="utf-8") if log.exists() else ""
    finally:
        A._extract_audio, A._media_duration, A.transcribe_segments = real

    print(f"  job={job.state!r} message={job.message!r}")
    assert job.state == A._JOB_DONE, f"the transcript was not saved: {job.message!r}"
    assert job.message == ("Saved “clip” · Speaker labels off "
                           "(diarization unavailable: no Hugging Face token in this test)"), (
        f"the finished job does not say it saved and why it has no labels: "
        f"{job.message!r}")
    assert reason in logged, f"the reason never reached the log: {logged!r}"


def test_a_save_without_speaker_labels_asked_for_says_only_saved():
    real = (A._plain_transcript, A.transcribe_audio, A._extract_audio,
            A._media_duration)
    A._plain_transcript = lambda job: "the transcript"
    A.transcribe_audio = lambda *a, **kw: "imported text"
    A._extract_audio = lambda src, dst, duration=None, on_pct=None: Path(dst).touch()
    A._media_duration = lambda p: 12.0
    try:
        with workdir("no-labels-asked") as home:
            state = labeling_state(home)
            rec_dir = home / "recordings" / "2026-10-07_09-00-00"
            imp_dir = home / "recordings" / "2026-10-07_09-30-00"
            rec_dir.mkdir(parents=True)
            imp_dir.mkdir(parents=True)
            write_wav(rec_dir / "audio.wav", np.zeros(16000, dtype=np.int16))
            recording = A.Job("transcribe", rec_dir, "stand-up", language="tr",
                              name="stand-up", open_app=None, base_path=state.base_path,
                              audio_path=rec_dir / "audio.wav", cfg=state.cfg,
                              steps=["transcribe"])
            imported = A.Job("import", imp_dir, "clip.mp4", language="tr", name="clip",
                             open_app=None, base_path=state.base_path,
                             audio_path=imp_dir / "audio.wav", cfg=state.cfg,
                             steps=["extract", "transcribe"])
            A._job_add(state, recording)
            A._job_add(state, imported)
            A._transcribe_and_save(state, recording, diarize_system=False)
            A._import_worker(state, imported, str(home / "clip.mp4"), False)
    finally:
        (A._plain_transcript, A.transcribe_audio, A._extract_audio,
         A._media_duration) = real

    messages = [recording.message, imported.message]
    print(f"  messages: {messages}")
    assert messages == ["Saved “stand-up”", "Saved “clip”"], messages


_METRICS_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import audiocript
from opentelemetry.sdk.metrics.export import MetricExportResult
from pyannote.audio.telemetry import metrics as telemetry
exports = []
telemetry.exporter.export = lambda data, **kw: exports.append(data) or MetricExportResult.SUCCESS
telemetry.track_model_init(object())
telemetry.provider.shutdown()
print("exports", len(exports))
"""


def test_pyannote_sends_no_usage_metrics():
    """Runs in its own interpreter with the shell's PYANNOTE_METRICS_ENABLED removed, so
    it measures the app's default and not the caller's setting. The recorder replaces
    the exporter until that interpreter exits, so a failing run sends nothing either."""
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    result = subprocess.run([sys.executable, "-c", _METRICS_PROBE, str(REPO_ROOT)],
                            env=env, capture_output=True, text=True, timeout=300)
    lines = result.stdout.strip().splitlines()
    print(f"  probe: {lines[-1] if lines else result.stderr.strip()[-300:]}")
    assert result.returncode == 0, result.stderr.strip()[-2000:]
    assert lines and lines[-1] == "exports 0", f"pyannote exported metrics: {lines[-1:]}"


if __name__ == "__main__":
    run(["test_diarize_reads_the_turns_of_a_pyannote_4_output",
         "test_the_configured_token_reaches_pyannote",
         "test_a_gated_repo_becomes_an_error_that_says_what_to_do",
         "test_a_recording_without_speaker_labels_says_why",
         "test_an_import_without_speaker_labels_says_why",
         "test_a_save_without_speaker_labels_asked_for_says_only_saved",
         "test_pyannote_sends_no_usage_metrics"], globals())

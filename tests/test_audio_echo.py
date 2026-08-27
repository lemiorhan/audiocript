"""Align separately captured microphone and system audio into one time base."""
import contextlib
import importlib
import io
from pathlib import Path

import numpy as np

from support import run, wav_samples, workdir, write_wav
import audio_echo as E


class SubtractProcessor:
    def __init__(self):
        self.calls = []

    def process(self, near, far):
        self.calls.append((near.copy(), far.copy()))
        return np.clip(near.astype(np.int32) - far.astype(np.int32),
                       -32768, 32767).astype(np.int16)


def _native_aec_available(importer=importlib.import_module):
    """Report whether the optional native AEC integration can be exercised."""
    try:
        importer("pywebrtc_audio")
    except ImportError:
        print("  pywebrtc_audio unavailable — native AEC integration test skipped")
        return False
    return True


def test_mic_prefix_is_removed_when_mic_started_first():
    """A wrong leading-source choice would retain the mic's pre-system audio."""
    with workdir("align-mic-first") as d:
        mic = np.arange(8000, dtype=np.int16)
        system = (10000 + np.arange(6000)).astype(np.int16)
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", system)
        aligned_mic = d / "aligned-mic.wav"
        aligned_system = d / "aligned-system.wav"
        # system starts 250 ms later, so 4,000 microphone frames are skipped.
        n = E.align_wav_pair(
            d / "mic.wav", d / "system.wav", aligned_mic, aligned_system,
            mic_started_ns=1_000_000_000,
            system_started_ns=1_250_000_000)
        assert n == 4000
        assert np.array_equal(wav_samples(aligned_mic), mic[4000:8000])
        assert np.array_equal(wav_samples(aligned_system), system[:4000])


def test_system_prefix_is_removed_when_system_started_first():
    """A reversed timestamp delta would retain the system's pre-mic audio."""
    with workdir("align-system-first") as d:
        mic = np.arange(6000, dtype=np.int16)
        system = (10000 + np.arange(8000)).astype(np.int16)
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", system)
        aligned_mic = d / "aligned-mic.wav"
        aligned_system = d / "aligned-system.wav"
        # mic starts 125 ms later, so 2,000 system frames are skipped.
        n = E.align_wav_pair(
            d / "mic.wav", d / "system.wav", aligned_mic, aligned_system,
            mic_started_ns=1_125_000_000,
            system_started_ns=1_000_000_000)
        assert n == 6000
        assert np.array_equal(wav_samples(aligned_mic), mic)
        assert np.array_equal(wav_samples(aligned_system), system[2000:8000])


def test_missing_timestamps_keep_both_prefixes_and_truncate_to_shorter_file():
    """Legacy captures without timestamps must preserve their original alignment."""
    with workdir("align-legacy") as d:
        mic = np.arange(5000, dtype=np.int16)
        system = (10000 + np.arange(4000)).astype(np.int16)
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", system)
        aligned_mic = d / "aligned-mic.wav"
        aligned_system = d / "aligned-system.wav"
        n = E.align_wav_pair(d / "mic.wav", d / "system.wav",
                             aligned_mic, aligned_system)
        assert E.leading_frame_offsets(None, None, 16000) == (0, 0)
        assert n == 4000
        assert np.array_equal(wav_samples(aligned_mic), mic[:4000])
        assert np.array_equal(wav_samples(aligned_system), system)


def test_alignment_copies_in_bounded_chunks_and_reports_cumulative_progress():
    """A whole-file copy or per-chunk progress callback would break this contract."""
    with workdir("align-chunks") as d:
        mic = np.arange(4500, dtype=np.int16)
        system = (10000 + np.arange(4500)).astype(np.int16)
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", system)
        seen = []
        n = E.align_wav_pair(d / "mic.wav", d / "system.wav",
                             d / "aligned-mic.wav", d / "aligned-system.wav",
                             chunk_frames=2000, on_progress=seen.append)
        assert n == 4500
        assert seen == [2000, 4000, 4500]


def test_echo_cancellation_streams_all_chunks_through_one_processor():
    """Recreating the processor or skipping/reordering a chunk would corrupt AEC state."""
    with workdir("cancel-echo-chunks") as d:
        mic = np.array([32767, 1200, -1400, 600, 55, -90, 30000, -30000,
                        7, 8], dtype=np.int16)
        system = np.array([-100, 200, -400, 900, -55, 10, -30000, 30000,
                           2, 10], dtype=np.int16)
        expected = np.array([32767, 1000, -1000, -300, 110, -100, 32767,
                             -32768, 5, -2], dtype=np.int16)
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", system)
        created = []

        def factory(sample_rate):
            processor = SubtractProcessor()
            created.append((sample_rate, processor))
            return processor

        progress = []
        output = d / "clean.wav"
        n = E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                          processor_factory=factory, chunk_frames=3,
                          on_progress=progress.append)

        assert n == len(mic)
        assert len(created) == 1
        assert created[0][0] == 16000
        calls = created[0][1].calls
        assert [len(near) for near, _ in calls] == [3, 3, 3, 1]
        assert all(len(near) == len(far) for near, far in calls)
        assert np.array_equal(np.concatenate([near for near, _ in calls]), mic)
        assert np.array_equal(np.concatenate([far for _, far in calls]), system)
        assert np.array_equal(wav_samples(output), expected)
        assert progress == [3, 6, 9, 10]


def test_echo_cancellation_removes_partial_output_when_processor_fails():
    """A processor exception on a later chunk must not publish partial audio."""
    with workdir("cancel-echo-failure") as d:
        samples = np.arange(8, dtype=np.int16)
        write_wav(d / "mic.wav", samples)
        write_wav(d / "system.wav", samples)
        output = d / "clean.wav"

        class FailingProcessor:
            def __init__(self):
                self.call_count = 0

            def process(self, near, far):
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("processor failed")
                return near.copy()

        try:
            E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                          processor_factory=lambda _: FailingProcessor(),
                          chunk_frames=3)
        except RuntimeError as error:
            assert str(error) == "processor failed"
        else:
            raise AssertionError("processor failure was not propagated")

        assert not output.exists()
        assert not (d / "clean.wav.part").exists()


def test_echo_cancellation_rejects_unequal_input_lengths():
    """Ignoring extra frames would silently publish an incomplete source pair."""
    with workdir("cancel-echo-length-mismatch") as d:
        write_wav(d / "mic.wav", np.arange(6, dtype=np.int16))
        write_wav(d / "system.wav", np.arange(7, dtype=np.int16))
        output = d / "clean.wav"
        try:
            E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                          processor_factory=lambda _: SubtractProcessor())
        except ValueError:
            pass
        else:
            raise AssertionError("unequal WAV lengths were accepted")
        assert not output.exists()
        assert not (d / "clean.wav.part").exists()


def test_echo_cancellation_rejects_malformed_processor_output():
    """Wrong output dtype, dimensions, or length must not create malformed audio."""
    bad_outputs = [
        np.arange(4, dtype=np.int32),
        np.arange(3, dtype=np.int16),
        np.arange(4, dtype=np.int16).reshape(4, 1),
    ]
    for index, bad_output in enumerate(bad_outputs):
        with workdir(f"cancel-echo-bad-output-{index}") as d:
            samples = np.arange(4, dtype=np.int16)
            write_wav(d / "mic.wav", samples)
            write_wav(d / "system.wav", samples)
            output = d / "clean.wav"

            class MalformedProcessor:
                def process(self, near, far):
                    return bad_output

            try:
                E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                              processor_factory=lambda _: MalformedProcessor())
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"malformed processor output {index} was accepted")
            assert not output.exists()
            assert not (d / "clean.wav.part").exists()


def test_echo_cancellation_rejects_wrong_wav_format():
    """Passing non-16-kHz PCM to a 16-kHz processor would corrupt its timing."""
    with workdir("cancel-echo-wav-format") as d:
        samples = np.arange(4, dtype=np.int16)
        write_wav(d / "mic.wav", samples, rate=8000)
        write_wav(d / "system.wav", samples)
        output = d / "clean.wav"
        try:
            E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                          processor_factory=lambda _: SubtractProcessor())
        except ValueError:
            pass
        else:
            raise AssertionError("wrong WAV format was accepted")
        assert not output.exists()
        assert not (d / "clean.wav.part").exists()


def test_native_aec_guard_skips_when_optional_module_cannot_import():
    """An unavailable optional module must skip only the native integration test."""
    def unavailable(module_name):
        assert module_name == "pywebrtc_audio"
        raise ModuleNotFoundError("No module named 'pywebrtc_audio'")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert not _native_aec_available(importer=unavailable)
    assert output.getvalue() == (
        "  pywebrtc_audio unavailable — native AEC integration test skipped\n")


def test_native_aec_reduces_echo_and_preserves_near_end_audio():
    """Disabling AEC or enabling destructive processing would violate its purpose."""
    if not _native_aec_available():
        return

    sample_rate = 16000
    total_frames = 8 * sample_rate
    rng = np.random.default_rng(20260826)
    smoothing = np.ones(7, dtype=np.float64) / 7

    far_float = np.convolve(rng.standard_normal(total_frames), smoothing,
                            mode="same") * 6000
    far = np.clip(np.rint(far_float), -32768, 32767).astype(np.int16)
    far[6 * sample_rate:] = 0

    echo = np.zeros(total_frames, dtype=np.float64)
    direct_delay = int(0.080 * sample_rate)
    later_delay = direct_delay + int(0.012 * sample_rate)
    echo[direct_delay:] += 0.45 * far[:-direct_delay]
    echo[later_delay:] += 0.15 * far[:-later_delay]

    near_start = 6 * sample_rate
    near_noise = np.convolve(rng.standard_normal(total_frames - near_start),
                             smoothing, mode="same") * 6000
    echo[near_start:] += near_noise
    mic = np.clip(np.rint(echo), -32768, 32767).astype(np.int16)

    with workdir("cancel-echo-native") as d:
        write_wav(d / "mic.wav", mic)
        write_wav(d / "system.wav", far)
        output = d / "clean.wav"
        n = E.cancel_echo(d / "mic.wav", d / "system.wav", output,
                          chunk_frames=sample_rate)
        clean = wav_samples(output)

    assert n == total_frames
    assert len(clean) == total_frames

    echo_slice = slice(2 * sample_rate, 6 * sample_rate)
    near_slice = slice(int(6.5 * sample_rate), 8 * sample_rate)

    def rms(samples):
        values = samples.astype(np.float64)
        return np.sqrt(np.mean(values * values))

    raw_echo_rms = rms(mic[echo_slice])
    clean_echo_rms = rms(clean[echo_slice])
    raw_near_rms = rms(mic[near_slice])
    clean_near_rms = rms(clean[near_slice])

    assert 20 * np.log10(clean_echo_rms / raw_echo_rms) <= -6.0
    assert clean_near_rms / raw_near_rms >= 0.5


def test_readme_describes_echo_cancellation():
    """Removing AEC setup guidance would leave users without its fallback path."""
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    readme = readme.casefold()
    assert "acoustic echo cancellation" in readme
    assert "headphones" in readme
    assert "pywebrtc-audio" in readme


if __name__ == "__main__":
    run(["test_mic_prefix_is_removed_when_mic_started_first",
         "test_system_prefix_is_removed_when_system_started_first",
         "test_missing_timestamps_keep_both_prefixes_and_truncate_to_shorter_file",
         "test_alignment_copies_in_bounded_chunks_and_reports_cumulative_progress",
         "test_echo_cancellation_streams_all_chunks_through_one_processor",
         "test_echo_cancellation_removes_partial_output_when_processor_fails",
         "test_echo_cancellation_rejects_unequal_input_lengths",
         "test_echo_cancellation_rejects_malformed_processor_output",
         "test_echo_cancellation_rejects_wrong_wav_format",
         "test_native_aec_guard_skips_when_optional_module_cannot_import",
         "test_native_aec_reduces_echo_and_preserves_near_end_audio",
         "test_readme_describes_echo_cancellation"],
        globals())

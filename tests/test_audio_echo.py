"""Align separately captured microphone and system audio into one time base."""
import numpy as np

from support import run, wav_samples, workdir, write_wav
import audio_echo as E


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


if __name__ == "__main__":
    run(["test_mic_prefix_is_removed_when_mic_started_first",
         "test_system_prefix_is_removed_when_system_started_first",
         "test_missing_timestamps_keep_both_prefixes_and_truncate_to_shorter_file",
         "test_alignment_copies_in_bounded_chunks_and_reports_cumulative_progress"],
        globals())

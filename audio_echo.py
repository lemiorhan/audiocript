"""Streaming alignment for separately captured microphone and system WAVs."""
import os
from pathlib import Path
import wave


_NANOSECONDS_PER_SECOND = 1_000_000_000


def leading_frame_offsets(mic_started_ns, system_started_ns, sample_rate):
    """Return leading frames to drop from microphone and system streams.

    Older capture manifests have no monotonic timestamps.  In that case their
    historical behaviour is preserved by keeping both prefixes.
    """
    if mic_started_ns is None or system_started_ns is None:
        return 0, 0
    delta_ns = int(system_started_ns) - int(mic_started_ns)
    frames = max(0, round(abs(delta_ns) * sample_rate / _NANOSECONDS_PER_SECOND))
    return (frames, 0) if delta_ns > 0 else (0, frames)


def _open_validated_wav(path, sample_rate):
    """Open one required PCM source, rejecting formats this module cannot align."""
    wf = wave.open(str(path), "rb")
    try:
        valid = (wf.getnchannels() == 1 and wf.getsampwidth() == 2
                 and wf.getframerate() == sample_rate and wf.getcomptype() == "NONE")
        if not valid:
            raise ValueError(
                f"{path} must be mono signed 16-bit PCM at {sample_rate} Hz")
        return wf
    except BaseException:
        wf.close()
        raise


def _write_wav_header(wf, sample_rate):
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)


def align_wav_pair(mic_path, system_path, aligned_mic_path, aligned_system_path,
                   mic_started_ns=None, system_started_ns=None, sample_rate=16000,
                   chunk_frames=1 << 20, on_progress=None):
    """Stream the common time overlap of two mono signed 16-bit PCM WAVs.

    Data is written to sibling ``.part`` files and only replaces destinations
    once both WAV writers have been closed successfully.  The return value is
    the number of frames in each aligned output.
    """
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    mic_out = Path(aligned_mic_path)
    system_out = Path(aligned_system_path)
    mic_part = Path(f"{mic_out}.part")
    system_part = Path(f"{system_out}.part")
    mic_part.unlink(missing_ok=True)
    system_part.unlink(missing_ok=True)

    try:
        with _open_validated_wav(mic_path, sample_rate) as mic, \
             _open_validated_wav(system_path, sample_rate) as system:
            mic_skip, system_skip = leading_frame_offsets(
                mic_started_ns, system_started_ns, sample_rate)
            mic.setpos(min(mic_skip, mic.getnframes()))
            system.setpos(min(system_skip, system.getnframes()))
            frames = min(mic.getnframes() - mic.tell(),
                         system.getnframes() - system.tell())

            with wave.open(str(mic_part), "wb") as aligned_mic, \
                 wave.open(str(system_part), "wb") as aligned_system:
                _write_wav_header(aligned_mic, sample_rate)
                _write_wav_header(aligned_system, sample_rate)
                copied = 0
                while copied < frames:
                    take = min(chunk_frames, frames - copied)
                    mic_chunk = mic.readframes(take)
                    system_chunk = system.readframes(take)
                    if len(mic_chunk) != take * 2 or len(system_chunk) != take * 2:
                        raise ValueError("input WAV ended before its declared frame count")
                    aligned_mic.writeframes(mic_chunk)
                    aligned_system.writeframes(system_chunk)
                    copied += take
                    if on_progress:
                        on_progress(take)

        os.replace(mic_part, mic_out)
        os.replace(system_part, system_out)
        return frames
    except BaseException:
        mic_part.unlink(missing_ok=True)
        system_part.unlink(missing_ok=True)
        raise

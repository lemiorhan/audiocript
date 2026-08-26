# Acoustic Echo Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove speaker audio leaking into the microphone before microphone and system channels are mixed.

**Architecture:** Capture a monotonic start estimate for each raw source, resample as today, then align the two 16 kHz working WAVs on a shared timeline. A focused `audio_echo.py` module streams the aligned microphone and system reference through one WebRTC AEC3 processor instance; `audiocript.py` owns orchestration, fallback, atomic output, and progress.

**Tech Stack:** Python 3.12+, NumPy, standard-library `wave`, `pywebrtc-audio` 0.1.x (native WebRTC AEC3), existing plain-script test harness.

**Spec:** `docs/superpowers/specs/2026-08-26-acoustic-echo-cancellation-design.md`

## Global Constraints

- Apply AEC only when both microphone and system-audio sources contain data.
- Keep raw native-rate capture files until `audio.wav` is atomically finalized.
- Process long recordings with bounded memory; never load a complete recording into RAM.
- Use 16,000 Hz, mono, echo cancellation on, noise suppression off, automatic gain control off, and initial stream delay zero.
- Preserve legacy manifests without `started_ns` by treating their startup skew as zero.
- On every AEC import or processing error, log the exception and create a usable aligned unprocessed mix.
- Do not require an audio device, network access, or listening judgment in automated tests.

---

### Task 1: Persist capture start estimates

**Files:**
- Modify: `audiocript.py:399-638`
- Modify: `tests/test_recording_start.py`

**Interfaces:**
- Produces: `DeviceRecorder.started_ns: int | None`
- Produces: `TapRecorder.started_ns: int | None`
- Produces: each recorder's `manifest()` includes `"started_ns"` only after capture starts.

- [ ] **Step 1: Write failing recorder timestamp tests**

Add deterministic fake-clock/fake-stream tests to `tests/test_recording_start.py`. Patch
`A.time.monotonic_ns` and `A.sd.InputStream`; make the fake stream's `start()` return
normally, then assert `DeviceRecorder.manifest()["started_ns"]` equals the clock value.
For `TapRecorder`, bypass process creation by patching `_begin` with a function that
sets format metadata and returns; assert `start()` records a non-null timestamp after
the helper becomes ready. Also assert an unstarted recorder omits the key.

```python
def test_started_recorders_publish_a_monotonic_start_estimate():
    # FakeInputStream.start succeeds; monotonic_ns returns 12_345.
    mic = A.DeviceRecorder(7)
    mic.start(raw_path)
    assert mic.manifest()["started_ns"] == 12_345

    tap = A.TapRecorder()
    tap._begin = lambda: (setattr(tap, "rate", 48000),
                          setattr(tap, "channels", 2))
    tap.start(tap_raw_path)
    assert tap.manifest()["started_ns"] == 12_345
```

- [ ] **Step 2: Run the timestamp tests and verify RED**

Run: `.venv/bin/python tests/test_recording_start.py`

Expected: FAIL because neither recorder manifest exposes `started_ns`.

- [ ] **Step 3: Implement capture start estimates**

Initialize `started_ns = None` in both constructors. In `DeviceRecorder.start()`, set
it immediately after `_stream.start()` succeeds. In `TapRecorder.start()`, set it
immediately after `_begin()` returns successfully. Reset it to `None` on failed start.
Extend both `manifest()` methods without emitting JSON `null`:

```python
manifest = {"kind": self.KIND, "file": Path(self._raw_path).name,
            "rate": self.rate, "channels": self.channels,
            "dtype": self.RAW_DTYPE}
if self.started_ns is not None:
    manifest["started_ns"] = self.started_ns
return manifest
```

- [ ] **Step 4: Run focused and existing startup tests**

Run: `.venv/bin/python tests/test_recording_start.py`

Expected: all tests PASS and failed recorder starts still leave no raw files.

- [ ] **Step 5: Commit**

```bash
git add audiocript.py tests/test_recording_start.py
git commit -m "feat: timestamp audio capture sources"
```

---

### Task 2: Stream-align microphone and system WAVs

**Files:**
- Create: `audio_echo.py`
- Create: `tests/test_audio_echo.py`
- Modify: `tests/run_all.py` only if its filename glob does not already discover the new test.

**Interfaces:**
- Produces: `leading_frame_offsets(mic_started_ns, system_started_ns, sample_rate) -> tuple[int, int]`
- Produces: `align_wav_pair(mic_path, system_path, aligned_mic_path, aligned_system_path, mic_started_ns=None, system_started_ns=None, sample_rate=16000, chunk_frames=1 << 20, on_progress=None) -> int`
- Requires: input WAVs are mono, signed 16-bit PCM at `sample_rate`.

- [ ] **Step 1: Write failing offset and alignment tests**

Create `tests/test_audio_echo.py` using `support.write_wav` and `support.wav_samples`.
Use ramp arrays so every retained frame is identifiable. Cover microphone starting
250 ms earlier, system starting 125 ms earlier, and both timestamps missing.

```python
def test_mic_prefix_is_removed_when_mic_started_first():
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
```

The system-first test expects 2,000 leading system frames removed. The legacy test
expects no leading removal and truncation to the shorter file.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: import/error failure because `audio_echo.py` does not exist.

- [ ] **Step 3: Implement bounded-memory WAV alignment**

Implement strict WAV-format validation, integer nanosecond-to-frame conversion using
rounding, frame skipping with `setpos`, common-overlap calculation, chunked copies,
and atomic `.part`-then-rename outputs. `leading_frame_offsets` returns `(mic_skip,
system_skip)` and treats a missing timestamp pair as `(0, 0)`.

```python
delta_ns = int(system_started_ns) - int(mic_started_ns)
frames = max(0, round(abs(delta_ns) * sample_rate / 1_000_000_000))
return (frames, 0) if delta_ns > 0 else (0, frames)
```

Call `on_progress(frames_copied)` after each shared chunk. On any exception, remove
both output `.part` files and do not replace completed destinations.

- [ ] **Step 4: Run alignment tests and verify GREEN**

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: all alignment tests PASS.

- [ ] **Step 5: Commit**

```bash
git add audio_echo.py tests/test_audio_echo.py
git commit -m "feat: align captured audio sources"
```

---

### Task 3: Add streaming WebRTC echo cancellation

**Files:**
- Modify: `audio_echo.py`
- Modify: `tests/test_audio_echo.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `default_processor_factory(sample_rate) -> object` whose object has `process(near: np.ndarray, far: np.ndarray) -> np.ndarray`.
- Produces: `cancel_echo(mic_path, system_path, output_path, sample_rate=16000, processor_factory=None, chunk_frames=16000, on_progress=None) -> int`.
- Consumes: aligned mono signed-16-bit WAV paths from Task 2.

- [ ] **Step 1: Write failing processor-streaming tests**

Use an injected processor factory that records construction and calls. Its `process`
returns `near - far` clipped to int16. Feed more than two chunks and assert exactly
one processor was created, chunks are consecutive/equal-length, the returned frame
count equals the input, and the written WAV equals the expected subtraction. Add a
processor that raises on its second call and assert no destination or `.part` remains.

```python
class SubtractProcessor:
    def process(self, near, far):
        self.calls.append((near.copy(), far.copy()))
        return np.clip(near.astype(np.int32) - far.astype(np.int32),
                       -32768, 32767).astype(np.int16)
```

- [ ] **Step 2: Run the streaming tests and verify RED**

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: FAIL because `cancel_echo` and `default_processor_factory` are absent.

- [ ] **Step 3: Implement the AEC wrapper and dependency**

Add `pywebrtc-audio>=0.1.0,<0.2` to `requirements.txt`. Lazily import the native
module so `audiocript.py` can start and fallback can run if the import is broken:

```python
def default_processor_factory(sample_rate):
    from pywebrtc_audio import AudioProcessor
    return AudioProcessor(sample_rate=sample_rate, num_channels=1,
                          echo_cancellation=True,
                          noise_suppression=False,
                          auto_gain_control=False,
                          stream_delay_ms=0)
```

Implement `cancel_echo` with one factory call, equal-size chunk reads, NumPy int16
conversion, shape/dtype validation, exact frame-count enforcement, progress in frames,
and atomic output replacement. Reject WAV format or length mismatches with `ValueError`.

- [ ] **Step 4: Install the locked requirement set and run focused tests**

Run: `.venv/bin/python -m pip install -r requirements.txt`

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: all fake-processor tests PASS.

- [ ] **Step 5: Add and run the deterministic native AEC integration test**

Generate eight seconds at 16 kHz from a fixed random seed. Low-pass the far signal
with a short moving average. Construct the microphone echo with an 80 ms delay, a
0.45 direct coefficient, and a 0.15 coefficient 12 ms later. Keep seconds 0-6
echo-only; set far to zero for seconds 6-8 and put independent filtered noise in the
microphone as near-end speech. Process in one-second chunks through the real factory.

Measure echo RMS over seconds 2-6 and near-only RMS over seconds 6.5-8. Assert:

```python
assert 20 * np.log10(clean_echo_rms / raw_echo_rms) <= -6.0
assert clean_near_rms / raw_near_rms >= 0.5
```

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: PASS without an audio device or network.

- [ ] **Step 6: Commit**

```bash
git add audio_echo.py tests/test_audio_echo.py requirements.txt
git commit -m "feat: stream microphone audio through WebRTC AEC"
```

---

### Task 4: Integrate alignment and AEC into finalization

**Files:**
- Modify: `audiocript.py:18-20,799-887`
- Modify: `tests/test_finalize.py`
- Modify: `tests/support.py`

**Interfaces:**
- Consumes: `audio_echo.align_wav_pair` and `audio_echo.cancel_echo` from Tasks 2 and 3.
- Changes: `_finalize_sources(project_dir, sources, save_channels=False, target_fs=16000, on_progress=None, echo_processor_factory=None)` accepts an injectable factory; `None` selects `audio_echo.default_processor_factory`.
- Preserves: return value is the final `audio.wav` frame count; outputs remain atomic.

- [ ] **Step 1: Write failing finalization behavior tests**

Extend capture fixtures so callers can provide `started_ns`. Add tests for:

1. A fake subtracting AEC processor proves `audio.wav` is built from cleaned mic plus
   system and `mic.wav` stores the cleaned signal.
2. A factory that raises proves finalization succeeds with the aligned original mic,
   writes no stale `.part`, and calls `_log_problem` with the exception.
3. A 250 ms microphone head start proves both saved channels begin at the common
   timestamp and have equal lengths.
4. A progress collector proves values remain within `[0, 1]`, never decrease, and end
   at exactly `1.0` across resample, alignment, AEC, and mix phases.

For tests that intentionally verify the historical resample/mix arithmetic rather
than AEC, inject an identity processor so the expected signal remains explicit:

```python
class IdentityProcessor:
    def process(self, near, far):
        return near.copy()
```

- [ ] **Step 2: Run finalization tests and verify RED**

Run: `.venv/bin/python tests/test_finalize.py`

Expected: new assertions FAIL because finalization neither aligns nor invokes AEC.

- [ ] **Step 3: Refactor finalization into explicit weighted phases**

Import `audio_echo`. After per-source resampling, identify parts by source `kind`.
When both mic and system exist, align them into new `.part` paths, attempt AEC into a
clean-mic `.part`, and select either clean or aligned-original mic for the final mix.
Use fixed monotonic phase ranges:

- resampling: `0.00..0.70`;
- alignment: `0.70..0.75`;
- AEC: `0.75..0.90`;
- two-pass mix: `0.90..1.00`.

If AEC raises, call `_log_problem("acoustic echo cancellation failed; using original microphone", exc)` and continue with aligned original files. Ensure all temporary
paths end in `.part`, are removed on success/failure, and `audio.wav` is still renamed
last. Preserve the existing one-source branch unchanged.

- [ ] **Step 4: Run finalization, recovery, and mixing tests**

Run: `.venv/bin/python tests/test_finalize.py`

Run: `.venv/bin/python tests/test_recovery_startup.py`

Run: `.venv/bin/python tests/test_stream_mix.py`

Expected: all tests PASS; the killed-finalize test retains both raw sources.

- [ ] **Step 5: Commit**

```bash
git add audiocript.py tests/test_finalize.py tests/support.py
git commit -m "feat: remove acoustic echo during finalization"
```

---

### Task 5: Document AEC behavior and verify the application

**Files:**
- Modify: `README.md:34-36,59-64,143-151,214-225,284-297,344`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Documents: automatic combined-source AEC, headphone recommendation, native wheel dependency, aligned/clean saved channels, and safe fallback.

- [ ] **Step 1: Add documentation assertions**

Add a lightweight `test_readme_describes_echo_cancellation` to
`tests/test_audio_echo.py` that reads `README.md` and requires the phrases
`acoustic echo cancellation`, `headphones`, and `pywebrtc-audio` case-insensitively.

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `.venv/bin/python tests/test_audio_echo.py`

Expected: FAIL because README does not yet describe AEC or its dependency.

- [ ] **Step 3: Update user and contributor documentation**

Explain that combined microphone/system recordings automatically remove speaker
leakage at finalization, headphones remain the cleanest option, and failures fall back
without losing a recording. Clarify that `mic.wav` is echo-cleaned when speaker labels
are enabled. Add `pywebrtc-audio` to acknowledgements and add an AEC entry under the
current changelog release.

- [ ] **Step 4: Run syntax, focused, and full-suite verification**

Run: `.venv/bin/python -m py_compile audiocript.py audio_echo.py`

Run: `.venv/bin/python tests/test_audio_echo.py`

Run: `.venv/bin/python tests/run_all.py`

Expected: compilation succeeds; every test file passes with no traceback or warning.

- [ ] **Step 5: Check the final diff and repository state**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only intentional AEC files are modified.

- [ ] **Step 6: Commit**

```bash
git add README.md CHANGELOG.md tests/test_audio_echo.py
git commit -m "docs: explain automatic acoustic echo cancellation"
```

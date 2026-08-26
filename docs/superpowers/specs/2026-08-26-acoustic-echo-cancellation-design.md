# Acoustic Echo Cancellation Design

## Goal

Prevent computer audio played through speakers from being recorded a second time
through the microphone when Audiocript captures microphone and system audio together.
The final recording must retain the original system channel and the local speaker's
voice while suppressing the acoustic copy of the system channel.

## Scope

This change applies only when a recording has both a microphone source and a system
audio source. Microphone-only recordings and imported media keep their current
behavior. The first implementation processes echo during finalization rather than in
the live capture callbacks.

The implementation must preserve Audiocript's crash-recovery model: native-rate raw
audio remains the source of truth until `audio.wav` has been written successfully.

## Chosen Approach

Use WebRTC AEC3 through the `pywebrtc-audio` Python package. Its macOS wheels contain
the native implementation for both arm64 and x86_64, so users do not need Homebrew,
Chromium `depot_tools`, or a multi-gigabyte WebRTC source checkout. Audiocript will
depend on `pywebrtc-audio>=0.1.0,<0.2` and use its `AudioProcessor` with:

- sample rate: 16,000 Hz;
- one channel;
- echo cancellation enabled;
- noise suppression disabled;
- automatic gain control disabled;
- stream delay left at zero initially so AEC3 estimates the residual acoustic delay.

Building the complete upstream WebRTC tree was rejected because its macOS checkout
and build toolchain are disproportionate to this application. A home-grown adaptive
filter was rejected because room reflections, nonlinear speaker response, double-talk,
and clock drift make it materially less reliable than AEC3.

## Capture Timing

Each recorder will expose a monotonic capture-start estimate. The microphone records
it immediately after `InputStream.start()` succeeds. The tap records it immediately
after the helper's format header makes the tap ready. The timestamp is therefore
available when the initial recovery manifest is written and is stored as `started_ns`
in that source's manifest.

The timestamp does not attempt to describe speaker propagation or hardware buffer
latency; AEC3 estimates that residual delay. Its purpose is to remove the much larger
startup skew caused by opening the microphone before compiling or starting the system
tap.

Older interrupted recordings have no `started_ns`. They remain recoverable and use
zero startup skew, which is the current behavior.

## Finalization Pipeline

The raw sources are independently resampled to 16 kHz mono WAV files as today. When
both microphone and system files exist:

1. Determine the later of their `started_ns` values.
2. Drop the corresponding leading samples from the earlier source so sample zero in
   both working streams represents the same approximate wall-clock instant.
3. Limit both working streams to their common overlapping duration.
4. Feed the aligned system samples to AEC3 as the far-end/reference stream and the
   aligned microphone samples as the near-end/capture stream.
5. Write AEC3's output as the microphone working WAV.
6. Mix the cleaned microphone WAV with the unchanged system WAV using the existing
   global peak-protected mixer.

AEC processing must be streaming and bounded-memory. The processor receives
consecutive chunks while retaining one `AudioProcessor` instance for the entire
recording so its adaptive state survives chunk boundaries. Input and output lengths
must match exactly.

When speaker labels are enabled, `mic.wav` contains the echo-cancelled microphone and
`system.wav` contains the aligned, otherwise unchanged system channel. This prevents
the remote speaker's leaked voice from being attributed to `[Me]`.

The final progress display gains an AEC phase between resampling and mixing. Progress
must remain monotonic from zero to one.

## Failure Behavior

AEC is best-effort and must never cost the user a recording. If the native module
cannot be imported, rejects the input, or raises while processing, Audiocript logs the
full error and falls back to mixing the time-aligned original microphone and system
channels. Temporary partial AEC files are removed.

An AEC failure does not alter microphone-only behavior and does not mark finalization
as failed. Failures in the existing resampling or final WAV write path retain their
current recovery semantics.

## Interfaces and Code Organization

AEC and alignment logic will live in a focused `audio_echo.py` module rather than
expanding `audiocript.py` further. It will provide small functions for:

- converting timestamp differences to leading sample offsets;
- producing aligned WAV inputs without loading a recording into memory;
- processing aligned mono PCM through an injected or default AEC processor factory.

`audiocript.py` remains responsible for capture, manifests, finalization orchestration,
progress aggregation, fallback selection, and the final mix. The AEC processor factory
is injectable so failure behavior and streaming state can be tested without mocking
global imports.

## Testing

Automated tests will cover:

- microphone-first and system-first timestamp alignment;
- legacy manifests without timestamps;
- exact output length and chunk-boundary continuity;
- measurable attenuation of a deterministic delayed synthetic echo while preserving
  independently generated near-end speech;
- AEC import or processing failure falling back to an aligned unprocessed mix;
- saved speaker-label channels using the cleaned microphone;
- monotonic progress reporting;
- all existing resampling, mixing, recovery, finalization, and recording-start tests.

The real-AEC integration test may use only deterministic generated PCM and local native
code; it must not require an audio device, network access, or a human listening test.

## Documentation

The README will explain that combined microphone/system recordings automatically use
AEC, that headphones still produce the cleanest and most predictable input, and that
AEC failure safely falls back to the original mix. The dependency list will credit
WebRTC audio processing alongside the existing audio libraries.

## Success Criteria

- After a one-second convergence interval, a deterministic synthetic echo-only segment
  is reduced by at least 6 dB RMS compared with its unprocessed microphone input.
- In a deterministic near-end-only segment, output RMS remains at least 50% of input
  RMS, proving that independent local speech is not removed with the echo.
- Startup skew between microphone and system capture no longer creates a second,
  time-shifted system track in `audio.wav`.
- AEC errors never discard an otherwise usable recording.
- Existing recordings without timing metadata remain recoverable.
- The full automated test suite passes on supported macOS and Python versions.

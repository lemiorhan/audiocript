// system_audio_tap.swift
//
// Captures the macOS system audio mix (all processes) using a Core Audio
// process tap + a private aggregate device, and streams it to stdout as raw
// little-endian float32 PCM. Playback stays AUDIBLE (muteBehavior = .unmuted),
// so unlike the BlackHole approach the user keeps hearing their audio and no
// output rerouting / Multi-Output Device is required.
//
// Protocol:
//   * First stdout line (UTF-8, '\n'-terminated):
//       "samplerate=<int> channels=<int> format=f32le"
//   * Then: continuous interleaved float32 frames until SIGTERM/SIGINT.
//
// Requires macOS 14.4+ (Core Audio process taps) and the process to have the
// "System Audio Recording" (TCC kTCCServiceAudioCapture) permission — the first
// run prompts the user to grant it.
//
// Build:
//   swiftc -O system_audio_tap.swift -o system_audio_tap \
//     -framework CoreAudio -framework AudioToolbox -framework Foundation

import CoreAudio
import AudioToolbox
import Foundation

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

var tapID = AudioObjectID(kAudioObjectUnknown)
var aggID = AudioObjectID(kAudioObjectUnknown)
var ioProcID: AudioDeviceIOProcID?
var writeFailed = false

// Idempotent: a signal, a failed stdout write and the start-up error paths can all
// reach this, and each object must be destroyed exactly once.
let cleanupLock = NSLock()
var cleanedUp = false

func cleanup() {
    cleanupLock.lock()
    defer { cleanupLock.unlock() }
    if cleanedUp { return }
    cleanedUp = true
    if let p = ioProcID, aggID != kAudioObjectUnknown {
        AudioDeviceStop(aggID, p)
        AudioDeviceDestroyIOProcID(aggID, p)
        ioProcID = nil
    }
    if aggID != kAudioObjectUnknown {
        AudioHardwareDestroyAggregateDevice(aggID)
        aggID = AudioObjectID(kAudioObjectUnknown)
    }
    if tapID != kAudioObjectUnknown {
        AudioHardwareDestroyProcessTap(tapID)
        tapID = AudioObjectID(kAudioObjectUnknown)
    }
}

func teardownAndExit() -> Never {
    cleanup()
    exit(0)
}

// 1) Create the tap: global mix of all processes, left audible (unmuted).
let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDesc.muteBehavior = .unmuted
tapDesc.isPrivate = true
tapDesc.name = "AudiocriptTap"

var st = AudioHardwareCreateProcessTap(tapDesc, &tapID)
if st != noErr || tapID == kAudioObjectUnknown {
    fail("AudioHardwareCreateProcessTap failed: \(st)")
}

// 2) Read the tap UID (needed for the aggregate's tap list).
var uidAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyUID,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var tapUIDCF: CFString = "" as CFString
var uidSize = UInt32(MemoryLayout<CFString>.size)
st = withUnsafeMutablePointer(to: &tapUIDCF) {
    AudioObjectGetPropertyData(tapID, &uidAddr, 0, nil, &uidSize, $0)
}
if st != noErr { cleanup(); fail("get tap UID failed: \(st)") }
let tapUID = tapUIDCF as String

// 3) Create a PRIVATE aggregate device that contains the tap.
//    NOTE: tap auto-start MUST be false — enabling it deadlocks
//    AudioDeviceCreateIOProcIDWithBlock; we start the device explicitly instead.
let aggUID = "AudiocriptAgg-" + UUID().uuidString
let desc: [String: Any] = [
    kAudioAggregateDeviceNameKey as String: "AudiocriptAgg",
    kAudioAggregateDeviceUIDKey as String: aggUID,
    kAudioAggregateDeviceIsPrivateKey as String: true,
    kAudioAggregateDeviceIsStackedKey as String: false,
    kAudioAggregateDeviceTapAutoStartKey as String: false,
    kAudioAggregateDeviceTapListKey as String: [
        [
            kAudioSubTapUIDKey as String: tapUID,
            kAudioSubTapDriftCompensationKey as String: true,
        ]
    ],
]
st = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
if st != noErr || aggID == kAudioObjectUnknown {
    cleanup(); fail("create aggregate failed: \(st)")
}

// 4) Get the aggregate's input stream format (rate / channels).
var fmtAddr = AudioObjectPropertyAddress(
    mSelector: kAudioDevicePropertyStreamFormat,
    mScope: kAudioObjectPropertyScopeInput,
    mElement: kAudioObjectPropertyElementMain)
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
st = AudioObjectGetPropertyData(aggID, &fmtAddr, 0, nil, &asbdSize, &asbd)
if st != noErr { cleanup(); fail("get aggregate format failed: \(st)") }
let rate = Int(asbd.mSampleRate.rounded())
let channels = Int(asbd.mChannelsPerFrame)
if rate <= 0 || channels <= 0 { cleanup(); fail("bad format rate=\(rate) ch=\(channels)") }

// 5) Emit the header, then stream PCM from the IOProc.
let header = "samplerate=\(rate) channels=\(channels) format=f32le\n"
FileHandle.standardOutput.write(header.data(using: .utf8)!)

let ioQueue = DispatchQueue(label: "tap.io")
let out = FileHandle.standardOutput
st = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggID, ioQueue) {
    (_, inInputData, _, _, _) in
    if writeFailed { return }
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
    guard abl.count > 0 else { return }
    let buf = abl[0]
    guard let mData = buf.mData, buf.mDataByteSize > 0 else { return }
    do {
        try out.write(contentsOf: Data(bytes: mData, count: Int(buf.mDataByteSize)))
    } catch {
        // The reader is gone (EPIPE). SIGPIPE is ignored so we land here instead of
        // being killed signal-style, and can still tear the tap down. Never do that
        // from inside the IOProc — AudioDeviceStop waits for this block to return —
        // so hand it to the main queue.
        writeFailed = true
        DispatchQueue.main.async { teardownAndExit() }
    }
}
if st != noErr || ioProcID == nil { cleanup(); fail("create IOProc failed: \(st)") }

st = AudioDeviceStart(aggID, ioProcID)
if st != noErr { cleanup(); fail("AudioDeviceStart failed: \(st)") }

// 6) Run until terminated; clean up on signal so the private aggregate/tap go away.
//    Every signal that can realistically reach us is handled, because the default
//    action for all of them is to terminate the process outright — which would leave
//    the process tap and the aggregate device behind for coreaudiod to reap:
//      SIGTERM  the app stopping the recording      SIGINT  Ctrl-C
//      SIGHUP   the terminal window being closed    SIGPIPE the reader going away
signal(SIGPIPE, SIG_IGN)          // surfaces as a write error instead (see IOProc)
let handledSignals: [Int32] = [SIGTERM, SIGINT, SIGHUP]
for sig in handledSignals { signal(sig, SIG_IGN) }
let signalSources: [DispatchSourceSignal] = handledSignals.map { sig in
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { teardownAndExit() }
    src.resume()
    return src
}
_ = signalSources                 // keep the sources alive for the process lifetime
FileHandle.standardError.write("tap streaming: rate=\(rate) channels=\(channels)\n".data(using: .utf8)!)
dispatchMain()

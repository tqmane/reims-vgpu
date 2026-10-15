# WebGL in stock Safari and Chrome

Updated 2026-10-15. On the disposable Tahoe 26.4 guest, **stock Safari 27
and stock Chrome 155 each pass all 22 authored WebGL 1/2 acceptance stages**.
The fix is the [ReimsMetalCompat guest bridge](../guest/metal-compat/README.md).
Neither browser executable is modified. Sustained performance, other OS
versions and production rollout remain separate qualification work.

## Failure and repair

The original x86 Metal plugin exposes a legacy capability/connection path
that current browser ANGLE initialization rejects. Merely selecting Apple's
newer IOGPU-family plugin is insufficient: its kernel requests do not match
the installed IOAccelerator-family kernel driver's records and selectors.
The bridge translates those requests and keeps the real driver's resource
IDs, compiler information and completion notifications.

| Reproduced failure | Repair and check |
|---|---|
| Initialization and resource creation use incompatible records | Translate device information, compiler data, buffers, textures, surfaces and queue records; native source compilation and GPU readback pass. |
| Resource-list lengths differ from allocations | Validate rounded-up KiB lengths, then send the exact allocation length to the legacy driver; boundary and IOSurface readback checks pass. |
| Continuous drawing rejects reused resource chunks | Read the count at offset `0x3e`; bytes at `0x3c` are padding and may be nonzero. Synthetic padding/count tests retain malformed-input rejection. |
| Shared-event command lists do not match the old transport | Preserve render and kernel lists, translate signal/infinite-wait operations, and retain owned command memory until actual completion. Cross-queue future-event ordering passes. |
| Native batching reaches external method 29 | Validate and prepare the complete batch before one legacy submission; track rollback by batch identity. A 64-command burst passes exact readback. |
| Empty command buffers contain only a modern preamble | Produce the measured empty legacy segment and wait for its real asynchronous callback; 32 empty submissions complete. |
| Completion callbacks deadlock or reorder | Drain outside legacy driver locks using one nonconcurrent thread call; four queues over 100 frames pass ten runs. |
| Chrome still selects the old plugin | Match the full `proc_selfname`, including Chrome 155's actual `Google Chrome Helper` name. The `(GPU)` display name is not its executable name on this version. |

Browser-scoped selection covers `com.apple.WebKit.GPU`,
`Google Chrome Helper`, and `Google Chrome Helper (GPU)`. Other processes,
including WindowServer, keep the old plugin. Other Chromium browsers are
**not yet qualified**; different helper names must not be assumed to match.

The bridge does not invent successful GPU completions, enable SwiftShader,
force browser GPU acceptance, disable visual effects, or disable SIP/AMFI.
Unsupported translations fail explicitly. Global activation is not supported:
a WindowServer framebuffer-fetch path remains incompatible with the current
Windows runtime. A host capability experiment failed the cold-boot check and
was withdrawn rather than weakening the validation.

## Acceptance evidence

Both APIs pass context/version checks, GLSL compilation/linking,
triangle/VBO pixels, texture upload, FBO/readPixels, blending, depth,
180 rendered frames, resource lifetime, deliberate loss/recovery and
unexpected context-loss checks. Chrome's live GPU helper was also observed
loading the stock IOGPU-family Metal plugin. Native authored probes verify
compute, managed uploads, memoryless MSAA resolve, IOSurfaces, shared events,
resource reuse, batches and empty-command completion.

The acceptance runs used 1920x1080, 8 vCPUs and 4 GiB guest RAM on a contended
Windows host. Browser callback rates from the 180-frame stage were:

| Browser | WebGL 1 rAF FPS | WebGL 2 rAF FPS |
|---|---:|---:|
| Safari 27 | 19.20 | 30.76 |
| Chrome 155 | 50.68 | 54.77 |

These are requestAnimationFrame intervals, not physical display presents or
a matched performance comparison. They do **not** establish stable 60 FPS.
[TestUFO](https://testufo.com/) remains the requested performance workload.
4K, extended stress and full Khronos conformance remain unqualified.

Context recovery must wait until after dispatch of the loss event, when
WebKit records `preventDefault()`. The repeated-frame fixture reuses a VBO,
keeps the canvas visible and cancels rAF on timeout; independent upload and
lifetime cases remain. Expected pixels and error checks were not relaxed.

## Build, update and rollback

The current supported target is Tahoe 26.4 build 25E246, kernel 25.4.0,
AppleParavirtGPU 15.0.0 and AppleParavirtGPUMetalIOGPUFamily 64.4.7. OpenCore
restricts the experimental kext to that kernel. Do not widen this bound to
install an OS update without checking its driver ABI and repeating a cold
boot and both browser suites on a disposable disk/firmware copy.

Export from the committed source with `source_snapshot.py`, following the
[build instructions](../guest/metal-compat/README.md). Dirty input is refused;
the build uses a private copy, verifies hashes before publication and includes
its source manifest in the signed kext. Pair results with that manifest.
A commit alone does not prevent an external process from rewriting a live
working file after the build. Unexpected writers need separate attribution.

The separately adapted WebKit used during diagnosis is not required for this
stock-browser fix and is not the supported delivery path. Private captures,
raw reports, screenshots and proprietary guest binaries stay outside Git.
Only authored bridge source, probes, synthetic tests and documentation are
published. Preserve the previous engine, disk, firmware and NVRAM for rollback.

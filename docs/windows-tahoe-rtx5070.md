# Windows Tahoe desktop observations on Ryzen 7900X / RTX 5070

This is a single-machine field report from 2026-10-18 through 2026-10-20,
not a performance qualification or a recommended CPU profile for all hosts.
It records a usable desktop configuration and two host-fatal failures in
the pinned renderer below. Raw guest captures, identifiers and Apple shaders
are omitted.

This report is provisional while newer renderer code is tested on Windows.
Upstream metal2vulkan now includes broader float framebuffer blending and
compressed-texture decoding. In particular, commit
[`930d746`](https://github.com/steelbrain/metal2vulkan/commit/930d746df786108cb8b08000070dddcfe04be207)
adds ASTC decoding through sampled twins. A separate Windows integration
now survives the Weather and Maps checks described below, but has visible
rendering and responsiveness problems. The failures below are baseline
observations, not claims about the newest upstream revision.

## Tested configuration

| Component | Observed configuration |
| --- | --- |
| Host | Windows 11 Pro, build 26300 |
| CPU | Ryzen 9 7900X, 12 physical cores / 24 logical processors |
| Cache topology | Two groups of six physical cores, each sharing 32 MiB L3 |
| Memory | 64 GiB installed; 32 GiB assigned to the retained guest |
| GPU | NVIDIA GeForce RTX 5070, 12 GiB, driver 617.42 |
| Storage | Guest QCOW2 on Samsung 980 PRO NVMe; virtual NVMe controller |
| Guest | macOS Tahoe 26.7.1, build 25G241, Darwin 25.6.0 |
| Boot profile | iMac20,1; Haswell-noTSX compatibility CPU; WHPX |
| Retained CPU allocation | 12 vCPUs, one socket, unrestricted process affinity |
| Display | One 1920x1080 display, advertised 60 Hz |
| Reims revision tested | `2627243024415e3fe87fb025f627049fc690a2b4` |
| metal2vulkan revision | `ea5d6b877ea54e45676d199782f3db1b54fbaaa4` |

The tests used a separately built guest Metal compatibility bridge with
global routing enabled through Lilu and `-reimsmetal`. They do not establish
the same behavior for an unmodified Apple guest graphics stack. The working
QEMU executable, host graphics library, OpenCore image, NVRAM, launch profile
and disk backing chain were preserved together. This report does not claim
a boot test of every later Reims commit.

## Desktop progress and limits of attribution

**Observed:** setting the measured `METAL_ERROR_MODE` preference to `5`
improved missing UI content, but did not by itself produce a responsive
desktop. The paired host build and guest compatibility bridge subsequently
produced a clear improvement in window movement and Dock interaction.
The user described a settled twelve-vCPU run as nearly native in ordinary
desktop use before opening Weather. This is subjective acceptance, not a
measured native-performance ratio or a sustained 60-FPS result.

The guest's disk-sleep setting was disabled during the experiment and kept
disabled in later matched comparisons. That timing prevents attributing
the early improvement to CPU placement alone. Windows was already using
its High Performance power scheme; changing it was not an intervention.

The user also reported substantial host and guest lag while interactive
desktop-control tooling was operating. Later diagnostics used background
SSH and logging without screenshots or injected desktop input. This avoids
that observer during comparisons, but its isolated overhead was not measured.

**CPU comparison:** six and twelve vCPUs were tested unrestricted and within
one L3 group. Stackless scheduling traces verified the restricted QEMU
threads executed within that group's logical processors. Process affinity
did not reserve those cores exclusively or restrict Windows' other work.
After startup settled, six vCPUs within a cache group felt better than six
unrestricted; the user ultimately preferred twelve unrestricted overall.
The retained profile therefore restores the preferred existing allocation;
it is not evidence that pinning or increasing vCPUs universally fixes pacing.

Startup activity changed responsiveness for several minutes. A reconstructed
profile also felt worse than the earlier control, so the exact tested disk,
firmware and runtime state were retained rather than treating matching CPU
counts as an identical starting point.

**Measurement limits:** passive guest timer/display-link callbacks and host
PresentMon captures were collected. Display callbacks sometimes stopped
during idle while the timer continued. Weather terminated both matched CPU
runs before their planned end. These interrupted or idle intervals cannot
support a before/after stutter percentage, and a display-link callback is
not proof that a frame reached the physical monitor. No causal CPU-tuning
speedup is claimed from these captures.

## Failure classes in the pinned build

### Weather: blended framebuffer fetch

**Observed:** opening Weather terminated the host Reims/QEMU process, with
the runtime refusing render-pipeline creation as `FramebufferFetch`,
`Needs::NotImplemented`. The same failure class occurred with six and twelve
vCPUs and was present before the CPU experiments. This separates it from
the CPU-placement experiment. It is a host runtime fatal, not merely an
application exit.

**Measured on this host:** a read-only Vulkan physical-device query reports
`fragmentShaderPixelInterlock = VK_TRUE` for the RTX 5070. Consequently this
is not evidence that the GPU lacks pixel interlock or that all framebuffer
fetch is unsupported.

**Read from the pinned source:** the storage-image framebuffer-fetch path
rejects blend combinations it cannot model. Its half-float blend path is
conditional on an exact half-blend policy, rounded math and a compatible
half output; normalized source-over support does not imply arbitrary
half-float fetch/blend support. See metal2vulkan's
`lower/fragment/fetch.rs`, `device/policy.rs` and the runtime's
`framebuffer_fetch.rs`. Reims exposes the opt-in as
`REIMS_VGPU_EXACT=half_blend` in `crates/reims-vgpu-vulkan/src/platform.rs`.
This option's existence alone does not demonstrate that it fixes Weather.

A correct fix needs an authored reproducer, pipeline acceptance and pixel
readback for the exact supported blend class, followed by a cold guest boot
and Weather test. Disabling strict checking or advertising all framebuffer
reads as unavailable is not a validated fix. The earlier capability-gating
experiment documented in the bridge README selected another unsupported
guest path and was withdrawn.

### Maps: ASTC texture format

**Observed:** opening Maps terminated the host while creating a texture with
`MTLPixelFormatASTC_8x8_LDR` (Metal value 212). The runtime refused the
required sampled-image format `VK_FORMAT_ASTC_8x8_UNORM_BLOCK`.

**Measured on this host:** `vkGetPhysicalDeviceFormatProperties` returns
zero optimal-tiling feature flags for both `ASTC_8x8_UNORM_BLOCK` and
`ASTC_8x8_SRGB_BLOCK`. In particular, neither supports sampled images under
this driver. RGBA16Float and BGRA8Unorm sampled-image support is present.
The query creates a Vulkan instance and reads physical-device properties;
it creates no device, queue or rendering workload.

**Inference:** Maps needs format emulation/decoding or a demonstrated valid
alternate guest path on this GPU/driver. CPU affinity, SMBIOS identity and
additional vCPUs cannot supply the missing Vulkan format support. Merely
claiming ASTC support would leave the texture's contents unimplemented.

## Authored offscreen feature probes

`guest/metal-compat/feature-gap-probe.m` contains source written for this
investigation, with no framework shaders, cached metallibs or Apple assets.
Compile it inside a logged-in macOS 15-or-newer graphics session:

```sh
clang -Wall -Wextra -Werror -fobjc-arc -framework Foundation -framework Metal \
  feature-gap-probe.m -o feature-gap-probe
./feature-gap-probe half-rounded
```

Run these only on disposable VM overlays with the working configuration
preserved: an unsupported operation can terminate the entire VM host process.
The probe has no window or input handling. Fetch modes create a pipeline and
submit one offscreen triangle into an 8x8 RGBA16Float attachment in slot 1,
with an unused BGRA8Unorm attachment in slot 0 and source-over blending.
After completion they synchronize the texture and print one pixel's half bits.
These bits are diagnostics, not a cross-device accuracy assertion.

Modes are `half-rounded`, `half-fast`, `float-rounded`, `float-fast` and
`astc`. The first four vary the source/output vector type and Metal math
mode; `astc` creates one private ASTC 8x8 LDR sampled texture and submits a
texture-to-buffer copy to force deferred host creation. It does not upload
assets, draw or assert the unspecified texture contents. Modes not listed as measured below are not validated.

**Measured:** the safe-math `half-rounded` draw terminated the baseline
runtime with the same `FramebufferFetch` / `NotImplemented` class at color
slot 1. With only `REIMS_VGPU_EXACT=half_blend` enabled in a separate test
process, that draw completed and returned pixel bits
`3385,3785,39a4,3b85`. Pipeline creation alone succeeded in the guest under
both configurations; the host refusal appeared on submission, so a
pipeline-creation-only probe would have missed it.

**Measured:** opening Weather with exact half-blend still terminated the
host with `FramebufferFetch` / `NotImplemented`. Thus the authored case
reproduces a relevant failure class but not all of Weather's requirements;
exact half-blend is not a validated Weather workaround and was not adopted
in the working launch profile.

**Measured:** the `half-fast` draw still terminated the exact-half-blend
process with the same named feature refusal. This isolates a smaller
unsupported math/blend combination without assuming Weather's complete
shader form is identical. The `float-*` modes were not run on this baseline.

**Measured:** ASTC texture creation alone returned a guest texture object.
Submitting its texture-to-buffer copy then terminated the host with
`MTLPixelFormatASTC_8x8_LDR` / `VK_FORMAT_ASTC_8x8_UNORM_BLOCK` sampled-image
format refusal, matching the Maps failure class. A creation-only texture
probe would therefore also have missed the deferred host refusal.
The probe compiled under Tahoe's Command Line Tools with the flags above.
This contribution changes no library capability answers, scheduling or
renderer implementation. No library/GPU regression suite was run for it;
the isolated guest probe is its targeted validation.

## Newer renderer trial: crashes improved, rendering still under test

A separate twelve-vCPU VM was cold-booted with a local integration of
upstream metal2vulkan `a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b`
and the Windows host changes. The tested renderer tree was
`836bce913a855b008a629184f7cd14cd5cb6aa78`; its Reims consumer was
`6696bb9dbac87a21a5562fb72245ce4481b8f45e`. These are local integration
revisions, not a released dependency or a change to this PR's renderer pin.
The normal launcher and preferred twelve-vCPU state were left in place.

Weather remained running for at least sixteen minutes. The user checked
other cities and reported that forecast cards and animations looked correct,
with stutters when switching cities. Maps remained running for at least nine
minutes, including manual panning, zooming, satellite view, traffic and 3D.
The user reported poor responsiveness, especially in 3D. Those observations
show that the earlier fatal paths were avoided in this trial; they do not
qualify either app for sustained use.

The user also reported button backgrounds flickering between light and dark
under graphical load, including an earlier occurrence in Calculator. The
cause is unresolved. This remains a correctness problem even when an app
stays open. The candidate has not replaced the preserved desktop build.

Authored native GPU tests on the RTX 5070 passed float framebuffer blending
and ASTC LDR/sRGB decoding with Vulkan validation enabled. The ASTC checks
covered private uploads and shared-texture replacement, nonconstant blocks,
GPU decode execution, sampled pixel comparisons and compressed readback.
Inside the guest, the four small fetch/blend probes and the ASTC submission
probe completed. A separate authored offscreen check completed 480 frames
across private and shared buffer-backed RGBA16Float targets, with two queues
and four frames in flight, without pixel differences greater than 0.003.
That limited blend check did not reproduce the button flicker.

These results do not constitute the renderer's full regression gate or a
before/after pacing benchmark. Command-buffer timing and the flicker still
need investigation. No renderer integration code is included in this PR.

## Boot-picker wait versus guest startup

The tested OpenCore image initially used `Misc/Boot/Timeout=0` and
`AllowSetDefault=false`, requiring manual selection of macOS before SSH
could start. An independent image with a five-second timeout,
`AllowSetDefault=true` and APFS/HFS discovery (`ScanPolicy=0x301`) booted
macOS and reached automatic login without picker input. The previous image
and NVRAM were preserved. A live QEMU process and a forwarded host SSH port
do not prove that the guest has passed its boot picker or started SSH.

## Repeating comparisons safely

1. Shut macOS down cleanly before freezing a checkpoint. Preserve the exact
   engine and library, guest bridge revision, OpenCore, NVRAM and launch
   environment. Hash those artifacts locally; keep machine identities private.
2. Use independent writable sibling overlays over the frozen disk, with
   separate NVRAM copies. Never resume a backing image writable while any
   child depends on it. Preserve the known working launcher.
3. Keep guest power settings, display mode and background load fixed. Let
   login activity settle, then compare first animations after a fixed idle
   period and repeated animations separately.
4. Collect bounded background logs and scheduling/presentation traces. Avoid
   interactive desktop control during the timed interval, and distinguish
   guest callback timing from host presentation and physical display timing.
5. Test Weather and Maps separately after desktop measurements. A host-fatal
   run is a failure/reproduction result, not a completed pacing benchmark.

Retesting these failures against the newer renderer is still in progress.
The setup changes in this PR do not implement either renderer fix. Shared clipboard,
file sharing and elastic memory/CPU allocation are separate integration
work; this experiment does not implement WSL-like resource management or
direct GPU passthrough.

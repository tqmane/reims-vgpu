# Windows native Tahoe: current implementation

Updated 2026-10-15. The development branch retains the working Windows-native
QEMU/WHPX desktop, Apple Account configuration, JIS input, USB selection,
OpenCore controls and VM management. Production rollout remains pending.

## Development branch

`windows-native-tahoe-upstream-sync-20261014` integrates upstream Reims,
its QEMU branch and metal2vulkan while retaining Windows section sharing,
reset/topology handling, SDL3/Vulkan and management scripts. See
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) for exact integration decisions.
The preserved baseline is `68d0a4d7`, with QEMU `c1dbb187` and runtime
`8556e6e4`; candidate checks use separate disk, firmware and NVRAM copies.

The development tree adds a native Windows C harness, authored WebGL and
Metal probes, frame-callback instrumentation, Windows runtime calibration
and the experimental x86 Metal user-client bridge. The bridge selects the
stock IOGPU-family plugin for the tested browser helpers, translating its
requests into the installed legacy kernel driver's ABI. Browser binaries,
SIP and AMFI remain unchanged.

## Verification

- Reims `cargo xtask check`: 2,895 selected tests passed, two existing skips.
- Runtime product gate: 4,543 tests passed, 22 existing ignored cases.
- Native release, QEMU and C harness build/run checks passed.
- Stock Safari 27 and Chrome 155: each passed all 22 GPU WebGL 1/2 stages on
  Tahoe 26.4, including pixels, 180 frames, resource lifetime and loss/recovery.
- Source snapshot tests, actual macOS kext build and signature verification
  passed. A deliberately changed input during compilation was rejected while
  preserving the previous signed output.

Stable 60 FPS, 4K, extended resource/device-loss stress, other browser helper
names and newer macOS driver ABIs remain unqualified. The separate diagnostic
WebKit app is not required for the stock-browser fix. See
[TEST_RESULTS.md](TEST_RESULTS.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and
[WEBGL_DIAGNOSIS.md](WEBGL_DIAGNOSIS.md) for evidence and limits.

# Known issues

Updated 2026-10-15. Stock Safari 27 and Chrome 155 now pass the authored GPU
WebGL 1/2 acceptance suite on the disposable Tahoe 26.4 guest. This does not
qualify every guest/browser version or establish production readiness.

## Experimental driver scope

The bridge is restricted to kernel 25.4.0 and the measured stock driver ABI.
Newer macOS versions require a separate driver/boot/browser qualification;
changing an OpenCore version bound alone is not compatibility work.
Browser selection currently uses full process names for Safari and Chrome.
Other browser helpers are not yet covered or qualified. Global activation
for WindowServer is unsupported. Finite shared-event waits and other
untranslated commands fail explicitly. Resource cleanup under failure and
mip/slice preservation need further stress coverage.

## Display width and stalls

Upstream documents an x86 Tahoe WindowServer/plugin tile-texture failure
above 2048 pixels wide. Initial qualification uses 1920x1080. Higher display
choices remain available with a warning; availability is not proof of stable
1440p or 4K operation.

Restored applications and Simulator services can delay normal shutdown.
Distinguish guest application responsiveness, window/frame progress and QMP
power state before diagnosing a GPU deadlock. A prior shutdown exceeded the
50-second ACPI deadline but subsequently reached QMP shutdown and exited.

## Performance and validation limits

- The reported stable 60 FPS has not yet been reproduced in a qualified,
  matched TestUFO comparison. Safari's initial nominal 60 FPS label also
  displayed a synchronization failure.
- The 180-frame acceptance stages measured 19.20/30.76 rAF FPS in Safari and
  50.68/54.77 in Chrome for WebGL 1/2 on a contended host. These are functional
  checks and browser callback intervals, not physical scanout measurements.
- Frame traces measure guest-frame callbacks. Window statistics measure host
  submission/completion latency; adapter samples include other applications.
- The full macOS behavior-lock suite requires its macOS runner. The Windows
  C harness and runtime tests do not replace it.
- Long-run device loss, all-resolution visual qualification and OS updates
  remain open. See [TEST_RESULTS.md](TEST_RESULTS.md).

## Working-file integrity

A verified bridge source file was observed reverting after a commit while
its known-good build remained intact. The writer is not yet attributed.
Source snapshot export now rejects dirty input, builds use a verified copy,
and the signed bundle records the exact source manifest. This prevents a
mismatched rebuild from inheriting earlier test results; it does not identify
or repair an external writer. Keep protected backups while investigating.

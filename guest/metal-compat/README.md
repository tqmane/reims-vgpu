# Experimental x86 Metal user-client bridge

This Lilu plugin translates the stock IOGPU-family Metal plugin's requests
to the x86 IOAccelerator-family kernel driver. It is under development:
existing Safari 27 and Chrome 155 have passed all 22 GPU WebGL acceptance stages in the disposable lab. Performance and broader qualification remain pending.
Do not install it in the production VM.

The current target is Tahoe 26.4 (25E246), AppleParavirtGPU 15.0.0 and
AppleParavirtGPUMetalIOGPUFamily 64.4.7, on Reims PCI `106b:eeee`.
The transport returns actual legacy resource IDs, compiler information and
GPU completions. It does not override GPU-family queries or emulate a
successful draw. Untranslated requests fail explicitly.

## Build and check

First review and commit changes to this directory, then export a new source
snapshot on the development host:

```sh
python guest/metal-compat/test_source_snapshot.py
python guest/metal-compat/source_snapshot.py export /path/to/bridge-source
```

Export refuses uncommitted source changes and existing destinations. It reads
Git's committed tree rather than copying live editor buffers. Transfer the
whole exported directory, including `source-manifest.json`, to the disposable
macOS guest. Run the commands below from that exported directory. Keep build
output outside it.

The build checks every source hash, compiles a private copy, and checks the
input snapshot again before publishing the signed kext. Failed checks preserve
the previous output. The signed bundle includes the commit and source hashes
in `Contents/Resources/source-manifest.json`; retain this file with test
results. Do not reuse test results for a different source manifest.

Build in the disposable macOS guest, using Python 3, Lilu 1.7.2 source
(`e4748cc081bf060302c7d3c44a643ce1d11b7e1d`) and MacKernelSDK
`3f750085caa17ec3a7880f11c11bf4f48cd6a164`:

```sh
clang++ -std=c++17 -Wall -Wextra -Werror abi-test.cpp -o /path/to/lab/abi-test
/path/to/lab/abi-test
bash build.sh /path/to/Lilu /path/to/MacKernelSDK /path/to/lab/build
clang -fobjc-arc -framework Foundation -framework Metal resource-probe.m -o /path/to/lab/resource-probe
clang -fobjc-arc -framework Foundation -framework Metal compute-probe.m -o /path/to/lab/compute-probe
```

Keep the compiled kext and private driver captures outside Git. Install
the authored kext after Lilu in a separate OpenCore image, restricted to
kernel `25.4.0`. Preserve the original image and all other settings. SIP and
AMFI remain enabled. Without `-reimsmetal`, only the diagnostic connection
type `0x524d0005` is enabled and ordinary apps continue using the old driver.
`-reimsmetal` selects the newer plugin for every client of the selected
Reims accelerator. Connection type 5 is translated independently of process
names, browser brands or helper suffixes. The existing device-info size and
private-data bounds are validated on every backend connection. Native
authored probes can also use the diagnostic connection;
`-reimsmetaloff` disables this Lilu plugin. Neither setting is a production
recommendation until the cold boot and browser checks pass.

The common path includes WindowServer and requires the host runtime to
implement its render operations, including blended framebuffer fetch.
Current global-path qualification is in progress. Historical browser-only
results below do not qualify this mode or additional OS versions.

## Measured progress, 2026-10-15

- CPU-only record translation checks pass, including truncated metadata,
  invalid resource IDs and 64-bit dirty-buffer ranges.
- The kernel bridge allocated and released 3,200 buffers and 100 textures;
  native source compilation and pipeline creation passed.
- Seven buffers over 100 GPU frames passed every 64-value readback.
- Four concurrent queues over 100 frames passed ten consecutive runs after
  serializing completion delivery. The original driver passed the same test.
- Native ANGLE Metal clear, shader, triangle, texture, FBO and readback passed.
- Global activation exposed additional WindowServer calls: memoryless
  texture creation, fences and dirty-resource notifications. Creation of the
  first two now passes; global display activation remains under investigation.
- Managed buffer upload, memoryless MSAA resolve and managed texture upload
  now pass exact GPU readback. The translator handles multiple encoder
  resource lists and variable-length mipmapped texture records.
- A memoryless/managed texture reuse probe passes 100 rounds, sampling the
  managed texture into another render target and checking all 256 pixels.
- Resource-list sizes are validated as rounded-up KiB values; the legacy
  submission receives the exact allocation length. Synthetic boundary and
  overflow checks pass, as does GPU/CPU readback of a 16x17 IOSurface.
- Shared-event signal and infinite wait commands pass a native GPU probe.
  Converted commands use owned shared memory held until actual completion.
- Both stock browsers pass all 22 WebGL 1/2 acceptance stages, including
  continuous rendering, resource lifetime and context loss/recovery.

The dirty-texture notification initially blocked resource release, causing
WindowServer to reuse IDs still held by the kernel. Forwarding its mip/slice
update to the legacy dirty ring removed that failure. The subsequent global
boot reached a blended framebuffer-fetch pipeline refused by the Windows
runtime. An experiment advertising framebuffer reads as unavailable instead
made guest rendering bind memoryless textures for shader reads, killing the
host. That capability change was withdrawn; the candidate uses the original
host library while browser-scoped driver translation is investigated.
Both stock browsers now pass context creation, shaders, triangle/VBO, textures, FBO/readback, blend/depth, 180 frames per API, resource lifetime, loss/recovery and context stability. This does not establish 60 FPS or validate other browsers.

Completion delivery must run outside the legacy callback's driver locks.
The thread call must also use `THREAD_CALL_OPTIONS_ONCE`: otherwise two
drainers can deliver scheduled/completed notifications out of order.
CPU aliases of existing device-pager memory use `kIOMemoryHostOnly`, without
attempting to DMA-wire the same pages again (`prepare()` returned unsupported).

Remaining qualification includes mip/slice preservation and
resource cleanup under failure, WindowServer, guest restart and a matched
TestUFO performance comparison.

## Prepare the bridge on Windows

The bridge is experimental. Use a disposable VM with the matching host
runtime. The pinned build in the [Windows field report](../../docs/windows-tahoe-rtx5070.md)
failed in Weather and Maps; newer renderer fixes are being retested. The Windows
setup below is restricted to the tested Darwin 25.6.0 kernel. It does not
change the CPU allocation, machine identity, SIP or AMFI.

Build the bundle from a committed source snapshot as described above, then
copy the whole `ReimsMetalCompat.kext` back to Windows. Keep its embedded
`Contents/Resources/source-manifest.json`. With the VM stopped and Lilu
1.7.2 or newer installed in its OpenCore image, run:

```powershell
./scripts/windows/prepare-metal-compatibility.ps1 -KextDirectory C:/lab/ReimsMetalCompat.kext -Routing Global
```

The command checks that the source manifest matches a complete committed
snapshot in this checkout. This checks source provenance; it is not a
reproducible-build or code-signature verification of the executable.
It installs a versioned bundle after Lilu, enables `-reimsmetal` and rebuilds
OpenCore. Existing bundles remain in place. A failed rebuild restores the
previous configuration, image and preparation manifest. The backup folder
is printed on completion; retain it with the VM checkpoint.

Use `-Routing Diagnostic` to prepare the diagnostic connection without
routing ordinary apps through the bridge. Run `REIMSTOOLS/setup.command`
inside the guest to apply the existing Metal preference and guest power
settings, then restart macOS. Global routing remains an explicit choice;
it is not enabled for every new VM.

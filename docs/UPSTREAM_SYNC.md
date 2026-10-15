# Windows upstream synchronization, 2026-10-14

> Required scope: the existing Safari and Chrome must both use GPU WebGL.
> The separate WebKit result below is diagnostic evidence, not completion
> of that requirement. Driver compatibility work is still in progress.


## Preserved baseline and candidate

| Repository | Protected Windows baseline | Upstream integrated |
|---|---|---|
| Reims | `68d0a4d7` | `86d492197228df29514a87c3de62b2ce59f2188f` |
| QEMU | `c1dbb187` | dedicated `steelbrain/reims-vgpu`: `8d795445f266186203cda0ef63dc6de21675b958` |
| metal2vulkan | `8556e6e4` | compatibility checkpoint `ecb80e0`, then `61a2efc7d355ed2b0c80d108ee4a3d4f11ca405e` |

The candidate branch in each user fork is
`windows-native-tahoe-upstream-sync-20261014`. No reset/rebase, force push,
upstream PR or production deployment is part of this synchronization.
Reims pins the Windows runtime fork exactly at
`d41a1c2fb6aad7ef52bc9e0c8be8957b3e71c935`; its QEMU gitlink is
`7100064d13b42bb1be16671fcf18435822b13c3c`.
Runtime branch head `d566958` additionally corrects authored test fixtures;
it changes no production source or source fingerprint, so the validated
production dependency stays at `d41a1c2`.

## Reims changes reviewed

| Commit | Classification / integration |
|---|---|
| `871e0931` terminal view | Automatic; Unix CLI, retained Windows manager |
| `b3dc2d27` Metal error mode | Automatic docs; Windows already applies the setting |
| `88c9d3e7` dma-buf import | Manual shared API adaptation; actual dma-buf operations remain Unix-only |
| `36102ed7` merge ancestry | Integrated ancestry |
| `027df8ad` Recovery setup | Automatic Unix CLI; existing Windows helpers retained |
| `dd1934ca` Skylake default | Unix CLI only; verified Windows Haswell/WHPX flags retained |
| `24b37c9a` coherent host images | Integrated with the Windows runtime dependency |
| `8624a1fc` two-bit fixture | Automatic authored fixture |
| `2004a430` width warning | Automatic TUI; corresponding Windows manager warning added |
| `8b007820` wide-display failure | Automatic documentation; 4K choices retained |
| `39d6ca78` installer helper | Unix CLI; does not replace Windows ISO workflow |
| `86d49219` runtime pin | Manual exact user-fork pin, preserving Windows support |

## Runtime changes reviewed

| Commit | Classification / integration |
|---|---|
| `e51abf3` workflow docs | Integrated; requested Windows development branch used |
| `11715e2` HostMemory API | Manual; retain Windows VirtualQuery validation, page sizing and lifetimes; move platform-neutral helper out of Unix-only module |
| `9a82a7c` noncoherent host copy | Automatic common correctness fix |
| `edb5ffd` coherence lane | Integrated validation/comments; not itself proof of a guest fix |
| `86f0338` newer clippy | Integrated; native Rust 1.98.1 build checked separately |
| `dc7cd12` source hash | Integrated; fork hash repinned after Windows integration |
| `6cc8b08` compressed copies | Automatic; authored BC3 edge-copy GPU check added |
| `ecb80e0` heap alignment | Automatic 256 KiB reservation support; no claim that this GPU needs it |
| `61a2efc` driver/validation fixes | Automatic common fixes; Windows checks retained |

The fork's calibration enables no larger private array and applies only to
NVIDIA 616.56 on device 2783. LF checkout attributes keep source fingerprints
identical across hosts. The source-hash calculation reproduces upstream's
original pin before calculating the fork's pin. The artifact execution format
is unchanged; translation-profile values remain in the artifact digest.

## QEMU integration

The single upstream AppleSMC change overlaps a Windows implementation already
covering four-byte key enumeration, key metadata and count queries. Keep its
seven-key behavior and tests rather than regress to the narrower upstream
variant. Merge ancestry is recorded by `7100064d`.

WHPX topology, XSAVE/MTRR reset, retained power cycles, section sharing,
BAR/interrupt and JIS changes remain in the Windows tree. The QEMU checkout
inside the agent worktree is an independent clone, not an initialized shared
submodule repository.

## Safety and acceptance

Offline overlays, private firmware/NVRAM copies, baseline executable hashes
and Git refs preserve the working environment. Comparison VMs use sibling
writable overlays over the same frozen baseline checkpoint. Production
inputs are never opened writable by a candidate.

Compilation, native GPU checks, boot, browser pixel correctness and sustained
performance are distinct acceptance layers. [TEST_RESULTS.md](TEST_RESULTS.md)
and [WEBGL_DIAGNOSIS.md](WEBGL_DIAGNOSIS.md) record failures as failures.
Stock Safari 27 and Chrome 155 each pass all 22 authored GPU WebGL 1/2
stages through the experimental x86 kernel bridge. Browser executables and
the host capability answers remain unchanged. The separately adapted WebKit
is diagnostic history, not the delivery path. Verified committed source
snapshots and signed source manifests tie each build to its inputs.
Stable 60 FPS, 4K, extended stress and newer macOS versions remain unqualified.

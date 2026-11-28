# macOS 27 (Golden Gate) Route A 対応 — Phase 0 現状棚卸し・差分調査レポート

- 方針: **既存 `tqmane` ブランチ維持（`git rebase`・クリーンリベース禁止）＋ Route A（x86_64 Tahoe カーネル + WHPX + ARM64 ユーザーランド翻訳）**
- 対象リポジトリ:
  1. `tqmane/reims-vgpu` (`reims-vgpu`) vs `steelbrain/reims-vgpu` (`upstream/master`)
  2. `tqmane/qemu-reims-vgpu` (`vendor/qemu`) vs `steelbrain/qemu-reims-vgpu` (`upstream/steelbrain/reims-vgpu-rebase-2`)
  3. `tqmane/reims-manager` (`reims-manager`)（上流なし・独自フロントエンド）

---

## 1. 各リポジトリの現行ブランチと `git merge-base`

| リポジトリ | ローカル現在地 (`HEAD`) | リモート trunk | 対比上流 (`upstream`) | `git merge-base` | 乖離規模 |
|---|---|---|---|---|---|
| `tqmane/reims-vgpu` | `windows-native-tahoe` (`fa38578ab`) | `origin/windows-native-tahoe` (`fa38578ab`) | `upstream/master` = `upstream/development` (`3731afd9b`) | `86d492197228df29514a87c3de62b2ce59f2188f` | **自側**: 非マージ **69** 件＋マージ **14** 件<br>**上流側**: 非マージ **4,710** 件（計 4,712 件） |
| `tqmane/qemu-reims-vgpu` (`vendor/qemu`) | `windows-native-tahoe` (`b17ae0919e` pin) | `origin/windows-native-tahoe` (`2de1d37b90`) | `upstream/steelbrain/reims-vgpu-rebase-2` (`4571bc5fec`)<br>(`upstream/master` = `e9fa1f66f0`) | `9c88811b81a935a172c5c873b2ada2d9b38a5f74` | **自側**: 旧上流 `8d795445f2` をマージ後に独自非マージ **14** 件（`b17ae0919e` → `2de1d37b90` 間に PR #3/#4 あり）<br>**上流側**: QEMU 新ベースへリベース済み（`merge-base` 比で多数、独自差分大） |
| `tqmane/reims-manager` | `main` (`b826d58`) | `origin/main` (`b826d58`) | なし（オリジナル） | — | — |

### 1.1 サブモジュール・ピン不整合の即時検出

1. **`reims` `HEAD` (`fa38578ab`) と `qemu` `origin/windows-native-tahoe` (`2de1d37b90`) のズレ**:
   - `tqmane/reims-vgpu` の `windows-native-tahoe` (`fa38578ab`) および `reims-manager/backend/engine-lock.json` (`engine-28979c8c-fbba9f15-15`) が参照する `reims` (`28979c8cf`) の時点でも、superproject 側の `vendor/qemu` gitlink は依然として **`b17ae0919ec29f7989a568b27651c32400fb4504`** を指している。
   - 一方、`reims-manager/backend/engine-lock.json` の `qemu.commit` 自体は `origin/windows-native-tahoe` (`2de1d37b90`) 内の **`fbba9f15b61fdef776dc88bebe483312f869cf12`**（`93048dd3f6` `host-instance-id` および `fbba9f15b6` xHCI short packet tail IOC を含む）を直接固定している。
   - また、`fa38578ab` には `engine-lock.json` (`28979c8cf`) より新しい 2 コミット（`ef66fad38` 非 strict Refusal スキップ、`fa38578ab` `metal2vulkan@343dadf0`）が積まれている。
2. **`metal2vulkan` 依存フォーク**:
   - `Cargo.toml` の `metal2vulkan` / `metal2vulkan-runtime` は、`steelbrain/metal2vulkan`（`86d49219` 時点では `ecb80e019d`）ではなく **`https://github.com/tqmane/metal2vulkan`** (`343dadf01cddff14a42b6de4b7b10e9593b2ff7a`) をピン留めしている。

---

## 2. 上流 `steelbrain` の構造転換（Phase 2/3 判断の前提事実）

`86d49219` 以降、上流 `steelbrain/reims-vgpu` および `steelbrain/qemu-reims-vgpu` は、アーキテクチャ・ツールチェーン・ABI を全面刷新している。段階的マージ（Phase 3）や PoC（Phase 2）の前提として以下を把握する必要がある。

| 領域 | `tqmane/windows-native-tahoe` (現行) | `upstream/master` / `upstream/steelbrain/reims-vgpu-rebase-2` (最新上流) | 影響 |
|---|---|---|---|
| 仮想 GPU デバイス | `apple-gfx-pci` (`106b:eeee`, Apple ParavirtualizedGraphics ABI) | **`reims-graphics-pci` (`1234:5247`)**。`59df0ffbb` で導入し、`3db95eb18`（Reims）・`96e1b470ea`（QEMU）で `apple-gfx` と paravirt プロトコル（`c723ef179`）を削除済み | QEMU 引数・QOM `/machine/peripheral/gfx`・`ReimsVgpu.rom` の互換が断絶 |
| Option ROM | `ReimsVgpu.rom` (`firmware/`) | **`ReimsGraphics.rom`** (`firmware/prebuilt/ReimsGraphics.rom`) | OpenCore / QEMU ROM 設定の切替必須 |
| ゲスト側 Metal/Kext | `guest/metal-compat` (`ReimsMetalCompat.kext` + `Lilu.kext`, **Darwin 25 専用**) + ゲスト純正 Apple Paravirt ドライバ | 上流自前 kext 群: `guest/prebuilt/` (`x86_64`: `Graft.kext`, `ReimsGraphics.kext`, `RgfxGate.kext`) / `guest/prebuilt-arm64e/` (`arm64e`) + `/Library/GPUBundles/reims-graphics.bundle` | Tahoe 以降は上流が Apple ゲスト依存から脱却。Route A では x86_64 kext ↔ ARM64 翻訳プロセス間の境界検証が焦点 |
| Windows ビルド環境・成果物 | MSYS2 **`UCRT64`** (`1.98.1-x86_64-pc-windows-gnu`, `rust-toolchain.toml` は `1.98.1`), `reims_vgpu_c.dll` を生成し `build/qemu` に配置 | MSYS2 **`CLANG64`** (`x86_64-pc-windows-gnullvm`, `rust-toolchain.toml` は `nightly-2026-11-03`), `xtask/src/dist/windows.rs` (`4ba8eb34f`) が静的 `lib/libreims_vgpu.a`（`llvm-objcopy` で `.rvg` シンボルリネーム）＋共有 `lib/libreims_vgpu.dll`＋`bin/reims-vgpu-compile.exe` (`30e368e9a`) を生成し QEMU と静的リンク | `reims-manager/tools/build-engine.ps1` / `package_engine.py` / `scripts/windows/environment.ps1` の更新が必要 |
| Rust クレート / `metal2vulkan` | `reims-vgpu-core` + `reims-vgpu-vulkan` (`VulkanPlatform` + `metal2vulkan-runtime` クレート) | `1ef5403e5` で `VulkanPlatform` と `metal2vulkan-runtime` を削除し `metal2vulkan-rt` (`rt`) へ一本化。`fd591f503` で `reims-vgpu-core` を `reims-vgpu-vulkan` 内へ統合 | Phase 1 の直接 cherry-pick は `1ef5403e5` / `59df0ffbb` より前のコミット（`86d49219..0e88fde0d` 等）に限定するのが安全 |
| macOS 27 Metal 水準 | `macOS 26.7` 相当ベース | `fcd44ed9e` 等で **`MetalLevel::MacOS27_0_1`** 応答に対応（`metal2vulkan` 連携） | Golden Gate のユーザー空間 Metal 要求を満たす可能性が高い |
| ウィンドウ制御 / マウス / フォーカス | 独自 C API 拡張（`hide_host_cursor`, `relative_mouse` 関連）＋ SDL 子 HWND フォーカスルーティング | ゲーム時オートキャプチャ（`game.rs` 等）・独自タイトル通知。C API への `hide_host_cursor` プロパティ公開や子 HWND 埋込みフォーカス制御は **未実装** | `reims-manager` の `VmConsoleHost.cs`・`host-cursor.ps1` と衝突するため Phase 3 で独自拡張の移植・再配線が必須 |

---

## 3. `tqmane` 固有コミットの 3 分類一覧

分類基準:
- **(A) 上流に既に取り込み済みの WHPX / Windows 修正（または上流の公式実装で置換済み）**
- **(B) `reims-manager` 連携・CI パッケージング・ゲスト運用に必須の独自変更（Phase 3 でも温存／移植必須）**
- **(C) その他（`tqmane/metal2vulkan` ピン、診断ログ、未上流化バグ修正、旧 in-repo UI、ドキュメント等）**

### 3.1 `tqmane/reims-vgpu`（`86d49219..fa38578ab` の全 69 非マージコミット）

> 上記に加え、マージコミット 14 件（PR #4〜#12 統合および統合ブランチマージ、`229dd24c7`, `d57d57af2`, `5049e6e71` 等）が存在し、`windows-native-tahoe` の履歴を構成している。

| 分類 | コミット | 要旨・主な変更ファイル | 上流比較・Phase 1/3 での扱い |
|---|---|---|---|
| **(A)** | `b4be21aff` | Windows core テスト（debug-event death 監視）と C API の Windows セクションベース・ゲストメモリ対応（`reims_vgpu.h`, `crates/reims-vgpu-c/src/device/memory/windows.rs`） | 上流が Windows セクションハンドルと death テストを統合済み（構造体レイアウト差に注意し上流側採用＋必要なら互換維持） |
| **(A)** | `c0096da43` | 初期 Windows Vulkan プラットフォーム (`os/windows/*`), `xtask/src/dist/windows.rs`, `scripts/windows/*` | `os/windows/*` と `dist/windows.rs` は上流の公式 Windows/CLANG64 実装で置換済み。ただし `scripts/windows/environment.ps1`, `build.ps1` と UCRT64 出力契約は **(B)**（`reims-manager` CI が直接実行） |
| **(A)** | `a0eeadca9` | Windows Vulkan section import・clock 修正と Tahoe VM 準備 | `os/windows/*` 本体は上流 `os/windows/*` に吸収済み、`scripts/windows/*` は **(B)** |
| **(A)** | `90fa9c0b9` | Windows セクション向け C ハーネス対応 (`crates/reims-vgpu-c/tests/c/harness.c`) | 上流 `master` で Windows 向け C ハーネス整備済み（上流採用可） |
| **(A)/(C)** | `c678499d8` | Windows Vulkan (`encode_render.rs`, `standin.rs`), `guest_ram/windows.rs` の検証修正 + `tqmane/metal2vulkan` ピン + `tests/webgl/*` + `benchmark.ps1` + ドキュメント | `encode_render`/`standin`/`guest_ram` 部は上流実装で置換。`tests/webgl/*` と `benchmark.ps1` は **(C)**、ピン部分は上流 `metal2vulkan` と競合整理要 |
| **(B)** | `cad1d7ec3` | `scripts/windows/environment.ps1` でネイティブ Git を優先設定 | `reims-manager/tools/build-engine.ps1` が直接ドットソースするため温存 |
| **(B)** | `d61daa51a`, `cbea489fa`, `474822fa5`, `c34483209`, `217e8e17a`, `bce6c05b8`, `1bbf1e055`, `63c7129ab`, `d8b9c2afb`, `b5f4d6dd0`, `76db12e16`, `ec9746a70`, `0fe4585b4` | `scripts/windows/*.ps1` 群（OpenCore 準備・`ScanPolicy=0x301`・FAT16 維持・`x86legacy`・25.4.0 sysctl パッチ・TSC 時刻対策・QEMU `-L` pc-bios・QMP パワーサイクル保持・SSH パス）と対応 `vendor/qemu` サブモジュールピン (`ec9746a70`, `0fe4585b4`) | `scripts/windows/environment.ps1` / `build.ps1` / `ucrt64.cmake` は `reims-manager` CI が直接依存。他スクリプトの知見は `reims-manager/backend` へ移行済みだが、検証用/互換用に履歴維持 |
| **(B)** | `7f3f5d127` | `vendor/qemu` ピン（SMC 4-byte、USB マイク、UsbDk 非キャプチャ列挙） | `reims-manager` の USB/Audio 契約に必要な QEMU ピン |
| **(B)** | `2d690d741`, `303f23566`, `ac60717cc`, `700379ce3` | **`guest/metal-compat/*` (`ReimsMetalCompat.kext`)** とソーススナップショット検証 (`source_snapshot.py`) | **Tahoe (Darwin 25) WebGL/Metal に必須**。`reims-manager/backend/guest-bridge-lock.json` と `.github/workflows/guest-bridge.yml` が `700379ce` をピン参照。Phase 3 でもディレクトリごと温存必須（Golden Gate には適用しない） |
| **(B)** | `dbac2faca`, `20aae681c` | **`hide_host_cursor` スコープドカーソル制御 C API**・Windows 解放ショートカット (`Ctrl+Alt+Home`) (`reims_vgpu.h`, `host_cursor_policy.rs`) | 上流に存在しない。`reims-manager` の QEMU `hide-host-cursor` QOM プロパティが依存しているため Phase 3 で維持または新 C API へ移植要 |
| **(B)** | `cc703b8c3`, `ba1f83074`, `18f8c6156` | **捕捉相対マウス入力 C API** と QEMU 連携 (`reims_vgpu.h`, `reims-vgpu-window`, `vendor/qemu`) | 上流は C API なしのゲーム自動捕捉 (`game.rs`) 方式で設計が異なる。`reims-manager` / 既存利用との互換方針を Phase 3 で要判断 |
| **(B)** | `f5c30e700`, `0425ed833`, `00df0adc2`, `271583d79`, `e080610bb`, `082498822`, `bf36211a5` | **埋め込み SDL 子ウィンドウのネイティブフォーカス所有権・キーボードルーティング** (`crates/reims-vgpu-window/src/native/linux/windows.rs`, `sdl/shell.rs`, `features.md`) | 上流に存在しない。`reims-manager` の `VmConsoleHost.cs` (SDL HWND 埋め込み) でキー入力を機能させるために **極めて重要**。上流が `native/sdl_thread`・`native/windows.rs` へ再構成したため、Phase 3 で構造移植必須 |
| **(C)** | `fa38578ab`, `28979c8cf`, `15e71da07`, `074602126`, `72da73db1`, `996f8e9ff`, `e26a85b4d`, `eb4d6cc0e`, `3bdf74749`, `b97cd2390`, `ef2fe2504`, `b8b4486c4` | `tqmane/metal2vulkan` フォーク側修正への Cargo ピン更新＋関連テスト/ドキュメント調整（AIR テクスチャ即時値・メモリ解放アンロック・大型 GPU 割当再利用・moved-writer・結合 submit・タイルカバレッジ・MS position・ordered framebuffer blending） | `upstream` が `metal2vulkan-rt` へ全面移行したため Phase 3 で `steelbrain/metal2vulkan` 最新へ収束可能か要検証（未反映修正がないか差分確認要） |
| **(C)** | `ef66fad38` | 非 strict モードで Vulkan パイプライン/Raw フォーマット拒否・ライブラリ読込失敗時に QEMU 即死せずログ+エラー返却 (`REIMS_VGPU_STRICT=1` で従来通り abort) | **上流未採用（Golden Gate PoC で非常に有用）**。Phase 2 PoC で未対応シェーダ/フォーマット踏破時の切り分けに効く。Phase 3 でも残置検討価値大 |
| **(C)** | `e2e30dc47`, `06eb5b49a`, `3ccfa3e79`, `ec5b039a1`, `f3ff6a0e0` | デフォルト無効の FIFO / Vulkan リソース生成・破棄スローオペレーション計測・サマリカウンタ (`timing.rs`, `commands.rs`) | 独自診断用（上流には別の `REIMS_VGPU_TRACE` がある） |
| **(C)** | `4d727a4fe` | Windows 実機での **Apple JIS 英数/かな IME メッセージの USB LANG2/LANG1 転送** (`reims-vgpu-window`, `vendor/qemu`) | 上流 `keymap.rs` には `0x66->0x91`, `0x68->0x90` 変換表のみあり、Windows IME 側の補正は未収録。JIS 実機維持のため Phase 3 で `native/sdl_thread` へ移植推奨 |
| **(C)** | `4560ce68b`, `c1feca039`, `68d0a4d77` | 旧 `apps/windows-manager/*` | 独立リポジトリ `tqmane/reims-manager` により役割終了（Phase 3 マージ時は凍結または削除候補） |
| **(C)** | `dd083497c`, `8890b9b72`, `64801a920`, `775c9caf9`, `d10043ad5`, `aec23a61a`, `b2e38c709`, `a9c6e9364`, `c6806c5ff`, `ae4066d02`, `b828dc4d0` | `tests/webgl/*` プローブとドキュメント類 (`docs/*.md`) | ドキュメント・再現ハーネス（独立、安全に維持可能） |

### 3.2 `tqmane/qemu-reims-vgpu`（`8d795445f2..2de1d37b90` の全 14 非マージコミット）

> 補足: `2de1d37b90` には上流 `8d795445f2`（`applesmc: answer GET_KEY_BY_INDEX`）以前が統合されており、上流 `rebase-2` 側でも `83de566436` として同内容を含む。

| 分類 | コミット | 要旨・変更対象ファイル | 上流 `rebase-2` 対応状況・扱い |
|---|---|---|---|
| **(A)** | `22229b3d96` (主体部) | Windows guest RAM のページファイルセクション共有 (`physmem.c`, `ramblock.h`, `oslib-win32.c`), `apple-gfx` Windows pkg-config ビルド (`meson.build`), WHPX 起動・CET・保守的 dirty 同期 (`whpx-all.c`), `symlink-install-tree.py` コピーフォールバック | RAM セクション共有・Meson/Windows 対応・CET は上流 `rebase-2` が別実装 (`reims-graphics-pci` + `CLANG64`) で対応済み。ただし **`scripts/symlink-install-tree.py` の権限なし環境コピーフォールバック** は上流未実装のため Windows 非管理者環境向けに維持推奨 |
| **(A)** | `fbba9f15b6` | `hw/usb/hcd-xhci.c`: short packet (ISP) 後の末尾 TRB の IOC を適切に報告 (+ `test-xhci-xfer-report.py`) | 上流 `rebase-2` の `ae97f3f533`（`post the event a TD's last TRB owes after a short packet`）が同等バグを修正済み。Phase 3 では上流版と動作整合確認し片側に統一 |
| **(A)? / 要検証** | `6fa3af097d` | `target/i386/whpx/whpx-all.c`: ウォームリセット時 XSAVE バッファ長ネゴ・OSXSAVE オフ時も拡張状態/保留イベントをリセット (+ `test-whpx-reset.c`) | 上流 `rebase-2` は新しい upstream QEMU ベースを取り込み `whpx_set_registers(WHPX_LEVEL_RESET_STATE)` で **`WHvResetPartition`** ＋ TSC deadline / Wait-for-SIPI を実行。OSXSAVE 未有効時や拡張状態の扱いが完全に賄われるかは Phase 3 実機テスト要 |
| **(B)** | `39d1431824`, `bac1ba2693` | `hw/display/apple-gfx.{c,h}` にライブ書換可能な QOM `hide-host-cursor` プロパティを追加・エラー文言改善 | 上流は `apple-gfx` ごと削除。`reims-manager` (`NativeGuestAndLaunchService.cs`, `NativeVmControlService.cs`, `host-cursor.ps1`) が `/machine/peripheral/gfx` で直接参照するため、`reims-graphics-pci` 移行時は C API 側ともども方針調整が必須 |
| **(B)** | `b17ae0919e` | `hw/display/apple-gfx.{c,h}`, `ui/input.c`: 相対マウスパケットの QOM ポリシーと USB HID/Tablet 動的ルーティング | Phase 1/2 の `apple-gfx-pci` ではそのまま維持。Phase 3 で `reims-graphics-pci` へ移行する場合は上流のゲーム自動捕捉との棲み分け確認要 |
| **(B)** | `c1dbb18729` | `hw/usb/host-libusb.c`: UsbDk バックエンドでインベントリ参照時にデバイスを open しない（誤リダイレクト防止） | **上流未実装** (`usb-host` 既存動作のまま)。`reims-manager` の USB デバイス管理・ホストストレージ誤奪取防止に必須のため Phase 3 でも必ず維持・移植 |
| **(B)** | `93048dd3f6` | `hw/usb/host-instance-id.h`, `host-libusb.c`: `usb-host` へ `host-instance-id` 完全一致フィルタ追加（同型 USB 複数接続や iPhone 再接続識別に利用） | **上流未実装**。`reims-manager` (`usb.ps1` / C# USB 転送制御) との連携仕様として Phase 3 でも温存・移植必須 |
| **(B)** | `934572a4a1`, `d72cf0427c` | `whpx-all.c`: `MSR_CORE_THREAD_COUNT` を構成済みコア/スレッド数で応答（なければ macOS が 8 vCPU を拒否する場合あり） | **上流 `rebase-2` 未収録**。Tahoe / Golden Gate のマルチコア起動維持のため Phase 3 でも必ず持ち越す |
| **(C)** | `45807f2484` | `whpx-all.c`: CPU リセット時に固定/可変 MTRR を明示ゼロクリア（OVMF `MemDetect.c` アサート回避） | **上流 `rebase-2` 未収録**（`WHvResetPartition` 頼み）。実機リスタート回帰防止のため Phase 3 で必要性を再確認し、必要なら移植 |
| **(C)** | `33fcf0d395` | `hw/input/hid.c` (`0x90/0x91` JIS Kana/Eisu マッピング) + `hw/usb/dev-hid.c` (Logical Maximum 符号なし 255 変更) | **上流 `rebase-2` 未収録**。日本語キーボードの英数/かな入力維持に必須のため Phase 3 でも移植 |
| **(C)** | `5891dd29e2` | `hw/misc/applesmc.c`: index / key-info コマンドと4バイト引数・実キー件数/メタデータ応答により `PerfPowerServices` ポーリングを停止 | **上流 `rebase-2` 未収録**（上流は `GET_KEY_BY_INDEX` のみ）。省電力・ログ静粛性のため Phase 3 でも移植推奨 |
| **(C)** | `3c4e29b272` | `audio/dsound*`, `hw/usb/dev-audio.c`: Windows DirectSound キャプチャ + USB マイクデバイス公開（録音デバイス無しホスト耐性も含む） | 一部（録音デバイス無しで再生を止めない点）は上流 `e435673c53` で対応済みだが、**USB マイク入力 (`dev-audio.c` capture) は上流未実装**。`reims-manager` のマイク設定機能のため Phase 3 でも維持要 |

> ※ 参考: 上流 `steelbrain/reims-vgpu-rebase-2` のうち、**`apple-gfx-pci` に依存しない独立 WHPX 改善**（`4025704118` AMD ホスト上の Intel MSR 回避、`b63d6420ad` `CPUID[0xD]` sub-leaf 0/1 EBX 修正、`0fdbf3810d` BQL 非保持での vCPU 実行）は未取り込みであり Phase 1 のバックポート候補となる。一方、`cc128f0e14`（CPUID leaf `0x4`）は `origin/windows-native-tahoe` の `cpuidExitList` に既に含まれており、`c3704402af`（`tsc_deadline_refused`）についても現行 `origin/windows-native-tahoe` が `WHvX64RegisterTscDeadline` を設定せず `whpx_vcpu_run` 側で `CPUID_EXT_TSC_DEADLINE_TIMER` を無効化しているため対象外とする。

---

## 4. `tqmane/reims-manager` のエンジン依存箇所監査

`tqmane/reims-manager` (`b826d58`) には、`reims-vgpu-pci` (`apple-gfx-pci`)、MSYS2 UCRT64 出力ファイル名、および **Darwin 25 (macOS 26 Tahoe)** 前提の OpenCore/Kext 制約が複数箇所に強く結合している。

| コンポーネント / ファイル | 主な確認行範囲 | 強い依存内容・ハードコード仕様 | Phase 2 (PoC) / Phase 4 (両対応化) での影響 |
|---|---|---|---|
| `backend/engine-lock.json` | L1–33 | `id`: `engine-28979c8c-fbba9f15-15`、`sources`: `reims` (`28979c8cf`), `qemu` (`fbba9f15b6`), `metal2vulkan` (`7f09d6be`), `libusb` (`7da948a0`)、OpenCore `1.0.8`、QEMU edk2 firmware、7z `26.04` | エンジン切替・新エンジン世代メタデータ管理の起点 |
| `backend/guest-bridge-lock.json` | L1–34 | `required_engine_id` および `compatible_engine_ids` でエンジン ID 列挙、`abi_family`: `"Darwin25"`, `min_kernel`: `"25.0.0"`, `max_kernel`: `"25.99.99"`, `Lilu 1.7.2` + `ReimsMetalCompat.kext` (`700379ce`) | Golden Gate (Darwin 26 相当想定) では絶対に誤適用させないガードが必須 |
| `tools/build-engine.ps1`<br>`tools/package_engine.py`<br>`.github/workflows/engine.yml` | L1–98<br>L1–223<br>L1–63 | `scripts/windows/environment.ps1` + `build.ps1` を MSYS2 `UCRT64` で呼び出し、`$source/build/qemu/reims_vgpu_c.dll`・`ReimsVgpu.rom` の存在と Git HEAD 整合を検証し `reims-engine.json` (`schema: 1`) を生成 | Phase 3 以降で上流 `CLANG64` (`libreims_vgpu.dll` / `libreims_vgpu.a` / `ReimsGraphics.rom` / `reims-vgpu-compile.exe`) を扱う場合、世代別パッケージング・環境切替が必要 |
| `src/ReimsManager/Services/NativeStatusService.cs` | L1–571 | `VerifyEnginePackage` が `reims-engine.json` と `engine-lock.json` の `id` / `sources` / SHA-256 を検証（`reims_vgpu_c.dll` 等の実体ハッシュチェック） | 世代フィールド導入後も、既存の `engine-28979c8c-fbba9f15-15` ロック検証を壊さない前方互換設計が必要 |
| `src/ReimsManager/Services/NativeEngineService.cs` | L1–228 | `TestNativeEngineAsync` が `qemu -device help` に `apple-gfx-pci` が含まれるかでネイティブエンジン有効性を判定 | 新エンジン（`reims-graphics-pci`）にも対応するには世代に応じたデバイスプローブ切替が必須 |
| `src/ReimsManager/Services/NativeGuestAndLaunchService.cs`<br>`backend/start.ps1` | L1–641<br>L1–472 | QEMU 起動引数で `-device '{"driver":"apple-gfx-pci","id":"gfx",...,"resolution":[...],"hide-host-cursor":...}'` を固定生成、`apple-gfx-pci,help` に `resolution` があるか検証、`REIMS_VGPU_LOG` / `REIMS_VGPU_STATS` / `REIMS_VGPU_WINDOW` を制御、保持済み VM の起動引数比較・サニタイズ | Phase 4 でエンジン世代別にデバイスドライバ名・プロパティ・`hide-host-cursor` 対応有無・OpenCore ROM/Kext を切り替える核心箇所 |
| `src/ReimsManager/Services/NativeVmControlService.cs`<br>`backend/host-cursor.ps1` | L1–495<br>L1–118 | QMP `qom-get` / `qom-set` の `/machine/peripheral/gfx` に対して `hide-host-cursor` を読み書きし、デバイスドライバが `apple-gfx-pci` であることも確認 | `reims-graphics-pci` 移行時に同等 QOM が未移植だと QMP カーソル制御が失敗するため、未対応時スキップまたは QEMU 側移植が必要 |
| `src/ReimsManager/VmConsoleHost.cs` | L1–310 | SDL トップレベル HWND を `EnumWindows` で捕捉し、`WS_POPUP` / `OwnerIndex` / `SetWindowRgn` で WPF `HwndHost` に埋め込み、フォーカス転送を実施（`REIMS_VGPU_WINDOW` を消去/無効化しない前提で外部 SDL ウィンドウを期待） | `reims` 側の埋め込み子ウィンドウ・キーボードフォーカス入力修正と表裏一体。エンジン変更時にウィンドウクラス・タイトル・フォーカス挙動を壊さない検証が必須 |
| `src/ReimsManager/Services/NativeOpenCoreService.cs`<br>`backend/opencore/base.py`<br>`backend/opencore/boot-policy.py`<br>`backend/opencore/apple-account.py`<br>`backend/opencore/guest-bridge.py`<br>`backend/prepare-opencore.ps1` | L1–991<br>L1–32<br>L1–9<br>L1–17<br>L1–161<br>L1–198 | <ul><li>`base.py`: `ScanPolicy=0x301`, `SecureBootModel='x86legacy'`, `DmgLoading='Signed'`, `Vault='Optional'`, `csr-active-config=00000000`, `-v keepsyms=1 serial=3 debug=0x108` に固定</li><li>`guest-bridge.py`: `abi_family == 'Darwin25'` かつ `min/max_kernel` 先頭 `25` を要求し `Lilu` + `ReimsMetalCompat.kext` + `-reimsmetal` を注入</li><li>`SetupVmCore`: Guest Bridge を強制有効化＋エンジン互換 ID チェックあり</li><li>**`ImportDiskCore`**: qcow2/vmdk/raw 取込時は Guest Bridge を自動有効化しない</li><li>`PrepareUpdatesAsync`: `RestrictEvents` を `25.0.0..25.99.99` で投入</li><li>`apple-account.py`: `25.4.0` 限定カーネルパッチ</li></ul> | <ul><li>**Phase 2 に好都合な点**: `ImportDiskAsync` 経由なら `ReimsMetalCompat.kext` 非注入でディスクを取り込めるが、現状 `base.py` の `SecureBootModel='x86legacy'`・`DmgLoading='Signed'`・`csr-active-config=00000000` を Golden Gate 向けに一時緩和するプロファイル/フラグが必要</li><li>**Phase 4**: OpenCore 設定ページ (`MainWindow.xaml` / `MainWindow.Settings.cs` / `MainWindow.Control.cs`) に `Golden Gate (Experimental)` プリセットを追加し、Darwin 25 専用ゲストアドオン適用をブロックするガードを導入</li></ul> |

---

## 5. フェーズ別リスク評価・次フェーズへの推奨アクション

### 5.1 Phase 1（現行ブランチ維持での最小バックポート）の選定方針

`apple-gfx-pci` と `reims-manager` 連携を壊さないため、`59df0ffbb`（`reims-graphics` 導入）および `1ef5403e5`（`metal2vulkan-runtime` 削除）**より前**の独立修正のみに絞るのが安全である。

1. **ピン整合確認**:
   - `tqmane/reims-vgpu` (`fa38578ab`) が `ef66fad38`（非 strict Vulkan Refusal 耐性）と `fa38578ab` (`metal2vulkan@343dadf0`) を含み、`vendor/qemu` `origin/windows-native-tahoe` (`2de1d37b90`) に既に `fbba9f15b6` / `93048dd3f6` が入っている状態を整理する。
2. **`tqmane/qemu-reims-vgpu` バックポート候補（`upstream/steelbrain/reims-vgpu-rebase-2` から選択適用・`apple-gfx` 非接触）**:
   - `4025704118` (`whpx: keep an Intel guest's MSRs that Hyper-V refuses on an AMD host`)
   - `b63d6420ad` (`whpx: i386: forward the dynamic EBX of CPUID[0xD] sub-leaves 0 and 1, not 1 and 2`)
   - `0fdbf3810d` (`target/i386/whpx: let a vCPU run on without the BQL when nothing is pending`)
3. **`tqmane/reims-vgpu` バックポート候補（`86d49219..0e88fde0d` から選択適用）**:
   - `7314c4559` ＋ `8a0f04d9e`: SkyLight 2048px 超タイルや mip 重複時のバッファ長計算修正（`crates/reims-vgpu-core/src/resources/texture/dimension.rs` のみの独立修正）
   - `4097f5ee5`: `reims-vgpu-core` の 2^64 境界丸め、競合 transfer buffer 上書き、IOSurface retain バグ修正
   - `f3a2f7176`: Vulkan ドライバ・バリデーションレイヤーのエラーを stderr へ中継（`crates/reims-vgpu-vulkan/src/answer.rs` のみの独立修正）
   - `0d44fea30`: ゲストプロセス強制終了時にキュー済み Metal 処理が QEMU を巻き込んで abort する不具合の修正（※ `crates/reims-ctl` と `tests/suite` も変更しているため、Phase 1 では `core`/`vulkan` 修正部のみの最小バックポートとして検証）
   - ※ **`metal2vulkan` 依存を動かす上流コミット（例: `fcd44ed9e` `MacOS27_0_1` レベル等）は、`tqmane/metal2vulkan@343dadf0` とのコード差分・クレート API 互換を確認するまで Phase 1 へ直接混ぜない**（フェーズ分離原則）。

### 5.2 Phase 2（Route A PoC 起動）の事前確認ポイント

- `reims-manager` 側で `ImportDisk` を使うことで `ReimsMetalCompat.kext`（Darwin 25 専用）を自動注入せずにハイブリッドディスクを取り込める。
- ただし現行 `base.py` が `SecureBootModel='x86legacy'`、`DmgLoading='Signed'`、`csr-active-config=00000000` を強制生成するため、Phase 2 検証時に OpenCore 緩和値（例: `SecureBootModel='Disabled'`, `DmgLoading='Any'`, `csr-active-config`, AMFI/SSV 無効化 boot-args）を実験用プロファイルまたは一時パッチとして注入し、シリアル・`REIMS_VGPU_LOG`・WindowServer クラッシュログを採取する。

### 5.3 Phase 3（必要な場合のみ）の段階的マージ節目と Route A 特有の境界

Phase 2 で Apple Paravirt 路線が限界に突き当たった場合の `git merge` マイルストーン候補（rebase 禁止・履歴保持）:

1. **Merge 1 (`0e88fde0d` 付近、136 コミット地点)**: `reims-graphics` 本格導入・大規模クレート削除前の最終地点まで統合し、`tqmane` 側 Windows / `reims-vgpu-window` / `guest/metal-compat` と衝突を最小限で解決。
2. **Merge 2 (`59df0ffbb` → `3db95eb18` → `fd591f503` → `4ba8eb34f` 付近、2,867 コミット地点)**: `reims-graphics-pci` (`1234:5247`)、`ReimsGraphics.rom`、`apple-gfx` 削除、`reims-vgpu-core` の `vulkan` 統合、`xtask/src/dist/windows.rs` (CLANG64) 導入までを取り込み、`reims-manager` 向け独自修正（埋め込みフォーカス・カーソル制御・USB・JIS・WHPX `MSR_CORE_THREAD_COUNT` 等）を新設計へ移植。
3. **Merge 3 (`upstream/master` = `3731afd9b` / `rebase-2` 最新 `4571bc5fec`)**: `metal2vulkan-rt` (`1ef5403e5`, 3,889 コミット地点)、`guest/prebuilt` 最新を取り込み（この構成では `metal2vulkan-rt` のデフォルトが macOS 27 水準に応答）。

**Route A 特有の技術リスク（`guest/` 境界）**:
- 上流 `guest/prebuilt/` (`x86_64`) の `RgfxGate.kext` は x86_64 版 `reims-graphics.bundle` を、一方 `guest/prebuilt-arm64e/` は `arm64e` 版 kext＋バンドルを供給する想定で作られている（`guest/prebuilt/set.toml` vs `guest/prebuilt-arm64e/set.toml` 確認済み）。
- **Route A（x86_64 カーネル ＋ ARM64 ユーザーランド翻訳）** では、「**x86_64 `ReimsGraphics.kext` ＋ x86_64 `RgfxGate.kext` の下で、ARM64 翻訳実行プロセスが `/Library/GPUBundles/reims-graphics.bundle` をロードして IOKit UserClient 通信する**」という非標準の交差アーキテクチャ構成になる。そのため Phase 3 へ進んだ場合、`reims-graphics.bundle` の `arm64`/`universal` スライス包含、`RgfxGate` の AMFI/Sandbox バイナリ承認条件、および `RgfxUserClientABI.hpp` 構造体レイアウトの検証・修正が主要作業となる見込み。

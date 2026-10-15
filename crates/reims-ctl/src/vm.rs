//! A VM: its directory, and the settings in its `vm.toml`.
//!
//! A VM's directory holds its `vm.toml`, its disks (unless a disk was used in place), its
//! firmware copies, `qemu.pid` while QEMU runs, the QMP socket where the path is short enough,
//! and `logs/`. The registry (`vms.toml` in the config directory) lists each VM by its name and
//! directory, so a VM may live anywhere.

use crate::host::Profile;
use crate::paths;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const CONFIG_FILE: &str = "vm.toml";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmConfig {
    pub name: String,
    pub profile: Profile,
    pub cpus: u32,
    pub memory_mib: u64,
    /// The system disk, relative to the VM's directory unless absolute.
    pub disk: PathBuf,
    #[serde(default)]
    pub disk_format: ImageFormat,
    /// How long `stop` waits for the guest to shut down before it powers the VM off.
    #[serde(default = "default_stop_timeout")]
    pub stop_timeout_secs: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub display: Display,
    #[serde(default)]
    pub gpu: Gpu,
    #[serde(default)]
    pub audio: Audio,
    #[serde(default)]
    pub network: Network,
    #[serde(default)]
    pub logs: Logs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q35: Option<Q35>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vmapple: Option<Vmapple>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_env: BTreeMap<String, String>,
}

fn default_stop_timeout() -> u64 {
    90
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    #[default]
    Qcow2,
    Raw,
}

impl ImageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormat::Qcow2 => "qcow2",
            ImageFormat::Raw => "raw",
        }
    }

    /// The format of the image at `path`, from its first bytes: qcow2's magic, else raw.
    pub fn detect(path: &Path) -> Result<ImageFormat> {
        use std::io::Read;
        let mut magic = [0u8; 4];
        let mut f =
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let n = f.read(&mut magic)?;
        Ok(if n == 4 && magic == *b"QFI\xfb" {
            ImageFormat::Qcow2
        } else {
            ImageFormat::Raw
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Display {
    /// The guest's display ports, 1 to 8, each with a window of its own.
    pub count: u32,
    /// The mode the guest runs first, `<width>x<height>`, or `host` for the host screen's own
    /// modes at its density.
    pub resolution: String,
    /// Whether the guest is told a density at which it runs a HiDPI desktop at the mode's pixels
    /// (3840x2160 then looks like 1920x1080).
    pub hidpi: bool,
    /// Borderless windows that fill their screens.
    pub borderless: bool,
    /// The refresh rate of the modes offered to the guest; the host screen's by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh: Option<u32>,
}

impl Default for Display {
    fn default() -> Display {
        Display {
            count: 1,
            resolution: "1920x1080".into(),
            hidpi: false,
            borderless: false,
            refresh: None,
        }
    }
}

impl Display {
    /// The `resolution` property apple-gfx takes, checked as QEMU checks it.
    pub fn qemu_resolution(&self) -> Result<String> {
        let r = self.resolution.trim();
        if r == "host" {
            if self.hidpi {
                bail!("resolution host takes the host screen's own density: turn HiDPI off");
            }
            return Ok("host".into());
        }
        let (w, h) = parse_resolution(r)?;
        if !self.hidpi {
            return Ok(format!("{w}x{h}"));
        }
        match hidpi_ppi(w) {
            Some(ppi) => Ok(format!("{w}x{h}@{ppi}")),
            None => bail!(
                "a HiDPI desktop needs a mode at least 1920 pixels wide (it looks half that size), \
                 not {w}x{h}"
            ),
        }
    }

    /// The size a HiDPI desktop looks like, half the mode's pixels, where it runs one.
    pub fn looks_like(&self) -> Option<(u32, u32)> {
        if !self.hidpi {
            return None;
        }
        let (w, h) = parse_resolution(self.resolution.trim()).ok()?;
        hidpi_ppi(w).map(|_| (w / 2, h / 2))
    }

    pub fn check(&self) -> Result<()> {
        if !(1..=8).contains(&self.count) {
            bail!("displays must be 1 to 8, not {}", self.count);
        }
        if let Some(r) = self.refresh
            && !(1..=1000).contains(&r)
        {
            bail!("refresh must be 1 to 1000 Hz, not {r}");
        }
        self.qemu_resolution().map(|_| ())
    }
}

pub fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .with_context(|| format!("resolution must be <width>x<height> or host, not {s:?}"))?;
    let w: u32 = w
        .trim()
        .parse()
        .with_context(|| format!("bad width in {s:?}"))?;
    let h: u32 = h
        .trim()
        .parse()
        .with_context(|| format!("bad height in {s:?}"))?;
    if !(1..=65535).contains(&w) || !(1..=65535).contains(&h) {
        bail!("each side of a resolution is 1 to 65535 pixels, not {s:?}");
    }
    Ok((w, h))
}

/// The millimetres a guest is told for `width` pixels at `ppi`, as apple-gfx computes them
/// (`apple_gfx_guest_millimeters`): in single precision, rounded.
fn guest_millimeters(width: u32, ppi: u32) -> u64 {
    let mm = width as f32 / (ppi as f64 / 25.46) as f32;
    (mm as f64 + 0.5) as u64
}

/// The whole centimetres the guest reads them as, rounded up.
fn guest_centimeters(width: u32, ppi: u32) -> u64 {
    guest_millimeters(width, ppi).div_ceil(10)
}

/// apple-gfx's `apple_gfx_guest_scale_unknown`: whether the scale a macOS guest runs a mode at
/// is not known from the density it is told, which QEMU refuses.
fn guest_scale_unknown(width: u32, ppi: u32) -> bool {
    let cm = guest_centimeters(width, ppi);
    let w = width as u64;
    if cm == 0 || guest_millimeters(width, ppi) > u16::MAX as u64 {
        return true;
    }
    if w * 35 >= 2021 * cm {
        return width < 1920 || cm < 29;
    }
    width > 3811 || w * 66 > 3811 * cm
}

/// A density at which the guest runs `width` HiDPI: about 29.5 cm across, so it reads 30 cm and
/// at least 146.667 ppi, where apple-gfx knows the scale (widths of 1920 and more).
pub fn hidpi_ppi(width: u32) -> Option<u32> {
    if width < 1920 {
        return None;
    }
    let ppi = (width as f64 * 25.46 / 295.0).round() as u32;
    let cm = guest_centimeters(width, ppi);
    let hidpi = width as u64 * 35 >= 2021 * cm;
    (hidpi && !guest_scale_unknown(width, ppi)).then_some(ppi)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Gpu {
    /// The Vulkan driver's manifest (ICD JSON) the loader is limited to: `VK_DRIVER_FILES`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_file: Option<PathBuf>,
    /// The PCI device ID the loader keeps, where the driver drives several GPUs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u32>,
    /// The GPU's name, for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Audio {
    pub enabled: bool,
    /// QEMU's audio backend (`coreaudio`, `pipewire`, `pa`, `alsa`, `none`); the host's best
    /// by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

impl Default for Audio {
    fn default() -> Audio {
        Audio {
            enabled: true,
            backend: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Network {
    pub enabled: bool,
    pub mac: String,
    /// A host port on 127.0.0.1 forwarded to the guest's SSH port.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_port: Option<u16>,
}

impl Default for Network {
    fn default() -> Network {
        Network {
            enabled: true,
            mac: random_mac(),
            ssh_port: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Logs {
    /// The library's log, the guest's serial console and QEMU's timestamps, into `logs/`.
    /// QEMU's own errors are kept either way.
    pub enabled: bool,
    /// `REIMS_VGPU_LOG`'s filter: a level, and `target=level` pairs.
    pub level: String,
}

impl Default for Logs {
    fn default() -> Logs {
        Logs {
            enabled: false,
            level: "info".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Q35 {
    /// The OpenCore image, in the VM's directory: made for it, or a copy of one given.
    pub opencore: PathBuf,
    #[serde(default)]
    pub opencore_format: ImageFormat,
    /// OVMF's code and variables (copies in the VM's directory).
    pub firmware_code: PathBuf,
    pub firmware_vars: PathBuf,
    /// The installer, a raw image (reims-util's offline installer, or a recovery image's raw
    /// copy) attached as a disk, whose writes go to a temporary overlay: used in place where
    /// absolute; relative, the VM's raw copy of a DMG (`create::installer_media`), which a change
    /// of installer deletes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installer: Option<PathBuf>,
    #[serde(default = "default_cpu_model")]
    pub cpu_model: String,
    #[serde(default = "default_cpu_flags")]
    pub cpu_flags: String,
    /// The AHCI port of the system disk: 3, after OpenCore's 2, for every VM reims-ctl makes.
    /// OpenCore's saved boot entry names the port.
    #[serde(default = "default_disk_port")]
    pub disk_port: u32,
}

/// Skylake-Client without TSX: macOS boots on it without an OpenCore CPUID spoof, and it has no
/// AVX-512, whose state XNU mishandles on AMD hosts (docs/qemu.md, "x86-64 macOS guests: the CPU
/// model").
pub fn default_cpu_model() -> String {
    "Skylake-Client-noTSX-IBRS".into()
}

pub fn default_cpu_flags() -> String {
    "+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check".into()
}

fn default_disk_port() -> u32 {
    3
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vmapple {
    /// The auxiliary storage Virtualization.framework made, relative to the VM's directory. QEMU
    /// boots a copy without its first 16 KiB block (`aux.img.trimmed`).
    pub aux: PathBuf,
    /// The machine identifier's ECID, which `-M vmapple,uuid=` takes.
    pub ecid: u64,
}

impl VmConfig {
    pub fn check(&self) -> Result<()> {
        check_name(&self.name)?;
        if self.cpus == 0 || self.cpus > 256 {
            bail!("cpus must be 1 to 256, not {}", self.cpus);
        }
        if self.memory_mib < 2048 {
            bail!("memory must be at least 2 GiB, not {} MiB", self.memory_mib);
        }
        self.display.check()?;
        match self.profile {
            Profile::Q35X86_64 if self.q35.is_none() => bail!("{}: no [q35] settings", self.name),
            Profile::VmappleArm64 if self.vmapple.is_none() => {
                bail!("{}: no [vmapple] settings", self.name)
            }
            _ => Ok(()),
        }
    }
}

impl VmConfig {
    /// What in these settings the VM's guest does not work with, each a sentence to show the
    /// user; empty for an arm64 VM. For an x86-64 VM: a display wider than
    /// [`X86_64_MAX_WIDTH`], or `host`, whose width is the host screen's, and audio off, without
    /// which Safari's streaming players fail (docs/qemu.md, "x86-64 macOS guests: audio").
    pub fn incompatibilities(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.profile != Profile::Q35X86_64 {
            return w;
        }
        let r = self.display.resolution.trim();
        let wide = r == "host" || parse_resolution(r).is_ok_and(|(w, _)| w > X86_64_MAX_WIDTH);
        if wide {
            let what = if r == "host" {
                "Resolution host gives this x86-64 VM the host screen's width".to_string()
            } else {
                format!("Resolution {r} is too wide for an x86-64 VM")
            };
            w.push(format!(
                "{what}: an x86-64 macOS guest's WindowServer crashes on a display wider than \
                 {X86_64_MAX_WIDTH} pixels (a bug in macOS), often at boot. Give the VM a \
                 resolution at most {X86_64_MAX_WIDTH} wide, such as 1920x1080 or 2048x1280."
            ));
        }
        if !self.audio.enabled {
            w.push(format!(
                "Audio is off, and without an audio device an x86-64 macOS guest's Safari can't \
                 play YouTube or other streaming video. Turn it on; to keep the guest silent, \
                 also run reims-ctl set {} --audio-backend none.",
                self.name
            ));
        }
        w
    }
}

/// The widest mode an x86-64 macOS guest runs. Its WindowServer draws a large window in tiles
/// 1024 pixels wide, its Metal plugin refuses every bottom-row tile but the first, and with
/// three tiles across WindowServer reads past its list of tiles and dies. Measured with macOS
/// 26.6.1: 2560x1600 died at 4 boots of 4 on an RTX 3090 and at about half on an Iris Xe,
/// 2048x1280 and 1920x1080 at none.
pub const X86_64_MAX_WIDTH: u32 = 2048;

pub fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with(['-', '.'])
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !ok {
        bail!(
            "a VM's name is 1 to 64 letters, digits, '-', '_' or '.', not starting with '-' or '.': {name:?}"
        );
    }
    Ok(())
}

/// A locally administered MAC in QEMU's 52:54:00 prefix.
pub fn random_mac() -> String {
    let b = random_bytes::<3>();
    format!("52:54:00:{:02x}:{:02x}:{:02x}", b[0], b[1], b[2])
}

fn random_bytes<const N: usize>() -> [u8; N] {
    use std::io::Read;
    let mut b = [0u8; N];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .is_err()
    {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let seed = t.as_nanos() ^ (std::process::id() as u128) << 64;
        for (i, x) in b.iter_mut().enumerate() {
            *x = (seed >> (i * 8)) as u8;
        }
    }
    b
}

/// A size such as `8G`, `8GiB`, `512M` or `1.5T`, in MiB; a bare number is GiB.
pub fn parse_size_mib(s: &str) -> Result<u64> {
    let t = s.trim();
    let split = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, unit) = t.split_at(split);
    let n: f64 = num
        .parse()
        .with_context(|| format!("not a size: {s:?} (try 8G or 512M)"))?;
    let mul = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "g" | "gb" | "gib" => 1024.0,
        "m" | "mb" | "mib" => 1.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0,
        _ => bail!("not a size: {s:?} (try 8G or 512M)"),
    };
    let mib = (n * mul).round();
    if !(1.0..=1e12).contains(&mib) {
        bail!("not a size: {s:?}");
    }
    Ok(mib as u64)
}

pub fn format_mib(mib: u64) -> String {
    if mib >= 1024 && mib.is_multiple_of(1024) {
        format!("{}G", mib / 1024)
    } else {
        format!("{mib}M")
    }
}

#[derive(Clone, Debug)]
pub struct Vm {
    pub dir: PathBuf,
    pub config: VmConfig,
}

impl Vm {
    pub fn load(dir: &Path) -> Result<Vm> {
        let file = dir.join(CONFIG_FILE);
        let text =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        let config: VmConfig =
            toml::from_str(&text).with_context(|| format!("parse {}", file.display()))?;
        Ok(Vm {
            dir: dir.to_path_buf(),
            config,
        })
    }

    pub fn save(&self) -> Result<()> {
        self.config.check()?;
        let text = toml::to_string_pretty(&self.config)?;
        let file = self.dir.join(CONFIG_FILE);
        let tmp = self.dir.join(format!(".{CONFIG_FILE}.tmp"));
        std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &file).with_context(|| format!("write {}", file.display()))?;
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// `p` against the VM's directory, unless absolute.
    pub fn path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.dir.join(p)
        }
    }

    pub fn disk(&self) -> PathBuf {
        self.path(&self.config.disk)
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.dir.join("logs")
    }

    pub fn qemu_log(&self) -> PathBuf {
        self.logs_dir().join("qemu.log")
    }

    pub fn serial_log(&self) -> PathBuf {
        self.logs_dir().join("serial.log")
    }

    pub fn pidfile(&self) -> PathBuf {
        self.dir.join("qemu.pid")
    }

    /// The QMP socket: in the VM's directory where the path fits a Unix socket's 104 bytes
    /// (macOS's limit; Linux's is 108), else in the runtime directory.
    pub fn qmp_socket(&self) -> PathBuf {
        let own = self.dir.join("qmp.sock");
        if own.as_os_str().len() < 100 {
            own
        } else {
            paths::runtime_dir().join(format!("{}.qmp", self.config.name))
        }
    }

    /// The trimmed auxiliary storage QEMU boots (vmapple).
    pub fn aux_trimmed(&self) -> Option<PathBuf> {
        let v = self.config.vmapple.as_ref()?;
        let aux = self.path(&v.aux);
        let mut name = aux.file_name()?.to_os_string();
        name.push(".trimmed");
        Some(aux.with_file_name(name))
    }
}

// ---------------------------------------------------------------- the registry

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default, rename = "vm")]
    pub vms: Vec<Entry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub name: String,
    pub dir: PathBuf,
}

impl Registry {
    pub fn load() -> Result<Registry> {
        let file = paths::registry_file();
        match std::fs::read_to_string(&file) {
            Ok(text) => toml::from_str(&text).with_context(|| format!("parse {}", file.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Registry::default()),
            Err(e) => Err(e).with_context(|| format!("read {}", file.display())),
        }
    }

    pub fn save(&self) -> Result<()> {
        let file = paths::registry_file();
        std::fs::create_dir_all(file.parent().unwrap())?;
        let tmp = file.with_extension("toml.tmp");
        std::fs::write(&tmp, toml::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, &file)?;
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.vms.iter().find(|e| e.name == name)
    }

    pub fn add(&mut self, name: &str, dir: &Path) -> Result<()> {
        if self.find(name).is_some() {
            bail!("a VM named {name} already exists");
        }
        self.vms.push(Entry {
            name: name.into(),
            dir: dir.to_path_buf(),
        });
        self.vms.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Option<Entry> {
        let i = self.vms.iter().position(|e| e.name == name)?;
        Some(self.vms.remove(i))
    }

    /// The VM named `name`.
    pub fn open(&self, name: &str) -> Result<Vm> {
        let e = self
            .find(name)
            .with_context(|| format!("no VM named {name} (reims-ctl list shows them)"))?;
        Vm::load(&e.dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidpi_densities_are_ones_qemu_accepts() {
        for w in [1920, 2048, 2560, 2880, 3024, 3456, 3840, 5120, 6016] {
            let ppi = hidpi_ppi(w).unwrap_or_else(|| panic!("no density for {w}"));
            assert!(!guest_scale_unknown(w, ppi), "{w}@{ppi}");
            let cm = guest_centimeters(w, ppi);
            assert!(
                w as f64 / cm as f64 * 2.54 >= 146.667,
                "{w}@{ppi} is not HiDPI ({cm} cm)"
            );
        }
        assert_eq!(hidpi_ppi(1280), None);
    }

    #[test]
    fn densities_match_apple_gfx_examples() {
        // docs/qemu.md: 2560x1600@200 runs HiDPI; @148 reads 44 cm, @146 45 cm.
        assert_eq!(guest_centimeters(2560, 148), 44);
        assert_eq!(guest_centimeters(2560, 146), 45);
        assert!(!guest_scale_unknown(2560, 200));
    }

    #[test]
    fn resolution_property() {
        let mut d = Display::default();
        assert_eq!(d.qemu_resolution().unwrap(), "1920x1080");
        d.hidpi = true;
        d.resolution = "3840x2160".into();
        assert!(d.qemu_resolution().unwrap().starts_with("3840x2160@"));
        assert_eq!(d.looks_like(), Some((1920, 1080)));
        d.resolution = "1280x800".into();
        assert!(d.qemu_resolution().is_err());
        assert_eq!(d.looks_like(), None);
        d.resolution = "host".into();
        assert!(d.qemu_resolution().is_err());
        d.hidpi = false;
        assert_eq!(d.qemu_resolution().unwrap(), "host");
        d.resolution = "19x".into();
        assert!(d.qemu_resolution().is_err());
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size_mib("8G").unwrap(), 8192);
        assert_eq!(parse_size_mib("8").unwrap(), 8192);
        assert_eq!(parse_size_mib("512M").unwrap(), 512);
        assert_eq!(parse_size_mib("1.5GiB").unwrap(), 1536);
        assert_eq!(parse_size_mib("1T").unwrap(), 1024 * 1024);
        assert!(parse_size_mib("eight").is_err());
        assert!(parse_size_mib("8Q").is_err());
        assert_eq!(format_mib(8192), "8G");
        assert_eq!(format_mib(1536), "1536M");
    }

    #[test]
    fn names() {
        assert!(check_name("macos-26.1_b").is_ok());
        for bad in ["", "-a", ".a", "a b", "a/b", &"x".repeat(65)] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }

    fn x86_64_config() -> VmConfig {
        VmConfig {
            name: "t".into(),
            profile: Profile::Q35X86_64,
            cpus: 4,
            memory_mib: 8192,
            disk: "disk.qcow2".into(),
            disk_format: ImageFormat::Qcow2,
            stop_timeout_secs: 90,
            extra_args: vec!["-boot".into(), "menu=on".into()],
            display: Display::default(),
            gpu: Gpu {
                driver_file: Some("/usr/share/vulkan/icd.d/nvidia_icd.json".into()),
                device_id: Some(0x2204),
                name: None,
            },
            audio: Audio::default(),
            network: Network {
                enabled: true,
                mac: "52:54:00:01:02:03".into(),
                ssh_port: Some(2222),
            },
            logs: Logs::default(),
            q35: Some(Q35 {
                opencore: "opencore.qcow2".into(),
                opencore_format: ImageFormat::Qcow2,
                firmware_code: "OVMF_CODE.fd".into(),
                firmware_vars: "OVMF_VARS.fd".into(),
                installer: Some("/isos/BaseSystem.img".into()),
                cpu_model: default_cpu_model(),
                cpu_flags: default_cpu_flags(),
                disk_port: 3,
            }),
            vmapple: None,
            extra_env: BTreeMap::from([("A".into(), "b".into())]),
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        assert_eq!(toml::from_str::<VmConfig>(&text).unwrap(), c);
    }

    #[test]
    fn incompatibilities_are_an_x86_64_vms_wide_displays_and_silence() {
        let mut c = x86_64_config();
        assert!(c.incompatibilities().is_empty());
        c.display.resolution = "2048x1280".into();
        assert!(c.incompatibilities().is_empty());
        c.display.resolution = "2560x1600".into();
        let w = c.incompatibilities();
        assert_eq!(w.len(), 1);
        assert!(
            w[0].starts_with("Resolution 2560x1600 is too wide"),
            "{w:?}"
        );
        c.display.resolution = "host".into();
        c.audio.enabled = false;
        let w = c.incompatibilities();
        assert_eq!(w.len(), 2);
        assert!(w[0].starts_with("Resolution host gives"), "{w:?}");
        assert!(
            w[1].contains("reims-ctl set t --audio-backend none"),
            "{w:?}"
        );
        // A resolution check() refuses is its to report.
        c.display.resolution = "19x".into();
        assert_eq!(c.incompatibilities().len(), 1);
        c.display.resolution = "host".into();
        c.profile = Profile::VmappleArm64;
        assert!(c.incompatibilities().is_empty());
    }
}

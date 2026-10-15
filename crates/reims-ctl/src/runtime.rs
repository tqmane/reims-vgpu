//! Starting, finding and stopping a VM's QEMU.
//!
//! QEMU runs detached: in a session of its own (so closing the terminal does not end it), and
//! reparented away from reims-ctl by a double fork (so a TUI that stays open never holds an
//! exited QEMU as a zombie). Its pid is the one QEMU writes to the VM's `qemu.pid`, counted only
//! while that process runs and its command line names the pid file, so a stale file or a reused
//! pid is never taken for the VM. A VM started from a printed script is found the same way.
//!
//! QEMU's stdout and stderr go to `logs/qemu.log`, kept for the last five starts
//! (`qemu.1.log`, ...), the serial console's likewise.

use crate::host::{Host, Os, Profile};
use crate::qemu::{self, Ctx, Invocation};
use crate::qmp;
use crate::vm::Vm;
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::time::{Duration, Instant};

/// QEMU's pid, while it runs.
pub fn pid(vm: &Vm) -> Option<i32> {
    let pid: i32 = std::fs::read_to_string(vm.pidfile())
        .ok()?
        .trim()
        .parse()
        .ok()?;
    if pid <= 0 || !alive(pid) {
        return None;
    }
    let pidfile = vm.pidfile();
    let needle = pidfile.as_os_str().as_encoded_bytes();
    let cmd = cmdline(pid)?;
    cmd.windows(needle.len())
        .any(|w| w == needle)
        .then_some(pid)
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks the pid.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(target_os = "linux")]
fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let b = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    // A zombie's command line is empty.
    (!b.is_empty()).then_some(b)
}

#[cfg(target_os = "macos")]
fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
    let mut size: libc::size_t = 0;
    // SAFETY: a size query; mib names a valid sysctl.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let mut buf = vec![0u8; size];
    // SAFETY: buf holds size bytes.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            buf.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    buf.truncate(size);
    Some(buf)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cmdline(_pid: i32) -> Option<Vec<u8>> {
    None
}

pub fn running(vm: &Vm) -> bool {
    pid(vm).is_some()
}

/// What is wrong with the host or the VM's files for a start, if anything.
pub fn preflight(vm: &Vm, host: Host) -> Result<()> {
    let c = &vm.config;
    c.check()?;
    if !crate::host::Profile::supported_on(host).contains(&c.profile) {
        bail!("{} is a {} VM, which {host} cannot run", c.name, c.profile);
    }
    let need = |p: &Path, what: &str| -> Result<()> {
        if !p.exists() {
            bail!("{what} {} is missing", p.display());
        }
        Ok(())
    };
    need(&vm.disk(), "the disk")?;
    match c.profile {
        Profile::Q35X86_64 => {
            let q = c.q35.as_ref().unwrap();
            need(&vm.path(&q.opencore), "the OpenCore image")?;
            need(&vm.path(&q.firmware_code), "the firmware")?;
            need(&vm.path(&q.firmware_vars), "the firmware variables")?;
            if let Some(i) = q.installer.as_ref().map(|i| vm.path(i))
                && !i.exists()
            {
                bail!(
                    "the installer {} is missing: reims-ctl eject {} takes it out (in the \
                     interface, clear Installer image in its edit form, e)",
                    i.display(),
                    c.name
                );
            }
        }
        Profile::VmappleArm64 => {
            need(
                Path::new(qemu::AVPBOOTER),
                "Virtualization.framework's boot ROM",
            )?;
            let v = c.vmapple.as_ref().unwrap();
            need(&vm.path(&v.aux), "the auxiliary storage")?;
        }
    }
    if host.os == Os::Linux {
        let kvm = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm");
        if let Err(e) = kvm {
            bail!(
                "/dev/kvm cannot be opened ({e}): enable virtualization in the firmware, and add \
                 yourself to the kvm group (sudo usermod -aG kvm $USER, then log in again)"
            );
        }
    }
    Ok(())
}

/// The reminder where an x86-64 VM's macOS gets installed.
pub const SET_METAL_ERROR_MODE: &str = "set Metal's error mode once macOS is installed (README)";

/// The host's `vm.max_map_count` (Linux).
pub fn host_max_map_count() -> Option<u64> {
    std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// The `vm.max_map_count` a guest of `memory_mib` needs on Linux, and the value to suggest
/// raising it to. docs/qemu.md, "Linux hosts: vm.max_map_count": guest memory views take a
/// mapping per run of 16 KiB pages; the stock 65,530 holds about 770 MiB of them.
pub fn max_map_count(memory_mib: u64) -> (u64, u64) {
    let want = (memory_mib * 64).max(262_144) + 16_384 + 65_536;
    // 2097152 covers a guest of up to 30 GiB, a new VM's default of up to 16 GiB among them.
    (want, want.next_power_of_two().max(2_097_152))
}

/// How to install macOS in an x86-64 VM that has an installer, for the CLI and the interface's
/// details; None for other VMs. An offline installer, which reims-util names
/// `…-offline-installer.img`, installs from its script; any other installer boots Recovery,
/// whose Reinstall macOS installs it.
pub fn install_steps(vm: &Vm) -> Option<String> {
    let q = vm.config.q35.as_ref()?;
    let installer = q.installer.as_ref()?;
    let offline = installer
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with("-offline-installer.img"));
    let last = if offline {
        "open Utilities > Terminal and run: sh /Volumes/Installer/install-macos.sh"
    } else {
        "pick Reinstall macOS"
    };
    Some(format!(
        "To install macOS, start {name}: OpenCore boots its installer into macOS Recovery. If the \
         VM's disk is new, open Disk Utility, erase it (QEMU HARDDISK Media) as APFS and quit \
         Disk Utility. Then {last}. Once macOS is installed, eject the installer (reims-ctl \
         eject {name}) and set Metal's error mode (README).",
        name = vm.name()
    ))
}

/// Warnings about the host, or the guest's last boot, for a start that would still work.
pub fn warnings(vm: &Vm, host: Host) -> Vec<String> {
    let mut w = Vec::new();
    if vm.config.profile == Profile::Q35X86_64 {
        // docs/qemu.md, "x86-64 macOS guests: Metal's error mode": without it, macOS 26's icon
        // service ends as it starts, again and again, and the library logs each command it left
        // queued. From Recovery, the `cd` expands the Data volume's name, and `defaults` leaves
        // the file at mode 0600, which the user's processes cannot read.
        let n = emptied_entry_lines(&vm.qemu_log());
        if n >= 3 {
            w.push(format!(
                "QEMU's latest log has {n} lines on GPU work left by crashed macOS processes. On \
                 x86-64 macOS 26 that's most likely its icon service, crashing because Metal's \
                 error mode isn't set, and the desktop can stay on its wallpaper. To set it, boot \
                 Recovery from OpenCore's picker, open Utilities > Terminal and run: cd \
                 /Volumes/*Data/Library/Preferences && defaults write \"$PWD/.GlobalPreferences\" \
                 METAL_ERROR_MODE -int 5 && chmod 644 .GlobalPreferences.plist && reboot"
            ));
        }
    }
    if host.os == Os::Linux {
        if let Some(n) = host_max_map_count() {
            let (want, raise) = max_map_count(vm.config.memory_mib);
            if n < want {
                w.push(format!(
                    "vm.max_map_count is {n}: a {} GiB guest needs about {want}. Raise it with \
                     sudo sysctl -w vm.max_map_count={raise} (and in /etc/sysctl.d to keep it)",
                    vm.config.memory_mib / 1024
                ));
            }
        }
        // docs/qemu.md, "Linux hosts: AMD GPUs and udmabuf": RADV imports the guest's memory as
        // dma-bufs udmabuf makes.
        if may_run_on_radv(&vm.config.gpu) {
            let usable = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/udmabuf");
            if let Err(e) = usable {
                w.push(format!(
                    "/dev/udmabuf cannot be opened ({e}): an AMD GPU needs it for the guest's \
                     memory, or the guest dies at boot. Add yourself to the group that owns it \
                     (kvm on Ubuntu; sudo usermod -aG kvm $USER, then log in again)"
                ));
            }
        }
    }
    w
}

/// Whether a VM may run on RADV, Mesa's driver for AMD GPUs: its GPU is one, or none is set and
/// the host has an AMD GPU and no NVIDIA one, which the library would pick first as discrete.
fn may_run_on_radv(gpu: &crate::vm::Gpu) -> bool {
    let radv = |file: &Path| {
        file.file_name()
            .is_some_and(|f| f.to_string_lossy().starts_with("radeon_icd"))
    };
    match &gpu.driver_file {
        Some(file) => radv(file) || gpu.name.as_deref().is_some_and(|n| n.contains("RADV")),
        None => {
            let vendors: Vec<String> = std::fs::read_dir("/sys/class/drm")
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|card| std::fs::read_to_string(card.path().join("device/vendor")).ok())
                .map(|v| v.trim().to_owned())
                .collect();
            vendors.iter().any(|v| v == "0x1002") && !vendors.iter().any(|v| v == "0x10de")
        }
    }
}

/// How many lines of a QEMU log, since the guest's last reset, name an object entry the guest
/// had emptied: the library logs them at its warn level when a command of a guest process that
/// has ended runs, and `device reset` at its info level. The TUI asks at every draw, so the count
/// carries on from the bytes already read while the file is the same (by inode) and has not
/// shrunk.
fn emptied_entry_lines(log: &Path) -> usize {
    use std::os::unix::fs::MetadataExt;
    const EMPTIED: &[u8] = b"has an empty entry";
    static READ: std::sync::Mutex<(std::path::PathBuf, u64, u64, usize)> =
        std::sync::Mutex::new((std::path::PathBuf::new(), 0, 0, 0));
    let Ok(mut f) = std::fs::File::open(log) else {
        return 0;
    };
    let Ok(meta) = f.metadata() else { return 0 };
    let mut r = READ.lock().unwrap_or_else(|e| e.into_inner());
    if r.0 != log || r.1 != meta.ino() || r.2 > meta.len() {
        *r = (log.to_path_buf(), meta.ino(), 0, 0);
    }
    let mut buf = Vec::new();
    if f.seek(SeekFrom::Start(r.2)).is_ok() {
        let _ = f.take(meta.len() - r.2).read_to_end(&mut buf);
    }
    // Whole lines only: the rest is read again once its line ends.
    let end = buf.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    for l in buf[..end].split(|&b| b == b'\n') {
        if l.ends_with(b"device reset") {
            r.3 = 0;
        } else if l.windows(EMPTIED.len()).any(|w| w == EMPTIED) {
            r.3 += 1;
        }
    }
    r.2 += end as u64;
    r.3
}

/// Keeps the last five of a log: `x.log` becomes `x.1.log`, and so on.
fn rotate(path: &Path) {
    let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
    let dir = path.parent().unwrap();
    let nth = |n: u32| dir.join(format!("{stem}.{n}.log"));
    let _ = std::fs::remove_file(nth(4));
    for n in (1..4).rev() {
        let _ = std::fs::rename(nth(n), nth(n + 1));
    }
    let _ = std::fs::rename(path, nth(1));
}

/// Writes the trimmed auxiliary storage a vmapple VM boots, if missing: the image without its
/// first 16 KiB block (docs/system/arm/vmapple.rst in vendor/qemu).
pub fn ensure_aux_trimmed(vm: &Vm) -> Result<()> {
    let Some(trimmed) = vm.aux_trimmed() else {
        return Ok(());
    };
    if trimmed.exists() {
        return Ok(());
    }
    let aux = vm.path(&vm.config.vmapple.as_ref().unwrap().aux);
    let mut src = std::fs::File::open(&aux).with_context(|| format!("open {}", aux.display()))?;
    src.seek(SeekFrom::Start(0x4000))?;
    let tmp = trimmed.with_extension("trimmed.tmp");
    let mut dst = std::fs::File::create(&tmp)?;
    std::io::copy(&mut src, &mut dst)?;
    dst.sync_all()?;
    std::fs::rename(&tmp, &trimmed)?;
    Ok(())
}

pub fn context(vm: &Vm, host: Host) -> Result<Ctx> {
    let qemu = crate::install::qemu_binary(vm.config.profile)?;
    let backends = crate::install::audio_backends(&qemu);
    let audio_backend = crate::install::default_audio_backend(&backends);
    if vm.config.audio.enabled
        && let Some(b) = &vm.config.audio.backend
        && !backends.is_empty()
        && !backends.contains(b)
    {
        bail!(
            "this QEMU has no {b} audio backend; it has {}",
            backends.join(", ")
        );
    }
    Ok(Ctx {
        host,
        qemu,
        audio_backend,
    })
}

/// The invocation `start` would run, with QEMU's real path where it is installed.
pub fn invocation_for_print(vm: &Vm, host: Host) -> Result<Invocation> {
    let ctx = context(vm, host).unwrap_or_else(|_| {
        let qemu = crate::install::qemu_binary_or_default(vm.config.profile);
        Ctx {
            host,
            qemu,
            audio_backend: crate::install::default_audio_backend(&[]),
        }
    });
    qemu::invocation(vm, &ctx)
}

pub fn now_stamp(fmt: &str) -> String {
    // SAFETY: time and localtime_r fill the structs given; strftime writes at most buf's size.
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = [0u8; 64];
        let f = std::ffi::CString::new(fmt).unwrap();
        let n = libc::strftime(buf.as_mut_ptr().cast(), buf.len(), f.as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

/// Starts the VM, and returns QEMU's pid once QMP answers.
pub fn start(vm: &Vm, host: Host) -> Result<i32> {
    if let Some(p) = pid(vm) {
        bail!("{} already runs (pid {p})", vm.name());
    }
    preflight(vm, host)?;
    if vm.config.profile == Profile::VmappleArm64 {
        ensure_aux_trimmed(vm)?;
    }
    let ctx = context(vm, host)?;
    let inv = qemu::invocation(vm, &ctx)?;

    let logs = vm.logs_dir();
    std::fs::create_dir_all(&logs)?;
    rotate(&vm.qemu_log());
    if vm.config.logs.enabled {
        rotate(&vm.serial_log());
    }
    let sock = vm.qmp_socket();
    if let Some(d) = sock.parent() {
        std::fs::create_dir_all(d)?;
    }
    let _ = std::fs::remove_file(&sock);
    let _ = std::fs::remove_file(vm.pidfile());

    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(vm.qemu_log())?;
    writeln!(
        log,
        "reims-ctl {} started {} at {}",
        env!("CARGO_PKG_VERSION"),
        vm.name(),
        now_stamp("%Y-%m-%d %H:%M:%S %z")
    )?;
    log.write_all(inv.shell_script("the command").as_bytes())?;
    writeln!(log, "----")?;
    let err = log.try_clone()?;

    let mut cmd = inv.command();
    cmd.current_dir(&vm.dir)
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(err);
    // SAFETY: only async-signal-safe calls between fork and exec: setsid, fork, _exit.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            match libc::fork() {
                -1 => Err(std::io::Error::last_os_error()),
                0 => Ok(()),
                _ => libc::_exit(0),
            }
        });
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("run {}", inv.program.display()))?;
    let _ = child.wait();

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut seen = false;
    loop {
        if let Some(p) = pid(vm) {
            seen = true;
            if qmp::Qmp::connect(&sock, Duration::from_secs(2)).is_ok() {
                return Ok(p);
            }
        } else if seen || Instant::now() > deadline - Duration::from_secs(50) {
            // QEMU wrote its pid file and exited, or never wrote it.
            std::thread::sleep(Duration::from_millis(200));
            if pid(vm).is_none() {
                bail!("QEMU exited:\n{}", tail(&vm.qemu_log(), 15));
            }
        }
        if Instant::now() > deadline {
            match pid(vm) {
                Some(p) => return Ok(p),
                None => bail!("QEMU did not start:\n{}", tail(&vm.qemu_log(), 15)),
            }
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// Stops the VM: macOS's Control-Option-Command-Power on the USB keyboard ("quit all apps,
/// then shut down", which asks about unsaved documents), then, a third of `timeout` in, the
/// machine's power button (ACPI's on q35, a GPIO on vmapple), and at `timeout` a power-off; with
/// `force`, the power-off at once. `progress` hears each step.
///
/// The keys come first because the power button alone does not shut a guest down: a macOS 26
/// vmapple guest did nothing with vmapple's in 90 s, where the keys shut it down in under 6 s
/// (2026-10-11, M3 Max).
pub fn stop(vm: &Vm, force: bool, timeout: Duration, mut progress: impl FnMut(&str)) -> Result<()> {
    let Some(p) = pid(vm) else {
        progress("not running");
        return Ok(());
    };
    let sock = vm.qmp_socket();
    if !force {
        let chord: Vec<_> = ["ctrl", "alt", "meta_l", "power"]
            .iter()
            .map(|k| json!({ "type": "qcode", "data": k }))
            .collect();
        let keys = json!({ "keys": chord, "hold-time": 200 });
        let asked = match qmp::run(&sock, "send-key", Some(keys)) {
            Ok(_) => {
                progress("asked the guest to shut down (Control-Option-Command-Power)");
                true
            }
            Err(e) => {
                progress(&format!("{e}"));
                false
            }
        };
        let start = Instant::now();
        let mut button = false;
        while start.elapsed() < timeout {
            if !alive(p) {
                progress("shut down");
                return Ok(());
            }
            if !button && (!asked || start.elapsed() >= timeout / 3) {
                button = true;
                if qmp::run(&sock, "system_powerdown", None).is_ok() {
                    progress("still running: pressed its power button");
                }
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        progress("the guest did not shut down in time: powering off");
    }
    power_off(vm, p, &mut progress)
}

fn power_off(vm: &Vm, p: i32, progress: &mut impl FnMut(&str)) -> Result<()> {
    let _ = qmp::run(&vm.qmp_socket(), "quit", None);
    if wait_gone(p, Duration::from_secs(10)) {
        progress("powered off");
        return Ok(());
    }
    // SAFETY: p is our QEMU's pid, checked by pid().
    unsafe { libc::kill(p, libc::SIGTERM) };
    if wait_gone(p, Duration::from_secs(5)) {
        progress("powered off (SIGTERM)");
        return Ok(());
    }
    // SAFETY: as above.
    unsafe { libc::kill(p, libc::SIGKILL) };
    if wait_gone(p, Duration::from_secs(5)) {
        progress("killed (SIGKILL)");
        return Ok(());
    }
    bail!("QEMU (pid {p}) did not exit")
}

fn wait_gone(p: i32, t: Duration) -> bool {
    let deadline = Instant::now() + t;
    while Instant::now() < deadline {
        if !alive(p) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    !alive(p)
}

/// The last `n` lines of a text file.
pub fn tail(path: &Path, n: usize) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(64 * 1024);
    let _ = f.seek(SeekFrom::Start(start));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Gpu;
    use std::path::PathBuf;

    #[test]
    fn the_suggested_max_map_count_covers_new_vms() {
        // A new VM's default is up to 16 GiB; CachyOS's stock 1048576 falls short of it.
        assert_eq!(max_map_count(16 * 1024), (1_130_496, 2_097_152));
        assert_eq!(max_map_count(4096), (344_064, 2_097_152));
        let (want, raise) = max_map_count(30 * 1024);
        assert!(want <= raise && raise == 2_097_152);
        assert_eq!(max_map_count(32 * 1024).1, 4_194_304);
    }

    /// A VM pinned to Mesa's radeon driver, or to a GPU named RADV, may run on RADV; one pinned
    /// to NVIDIA's may not
    #[test]
    fn a_pinned_gpu_decides_radv() {
        let gpu = |file: &str, name: &str| Gpu {
            driver_file: Some(PathBuf::from(file)),
            device_id: None,
            name: Some(name.to_owned()),
        };
        assert!(may_run_on_radv(&gpu(
            "/usr/share/vulkan/icd.d/radeon_icd.json",
            "AMD Radeon 780M Graphics (RADV PHOENIX)"
        )));
        assert!(may_run_on_radv(&gpu(
            "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "AMD Radeon 780M Graphics"
        )));
        assert!(!may_run_on_radv(&gpu(
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "NVIDIA GeForce RTX 3090"
        )));
    }

    #[test]
    fn emptied_entry_lines_count_on_from_what_was_read() {
        let d = std::env::temp_dir().join(format!("reims-ctl-emptied-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let log = d.join("qemu.log");
        let fault = "reims-vgpu WARN reims_vgpu_core::tasks::local::exec: task 2: exec faulted: \
                     resource 3 not found: object 3 has an empty entry\n";
        let append = |s: &str| {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log)
                .unwrap();
            f.write_all(s.as_bytes()).unwrap();
        };
        assert_eq!(emptied_entry_lines(&log), 0);
        append("reims-ctl 0.1.0 started t\n");
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        // A line still being written counts once it ends.
        append(&fault[..40]);
        assert_eq!(emptied_entry_lines(&log), 1);
        append(&fault[40..]);
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 3);
        // The guest's reboot in the same run (as after setting the key from Recovery) counts
        // from the device's reset.
        append("reims-vgpu INFO reims_vgpu_core::device::lifecycle: device reset\n");
        assert_eq!(emptied_entry_lines(&log), 0);
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        // A start's rotation leaves a new file, counted from its start.
        std::fs::rename(&log, d.join("qemu.1.log")).unwrap();
        append(&"x".repeat(4096));
        append("\n");
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }
}

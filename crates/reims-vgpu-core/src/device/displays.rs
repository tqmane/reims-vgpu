//! The device's display ports: the guest display each port has once the guest gives it a page, the
//! host's display parked on a port with no guest display, their attach and detach, and the EFI
//! display.
//!
//! The ports live in the device's `displays` ([`DisplaysLocked`], a staged mutex). What must not
//! wait on the lock (MMIO, the EFI display's binding, the teardown's reads) goes through their
//! unlocked view (`displays.atomics()`, [`DisplaysAtomics`](crate::device::DisplaysAtomics)). A
//! store under the lock is published to that view at a fixed point: a parked client's store at once
//! ([`set_client_at`]), and the guest display slot's at the end of a page change.
//!
//! **Ordering contracts** (testing.md 7):
//! - The displays lock, then a guest display's mutex: an attach or detach sets the guest display's
//!   client under the displays lock; a display's end detaches on the thread of the host's last
//!   release, waiting there for both.
//! - A page with no root node keeps the new guest display (and its page's mapping) for good.
//! - Page 0 clears the root word only while running.
//! - Detaching through a guest display leaves the display's own guest display set.
//! - The EFI display binds only port 0's parked client; any non-zero value enables it.

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use crate::device::{DeviceCore, DisplaysLocked, RunState};
use crate::display::descriptor::DisplayDescriptor;
use crate::display::efi::EfiDisplay;
use crate::display::guest_display::GuestDisplay;
use crate::display::{Display, DisplayCore, DisplayHandle, DisplayRefusal};
use crate::host::{HostTraceRange, PhysRange, TraceRangeHandler};
use crate::iosurface::{HostDeviceCore, IoSurfaceHostDevice};
use crate::num::{ToUsize, Truncate};
use crate::platform::Platform;
use crate::raise::{DeathId, Kept, OnFault, RaiseId, RaiseResult, keep_forever};
use crate::rt::Hold;
use crate::sync::StagedGuard;

/// Where the display state page words start in the root node (`rootNode[0x240 + port]`).
pub const ROOT_DISPLAY_WORDS: u64 = 0x240;

/// What the display ports raise. The FIFO's opcode 1 and the doorbell catch it; a resume logs
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// A port at or above the device's port count.
    PortOutOfRange,
    /// A page for a port that has a guest display.
    PortInUse,
    /// Page 0, or a guest display lookup, for a port with none.
    PortNotInUse,
    /// A page before the root node.
    NoRootNode,
}

impl RaiseKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::PortOutOfRange => "device.displays.out_of_range",
            RaiseKind::PortInUse => "device.displays.in_use",
            RaiseKind::PortNotInUse => "device.displays.not_in_use",
            RaiseKind::NoRootNode => "device.displays.no_root_node",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Displays(k)
    }
}

/// The device's displays whose end has not detached them yet, each by the host reference's
/// identity (the `Weak` the ports and the guest display hold) and with its core. A `Weak` stops
/// upgrading the moment the last release begins, but the ports and the guest display still reach
/// the display while that release waits in the detach: the identity maps to the core here
/// ([`DeviceCore::display_core`]). A display registers once the host's reference exists, before
/// its attach, and its end deregisters it once the unplug has detached it, before it releases
/// what the core holds (the entry's `Weak` of the core would keep that release from running).
///
/// A leaf lock, never held across a call, and apart from [`DisplaysLocked`], since a display's
/// client is set under the displays lock.
pub(crate) struct DisplayRegistry<P: Platform> {
    entries: parking_lot::Mutex<Vec<DisplayEntry<P>>>,
}

/// One display in a [`DisplayRegistry`].
struct DisplayEntry<P: Platform> {
    /// The host reference's identity.
    display: Weak<Display<P>>,
    /// Its core.
    core: Weak<DisplayCore<P>>,
}

impl<P: Platform> DisplayRegistry<P> {
    /// No display.
    pub(crate) fn new() -> DisplayRegistry<P> {
        DisplayRegistry {
            entries: parking_lot::Mutex::new(Vec::new()),
        }
    }
}

impl<P: Platform> DeviceCore<P> {
    /// Registers `display`, whose host reference now exists, with its core
    /// ([`DisplayRegistry`]).
    pub(crate) fn register_display(&self, display: &Arc<Display<P>>) {
        let entry = DisplayEntry {
            display: Arc::downgrade(display),
            core: Arc::downgrade(display.core()),
        };
        self.display_cores.entries.lock().push(entry);
    }

    /// Deregisters the display whose core is `core`: its end has detached it.
    pub(crate) fn deregister_display(&self, core: &Arc<DisplayCore<P>>) {
        self.display_cores
            .entries
            .lock()
            .retain(|e| !std::ptr::eq(e.core.as_ptr(), Arc::as_ptr(core)));
    }

    /// The core behind an unretained display of this device (a port's parked client, a guest
    /// display's client, a display's own blocks). It takes no reference to the host's display, so
    /// the host's last release runs the end on the host's thread whatever the caller holds, and
    /// it still answers while that end waits in the detach, where [`Weak::upgrade`] already fails.
    /// `None` once the end has detached the display, or for a display the initializer never
    /// finished.
    pub fn display_core(&self, display: &Weak<Display<P>>) -> Option<Arc<DisplayCore<P>>> {
        self.display_cores
            .entries
            .lock()
            .iter()
            .find(|e| Weak::ptr_eq(&e.display, display))
            .and_then(|e| e.core.upgrade())
    }
}

/// Where the display ports end the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// The root node's display word written through NULL (unreachable: the node is tested).
    RootWordNull,
    /// A port with no claim has a parked client.
    ParkedOnFreePort,
    /// A detach of a port out of range.
    DetachOutOfRange,
    /// A detach of a port not claimed.
    DetachNotClaimed,
    /// A detach of a display that is not the port's guest display's client.
    DetachOtherGuestDisplayClient,
    /// A detach of a display that is not the port's parked client.
    DetachOtherParkedClient,
    /// A detach of a parked display that holds a guest display.
    DetachParkedWithGuestDisplay,
    /// Page 0 for a port that has a parked client beside its guest display.
    ParkedBesideGuestDisplay,
    /// The EFI display bound to a display whose last release has begun (the runtime's
    /// `objc_storeWeak` refusal).
    EfiBindDeallocating,
}

impl DeathKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::RootWordNull => "device.displays.root_word_null",
            DeathKind::ParkedOnFreePort => "device.displays.parked_on_free_port",
            DeathKind::DetachOutOfRange => "device.displays.detach_out_of_range",
            DeathKind::DetachNotClaimed => "device.displays.detach_not_claimed",
            DeathKind::DetachOtherGuestDisplayClient => {
                "device.displays.detach_other_guest_display_client"
            }
            DeathKind::DetachOtherParkedClient => "device.displays.detach_other_parked_client",
            DeathKind::DetachParkedWithGuestDisplay => {
                "device.displays.detach_parked_with_guest_display"
            }
            DeathKind::ParkedBesideGuestDisplay => "device.displays.parked_beside_guest_display",
            DeathKind::EfiBindDeallocating => "device.displays.efi_bind_deallocating",
        }
    }

    /// The kind of death.
    pub(crate) const fn class(self) -> crate::raise::death::DeathKind {
        use crate::raise::death::DeathKind as K;
        match self {
            DeathKind::RootWordNull => K::NullDeref,
            DeathKind::EfiBindDeallocating => K::Abort,
            _ => K::Assert,
        }
    }

    /// The site a death of this kind names (tests).
    #[cfg(test)]
    pub(crate) fn site(self) -> crate::raise::death::DeathSite {
        crate::raise::death::DeathId::from(self).site()
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Displays(k)
    }
}

/// The port's bit in the busy mask: `1 << port` is an `int` (lsl w, port mod 32) sign-extended
/// to the 64-bit mask.
// The bit is an `int` shift; ports are below 8 here, so it never overflows.
fn port_bit(port: u64) -> u64 {
    crate::num::sign_extend_32(crate::num::bit32(port.low_u32()))
}

/// The `u32` word offset of port `port`'s display word in the root node.
fn display_word_offset(port: u64) -> u64 {
    ROOT_DISPLAY_WORDS.wrapping_add(port).wrapping_mul(4)
}

/// The port's guest display in the locked state (`None` past the ports).
fn guest_display_at<P: Platform>(d: &DisplaysLocked<P>, port: u64) -> Option<Arc<GuestDisplay<P>>> {
    d.guest_display.get(port.to_usize()).cloned().flatten()
}

/// The port's parked client in the locked state.
fn client_at<P: Platform>(d: &DisplaysLocked<P>, port: u64) -> Option<Weak<Display<P>>> {
    d.display_client.get(port.to_usize()).cloned().flatten()
}

/// Parks `client` on the port (or clears it), published at once to the unlocked readers (the
/// EFI display's bind).
fn set_client_at<P: Platform>(
    d: &mut StagedGuard<'_, DisplaysLocked<P>>,
    port: u64,
    client: Option<Weak<Display<P>>>,
) {
    if let Some(slot) = d.display_client.get_mut(port.to_usize()) {
        *slot = client;
    }
    d.publish();
}

/// Two unretained display pointers are the same object, nil included.
fn same_client<P: Platform>(a: Option<&Weak<Display<P>>>, b: Option<&Weak<Display<P>>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Weak::ptr_eq(a, b),
        _ => false,
    }
}

/// The port-range raise.
fn out_of_range(port: u64, count: u32) -> crate::raise::Raise {
    crate::raise::Raise::new(RaiseKind::PortOutOfRange).with_detail(format_args!(
        "display port {port} is past the last port, {}",
        count.wrapping_sub(1)
    ))
}

impl<P: Platform> DeviceCore<P> {
    /// A new display on `port` (it validates, reserves the serial number and attaches itself):
    /// the host's first reference, or the reason for a refusal: a port out of range or taken, a
    /// serial number another display of the process has, or pipelines the platform refuses. It
    /// runs in the host's call with no handler live, so a raise from it ends the process at the
    /// raise.
    pub fn new_display(
        self: &Arc<Self>,
        descriptor: &DisplayDescriptor,
        port: u64,
        serial_num: u32,
    ) -> Result<DisplayHandle<P>, DisplayRefusal> {
        // The display retains the device.
        match Display::new(&self.platform, self.clone(), descriptor, port, serial_num) {
            Ok(display) => display.map(DisplayHandle::new),
            Err(raised) => raised.fatal(),
        }
    }

    /// The display ports the host gave the device.
    pub fn display_port_count(&self) -> u32 {
        self.display_port_count.load(Ordering::Relaxed)
    }

    /// The Metal device, borrowed: the displays send to it without retaining it, and retain it
    /// (`clone_ref`) only where they keep it (the pipeline cache).
    pub fn mtl_device_ref(&self) -> &P::Device {
        &self.mtl_device
    }

    /// Whether the displays store the guest's protection options: always, whatever the
    /// descriptor's `enableProtectedContent` (the module doc of `device`).
    pub fn enable_protected_content(&self) -> bool {
        self.caps.protected_content
    }

    /// The host's `addTraceRange` block over the EFI framebuffer, or no range without one.
    pub fn add_trace_range(&self, range: PhysRange, handler: TraceRangeHandler) -> HostTraceRange {
        match &self.blocks.add_trace_range {
            Some(add) => add(range, handler),
            None => HostTraceRange::NULL,
        }
    }

    /// The host's `removeTraceRange` block, when it has one.
    pub fn remove_trace_range(&self, range: HostTraceRange) {
        if let Some(remove) = &self.blocks.remove_trace_range {
            remove(range);
        }
    }

    /// Claims `port` for `display`: hands it the port's guest display if there is one, else parks
    /// it (unretained) until the guest gives the port a page. `false` for a port out of range or
    /// already claimed. The EFI display is not told.
    pub fn attach_display(&self, display: Weak<Display<P>>, port: u64) -> bool {
        if port >= u64::from(self.display_port_count.load(Ordering::Relaxed)) {
            return false;
        }
        let mut d = self.displays.lock();
        let attached = d.display_port_busy_mask & port_bit(port) == 0;
        if attached {
            d.display_port_busy_mask |= port_bit(port);
            if let Some(guest_display) = guest_display_at(&d, port) {
                // Nothing catches: a raise leaves through the lock to a caller with none.
                if let Err(raised) = guest_display.set_client(OnFault::Die, Some(display.clone())) {
                    drop(d);
                    raised.fatal();
                }
                // +0: the core, never an upgrade (the initializer holds the display, so this is
                // never the last release).
                if let Some(core) = self.display_core(&display)
                    && let Err(raised) =
                        core.set_guest_display(OnFault::Die, Some(Arc::downgrade(&guest_display)))
                {
                    drop(d);
                    raised.fatal();
                }
            } else {
                crate::fatal_unless!(client_at(&d, port).is_none(), DeathKind::ParkedOnFreePort);
                set_client_at(&mut d, port, Some(display));
            }
        }
        attached
    }

    /// Gives `port` up: the guest display forgets its client, or the parked client is dropped; port
    /// 0 also unbinds the EFI display. `display` may be mid-drop (its `Weak` no longer upgrades):
    /// the display's end calls this.
    // With a guest display, the display's own `guest_display` is left set.
    pub fn detach_display(&self, display: &Weak<Display<P>>, port: u64) {
        let mut d = self.displays.lock();
        crate::fatal_unless!(
            port < u64::from(self.display_port_count.load(Ordering::Relaxed)),
            DeathKind::DetachOutOfRange
        );
        crate::fatal_unless!(
            d.display_port_busy_mask & port_bit(port) != 0,
            DeathKind::DetachNotClaimed
        );
        if let Some(guest_display) = guest_display_at(&d, port) {
            // The guest display's client read without its mutex (a leaf lock): the device never
            // waits on a guest display's state machine here.
            crate::fatal_unless!(
                same_client(guest_display.client().as_ref(), Some(display)),
                DeathKind::DetachOtherGuestDisplayClient
            );
            // Nothing catches: a raise leaves through the lock to the display's end.
            if let Err(raised) = guest_display.set_client(OnFault::Die, None) {
                drop(d);
                raised.fatal();
            }
        } else {
            let parked = client_at(&d, port);
            crate::fatal_unless!(
                parked.is_none() || same_client(parked.as_ref(), Some(display)),
                DeathKind::DetachOtherParkedClient
            );
            // The ending display's guest display, read unretained (its core stays reachable until
            // its end has detached it).
            let client_guest_display_nil = self
                .display_core(display)
                .is_none_or(|core| core.guest_display.lock().is_none());
            crate::fatal_unless!(
                client_guest_display_nil,
                DeathKind::DetachParkedWithGuestDisplay
            );
            set_client_at(&mut d, port, None);
        }
        if port == 0
            && let Some(efi) = self.displays.atomics().efi_display()
        {
            efi.set_display(None);
        }
        d.display_port_busy_mask &= !port_bit(port);
    }

    /// A non-zero `page` makes the port's guest display over that guest page (and hands it a parked
    /// client); page 0 releases it and parks its client again. Raises for a port out of range,
    /// a port already (or not) in use, and a page with no root node.
    // Without a root node the new guest display (and its page mapping) is kept for good.
    // Page 0 clears the root word only while running.
    pub fn set_shared_display_state_page(
        &self,
        page: u32,
        port: u64,
        restoring: bool,
    ) -> RaiseResult<()> {
        let count = self.display_port_count.load(Ordering::Relaxed);
        if port >= u64::from(count) {
            return Err(out_of_range(port, count));
        }
        let mut d = self.displays.lock();
        // A raise unwinds through the guard.
        let set = self.set_shared_display_state_page_locked(&mut d, page, port, restoring);
        drop(d);
        set
    }

    /// The page change, under the displays lock.
    fn set_shared_display_state_page_locked(
        &self,
        d: &mut StagedGuard<'_, DisplaysLocked<P>>,
        page: u32,
        port: u64,
        restoring: bool,
    ) -> RaiseResult<()> {
        let guest_display = guest_display_at(d, port);
        let guest_display = if page != 0 {
            if guest_display.is_some() {
                crate::raise!(
                    RaiseKind::PortInUse,
                    format_args!("display port {port} is in use")
                );
            }
            // Maps the page, or raises itself.
            let guest_display =
                GuestDisplay::new(self.platform.clone(), self.weak_self.clone(), port, page)?;
            if restoring {
                guest_display.will_restore();
            } else {
                let root = self.root_node.lock();
                let Some(root_node) = root.root_node else {
                    // The new guest display is kept for good on the raise: its page's unmap
                    // never comes.
                    keep_forever(guest_display, Kept::UnmapNeverComes);
                    crate::raise!(
                        RaiseKind::NoRootNode,
                        format_args!("display port {port}'s state page set before the root node")
                    );
                };
                root_node.write_u32(display_word_offset(port), page, DeathKind::RootWordNull);
            }
            if let Some(client) = client_at(d, port) {
                // A raise keeps the new guest display for good, as the root node's does. The core,
                // +0, never an upgrade, whose drop could run the display's end here under the
                // displays lock.
                let handed = match self.display_core(&client) {
                    Some(core) => core
                        .set_guest_display(OnFault::Return, Some(Arc::downgrade(&guest_display))),
                    None => Ok(()),
                }
                .and_then(|()| guest_display.set_client(OnFault::Return, Some(client)));
                if let Err(raised) = handed {
                    keep_forever(guest_display, Kept::UnmapNeverComes);
                    return Err(raised);
                }
                set_client_at(d, port, None);
            }
            Some(guest_display)
        } else {
            let Some(guest_display) = guest_display else {
                crate::raise!(
                    RaiseKind::PortNotInUse,
                    format_args!("display port {port} is not in use")
                );
            };
            let client = guest_display.client();
            guest_display.set_client(OnFault::Return, None)?;
            // The client's core (none for no client), +0, never an upgrade, whose drop could run
            // the display's end here under the displays lock.
            if let Some(core) = client.as_ref().and_then(|c| self.display_core(c)) {
                core.set_guest_display(OnFault::Return, None)?;
            }
            crate::fatal_unless!(
                client_at(d, port).is_none(),
                DeathKind::ParkedBesideGuestDisplay
            );
            set_client_at(d, port, client);
            // This reference, then the slot's: the guest display's end runs here. The unlocked view
            // keeps the slot's until the store below publishes it.
            drop(guest_display);
            if let Some(slot) = d.guest_display.get_mut(port.to_usize()) {
                drop(slot.take());
            }
            {
                let root = self.root_node.lock();
                if let Some(root_node) = root.root_node
                    && self.run_state.load() == RunState::Running
                {
                    root_node.write_u32(display_word_offset(port), 0, DeathKind::RootWordNull);
                }
            }
            None
        };
        // Seen at once by the unlocked readers (the teardown, the pause, ...).
        if let Some(slot) = d.guest_display.get_mut(port.to_usize()) {
            *slot = guest_display;
        }
        d.publish();
        Ok(())
    }

    /// The port's guest display; raises for a port out of range or with none.
    pub fn get_guest_display_at_port(&self, port: u64) -> RaiseResult<Arc<GuestDisplay<P>>> {
        let count = self.display_port_count.load(Ordering::Relaxed);
        if port >= u64::from(count) {
            return Err(out_of_range(port, count));
        }
        let d = self.displays.lock();
        let Some(guest_display) = guest_display_at(&d, port) else {
            crate::raise!(
                RaiseKind::PortNotInUse,
                format_args!("display port {port} is not in use")
            );
        };
        Ok(guest_display)
    }

    /// The port's parked client, else its guest display's client; `None` for a port out of range.
    pub fn get_display_at_port(&self, port: u64) -> Option<Arc<Display<P>>> {
        if port >= u64::from(self.display_port_count.load(Ordering::Relaxed)) {
            return None;
        }
        let d = self.displays.lock();
        let display =
            client_at(&d, port).or_else(|| guest_display_at(&d, port).and_then(|n| n.client()));
        display.and_then(|d| d.upgrade())
    }

    /// Page 0 for every port that has a guest display, each catching and ending its raise in the
    /// hold's holder. Called by the root's teardown.
    ///
    /// It runs in the teardown, whose callers have no handler: the host's MMIO write, a resume,
    /// a stop or reset, and the device's shutdown. Each page 0 runs under a handler of its own.
    pub fn teardown_displays(&self, hold: Hold<'_>) {
        let count = self.display_port_count.load(Ordering::Relaxed);
        for port in 0..u64::from(count) {
            // The port's guest display, read without the lock (it never waits behind a page 0 or a
            // detach that holds it).
            if self.displays.atomics().has_guest_display(port) {
                // With a guest display on the port, page 0 raises only if the slot empties between
                // this check and the call's lock (another thread's page 0 on the port): the client
                // set it runs finds no client for the state machine to update, and the display's
                // `set_guest_display(None)` sets no mode list, so neither raises.
                let dropped = self.set_shared_display_state_page(0, port, false);
                if let Err(raised) = dropped {
                    hold.land(raised);
                }
            }
        }
    }

    /// Non-zero: makes the EFI display if there is none (bound to port 0's parked client) and
    /// stores `enabled`. Zero: tears it down and zeroes every EFI register.
    // Only the parked port-0 client is bound; any non-zero value is stored.
    pub fn set_efi_display(&self, enabled: u32) {
        // The EFI display, port 0's parked client and the EFI registers: every access here is
        // without the displays lock, so it never waits behind a page 0 or a detach holding it.
        let views = self.displays.atomics();
        if enabled != 0 {
            if !views.has_efi_display() {
                let efi = EfiDisplay::new(&self.platform, self.weak_self.clone());
                // Stored with no lock and no release of what the slot held (none, unless another
                // thread stored one meanwhile: that one is kept for good, and its timer keeps
                // firing).
                if let Some(replaced) = views.replace_efi_display(Some(efi)) {
                    keep_forever(replaced, Kept::TimerKeepsFiring);
                }
                // The receiver, then the argument, each read now without the lock.
                let receiver = views.efi_display();
                let client = views.display_client(0);
                if let Some(efi) = receiver {
                    // The EFI display holds the display weakly: forming a weak reference to one
                    // whose release has begun (its count at 0, its end not yet at the detach that
                    // empties the slot) is a death. The EFI display is made and stored first;
                    // `enabled` is never written.
                    if client.as_ref().is_some_and(|w| w.strong_count() == 0) {
                        crate::fatal!(DeathKind::EfiBindDeallocating);
                    }
                    efi.set_display(client);
                }
            }
            views.enabled.store(enabled, Ordering::Relaxed);
        } else {
            // The teardown, the slot read without the lock.
            let efi = views.efi_display();
            if let Some(efi) = &efi {
                efi.teardown();
            }
            // The slot's reference goes (the teardown's own release).
            let old = views.replace_efi_display(None);
            drop(old);
            drop(efi);
            // The registers zeroed, no lock.
            for cell in [
                &views.enabled,
                &views.start,
                &views.length,
                &views.depth,
                &views.mode,
                &views.stride,
            ] {
                cell.store(0, Ordering::Relaxed);
            }
        }
    }

    /// The process's IOSurface host device, looked up while the cached pointer is empty and the
    /// device uses the mapper, under the device's `@synchronized` lock. The cache is unretained:
    /// once filled it is never looked up again, and a dead one answers `None`.
    ///
    /// The answer is the host device's core ([`IoSurfaceHostDevice::core_weak`], upgraded),
    /// never a reference to the host's device: a caller can neither keep the host's object alive
    /// nor run its end on the caller's thread, and while that end waits in its `dispatch_sync`
    /// the core still answers.
    pub fn mapper_service(&self) -> Option<Arc<HostDeviceCore<P>>> {
        let mut service = None;
        self.synchronized(&mut || {
            // The cached pointer, a leaf read.
            let cached = self.mapper_service.lock().clone();
            let cached = match cached {
                Some(cached) => Some(cached),
                None => {
                    if !self.using_mapper_service {
                        service = None;
                        return;
                    }
                    // The lookup's temporaries are gone before the store (under the lock).
                    let found = IoSurfaceHostDevice::latest_core(self.platform.as_ref());
                    *self.mapper_service.lock() = found.clone();
                    found
                }
            };
            service = cached.as_ref().and_then(Weak::upgrade);
        });
        service
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::EfiRegisters;
    use crate::device::test_support::{Opts, TestDevice, device};
    use crate::raise::death::DeathSite;
    use crate::testkit::FakePlatform;
    use crate::testkit::capture::catch_death;

    fn dev() -> TestDevice {
        device(Opts::default())
    }

    /// `(detail, kind)` of a display ports raise.
    fn thrown<T: std::fmt::Debug>(r: RaiseResult<T>) -> (String, RaiseKind) {
        let e = r.expect_err("expected a raise");
        match e.id() {
            RaiseId::Displays(k) => (e.detail().unwrap_or_default().to_owned(), k),
            other => panic!("not a display ports raise: {other}"),
        }
    }

    fn died(r: Result<impl std::fmt::Debug, DeathSite>, kind: DeathKind) {
        assert_eq!(r.expect_err("expected a death"), kind.site());
    }

    /// A live display whose own host is another device (one port), not the device under test:
    /// that device sees it only through the `Weak`s a test hands it, and reaches its core as the
    /// display's own device does, until the display's end begins. The other device's handle comes
    /// back beside it: its drop shuts that device down.
    fn real_display(
        t: &TestDevice,
    ) -> (
        Arc<Display<FakePlatform>>,
        Arc<crate::device::Handle<FakePlatform>>,
    ) {
        use crate::testkit::host::{FakeHostQueue, QueueMode};
        let mut c = crate::device::config::DeviceConfig::new(Some(t.platform.device()));
        c.display_port_count = 1;
        c.blocks = t.host.blocks();
        let other = DeviceCore::create(t.platform.clone(), c).unwrap();
        let host: Arc<DeviceCore<FakePlatform>> = Arc::clone(&other);
        let mut desc = DisplayDescriptor::sdk_defaults();
        desc.queue = Some(FakeHostQueue::new(QueueMode::Manual));
        desc.new_frame_event_handler = Some(Arc::new(|| ()));
        let display = Display::new(&t.platform, host, &desc, 0, 0x1234)
            .unwrap()
            .unwrap();
        t.device.register_display(&display);
        let (dev, core) = (Arc::downgrade(&t.device), Arc::downgrade(display.core()));
        display.core().drop_probe.lock().replace(Box::new(move || {
            if let (Some(dev), Some(core)) = (dev.upgrade(), core.upgrade()) {
                dev.deregister_display(&core);
            }
        }));
        (display, other)
    }

    /// An unretained client with no object behind it.
    fn client() -> Weak<Display<FakePlatform>> {
        Weak::new()
    }

    mod attach_display {
        use super::*;

        /// a port out of range is refused
        #[test]
        fn port_out_of_range() {
            let t = dev();
            assert!(!t.device.attach_display(client(), 2));
            assert_eq!(t.device.displays.lock().display_port_busy_mask, 0);
        }

        /// no guest display: the client is parked and the port marked busy; a second attach is
        /// refused
        #[test]
        fn parks_then_busy() {
            let t = dev();
            assert!(t.device.attach_display(client(), 1));
            {
                let d = t.device.displays.lock();
                assert_eq!(d.display_port_busy_mask, 0b10);
                assert!(d.display_client[1].is_some());
            }
            assert!(!t.device.attach_display(client(), 1));
        }

        /// a parked client on a free port dies
        #[test]
        fn parked_client_on_free_port_dies() {
            let t = dev();
            t.device.displays.lock().display_client[0] = Some(client());
            died(
                catch_death(|| t.device.attach_display(client(), 0)),
                DeathKind::ParkedOnFreePort,
            );
        }

        /// the port bit is an int shift, sign-extended
        #[test]
        fn port_bits() {
            assert_eq!(port_bit(0), 1);
            assert_eq!(port_bit(7), 0x80);
            assert_eq!(port_bit(31), 0xffff_ffff_8000_0000);
            assert_eq!(port_bit(32), 1);
        }
    }

    mod detach_display {
        use super::*;

        /// a port out of range dies
        #[test]
        fn port_out_of_range_dies() {
            let t = dev();
            died(
                catch_death(|| t.device.detach_display(&client(), 2)),
                DeathKind::DetachOutOfRange,
            );
        }

        /// a port not claimed dies
        #[test]
        fn port_not_busy_dies() {
            let t = dev();
            died(
                catch_death(|| t.device.detach_display(&client(), 1)),
                DeathKind::DetachNotClaimed,
            );
        }

        /// no guest display: the parked client is dropped and the port freed
        #[test]
        fn unparks() {
            let t = dev();
            let c = client();
            assert!(t.device.attach_display(c.clone(), 1));
            t.device.detach_display(&c, 1);
            let d = t.device.displays.lock();
            assert_eq!(d.display_port_busy_mask, 0);
            assert!(d.display_client[1].is_none());
        }

        /// a parked client other than the one detaching dies
        #[test]
        fn parked_other_client_dies() {
            let t = dev();
            let (d, _other) = real_display(&t);
            assert!(t.device.attach_display(client(), 1));
            died(
                catch_death(|| t.device.detach_display(&Arc::downgrade(&d), 1)),
                DeathKind::DetachOtherParkedClient,
            );
        }

        /// a parked client that holds a guest display (another port's) dies
        #[test]
        fn parked_client_with_guest_display_dies() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            let (d, _other) = real_display(&t);
            d.set_guest_display(OnFault::Die, Some(Arc::downgrade(&guest_display)))
                .unwrap();
            assert!(t.device.attach_display(Arc::downgrade(&d), 1));
            died(
                catch_death(|| t.device.detach_display(&Arc::downgrade(&d), 1)),
                DeathKind::DetachParkedWithGuestDisplay,
            );
            // The display's own host would die detaching a display that holds a guest display.
            d.set_guest_display(OnFault::Die, None).unwrap();
        }

        /// a claimed port with no parked client detaches too (`!_displayClient[port]`)
        #[test]
        fn claimed_without_client() {
            let t = dev();
            t.device.displays.lock().display_port_busy_mask = 1;
            t.device.detach_display(&client(), 0);
            assert_eq!(t.device.displays.lock().display_port_busy_mask, 0);
        }
    }

    mod shared_state_page {
        use super::*;

        /// the guest display's release raising at its shared-state mapping's release (the host's
        /// `unmapMemory` block) ends the process there, with the root word as it was
        #[test]
        fn a_raising_unmap_in_the_guest_displays_release_dies() {
            let t = dev().with_root_page();
            let word = crate::device::displays::ROOT_DISPLAY_WORDS + 1;
            {
                {
                    t.device
                        .set_shared_display_state_page(0x10, 1, false)
                        .unwrap();
                }
            }
            t.platform.script().push(
                "unmapMemory",
                crate::testkit::Step::raise("HostRefused", "no"),
            );
            let d = catch_death(|| {
                let _ = t.device.set_shared_display_state_page(0, 1, false);
            })
            .unwrap_err();
            assert_eq!(
                (d.kind, d.file),
                (crate::raise::death::DeathKind::HostRaised, "unmapMemory")
            );
            assert_eq!(t.guest_root_word(word), 0x10);
        }

        /// a port out of range throws with the last valid port
        #[test]
        fn port_out_of_range_throws() {
            let t = dev();
            assert_eq!(
                thrown(t.device.set_shared_display_state_page(5, 2, false)),
                (
                    "display port 2 is past the last port, 1".into(),
                    RaiseKind::PortOutOfRange
                )
            );
        }

        /// page 0 on a port with no guest display throws that it is not in use
        #[test]
        fn page_zero_without_guest_display_throws() {
            let t = dev();
            assert_eq!(
                thrown(t.device.set_shared_display_state_page(0, 1, false)),
                (
                    "display port 1 is not in use".into(),
                    RaiseKind::PortNotInUse
                )
            );
        }

        /// a page without a root node throws and keeps the guest display for good
        #[test]
        fn page_without_root_throws() {
            let t = dev();
            // the guest display's page is mapped through the root task: make one, then drop the
            // node
            t.device
                .set_root_page(Hold::NONE, crate::device::test_support::ROOT_PAGE);
            t.device.root_node.lock().root_node = None;
            assert_eq!(
                thrown(t.device.set_shared_display_state_page(0x10, 0, false)),
                (
                    "display port 0's state page set before the root node".into(),
                    RaiseKind::NoRootNode
                )
            );
            assert!(t.device.displays.lock().guest_display[0].is_none());
        }

        /// a page makes the guest display and writes the root word; the parked client moves to it;
        /// page 0 re-parks it and clears the word in run state 0 only
        #[test]
        fn page_then_zero() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 1, false)
                .unwrap();
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS + 1), 0x10);
            assert_eq!(
                thrown(t.device.set_shared_display_state_page(0x11, 1, false)),
                ("display port 1 is in use".into(), RaiseKind::PortInUse)
            );
            t.device.run_state.store(RunState::Paused);
            t.device.set_shared_display_state_page(0, 1, false).unwrap();
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS + 1), 0x10);
            assert!(t.device.displays.lock().guest_display[1].is_none());
            // in run state 0 page 0 clears the word
            t.device.run_state.store(RunState::Running);
            t.device
                .set_shared_display_state_page(0x12, 1, false)
                .unwrap();
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS + 1), 0x12);
            t.device.set_shared_display_state_page(0, 1, false).unwrap();
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS + 1), 0);
        }
    }

    mod shared_state_page_clients {
        use super::*;

        /// restoring: no root word; a parked client moves to the guest display and back on page 0
        #[test]
        fn restoring_and_parked_client() {
            let t = dev().with_root_page();
            assert!(t.device.attach_display(client(), 1));
            t.device
                .set_shared_display_state_page(0x10, 1, true)
                .unwrap();
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS + 1), 0);
            {
                let d = t.device.displays.lock();
                assert!(d.display_client[1].is_none());
                let guest_display = d.guest_display[1].clone().unwrap();
                assert!(guest_display.client().is_some());
                assert!(guest_display.guest_display_mutex.lock().will_restore);
            }
            t.device.set_shared_display_state_page(0, 1, false).unwrap();
            let d = t.device.displays.lock();
            assert!(d.display_client[1].is_some());
            assert!(d.guest_display[1].is_none());
        }

        /// page 0 while a client is parked on the port dies
        #[test]
        fn page_zero_with_parked_client_dies() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            t.device.displays.lock().display_client[0] = Some(client());
            died(
                catch_death(|| t.device.set_shared_display_state_page(0, 0, false)),
                DeathKind::ParkedBesideGuestDisplay,
            );
            // the process is dead after the first; here the device's teardown meets the same slot
            died(
                catch_death(move || drop(t)),
                DeathKind::ParkedBesideGuestDisplay,
            );
        }
    }

    mod attach_detach_with_guest_display {
        use super::*;

        /// with a guest display the client goes to it, and detaching takes it off again
        #[test]
        fn guest_display_takes_client() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let c = client();
            assert!(t.device.attach_display(c.clone(), 0));
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            assert!(guest_display.client().is_some());
            assert!(t.device.displays.lock().display_client[0].is_none());
            t.device.detach_display(&c, 0);
            assert!(guest_display.client().is_none());
            assert_eq!(t.device.displays.lock().display_port_busy_mask, 0);
        }

        /// detaching through the guest display leaves the display's own `guest_display` set
        #[test]
        fn detach_leaves_guest_display() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            let (d, _other) = real_display(&t);
            let w = Arc::downgrade(&d);
            assert!(t.device.attach_display(w.clone(), 0));
            assert!(d.guest_display().is_some());
            t.device.detach_display(&w, 0);
            let guest_display = t.device.get_guest_display_at_port(0).unwrap();
            assert!(guest_display.client().is_none());
            assert!(d.guest_display().is_some());
            // The display's own host would die detaching a display that holds a guest display.
            d.set_guest_display(OnFault::Die, None).unwrap();
        }

        /// detaching a client the guest display does not hold dies
        #[test]
        fn guest_display_other_client_dies() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            t.device.displays.lock().display_port_busy_mask = 1;
            died(
                catch_death(|| t.device.detach_display(&client(), 0)),
                DeathKind::DetachOtherGuestDisplayClient,
            );
        }
    }

    mod teardown_displays_guest_displays {
        use super::*;

        /// every guest display released; the root words kept outside run state 0
        #[test]
        fn releases_guest_displays() {
            let t = dev().with_root_page();
            t.device
                .set_shared_display_state_page(0x10, 0, false)
                .unwrap();
            t.device
                .set_shared_display_state_page(0x11, 1, false)
                .unwrap();
            t.device.run_state.store(RunState::Paused);
            t.device.teardown_displays(Hold::NONE);
            assert!(
                t.device
                    .displays
                    .lock()
                    .guest_display
                    .iter()
                    .all(Option::is_none)
            );
            assert_eq!(t.guest_root_word(ROOT_DISPLAY_WORDS), 0x10);
        }
    }

    mod guest_display_at_port {
        use super::*;

        /// out of range and not in use throw
        #[test]
        fn throws() {
            let t = dev();
            assert_eq!(
                thrown(t.device.get_guest_display_at_port(9).map(|_| ())),
                (
                    "display port 9 is past the last port, 1".into(),
                    RaiseKind::PortOutOfRange
                )
            );
            assert_eq!(
                thrown(t.device.get_guest_display_at_port(0).map(|_| ())),
                (
                    "display port 0 is not in use".into(),
                    RaiseKind::PortNotInUse
                )
            );
        }
    }

    mod display_at_port {
        use super::*;

        /// nothing out of range, nothing on an empty port, nothing for a dead client
        #[test]
        fn none() {
            let t = dev();
            assert!(t.device.get_display_at_port(5).is_none());
            assert!(t.device.get_display_at_port(0).is_none());
            assert!(t.device.attach_display(client(), 0));
            assert!(t.device.get_display_at_port(0).is_none());
        }
    }

    mod teardown_displays {
        use super::*;

        /// no guest displays: nothing happens
        #[test]
        fn no_guest_displays() {
            let t = dev();
            t.device.teardown_displays(Hold::NONE);
            assert!(
                t.device
                    .displays
                    .lock()
                    .guest_display
                    .iter()
                    .all(Option::is_none)
            );
        }
    }

    mod set_efi_display {
        use super::*;

        /// 0 zeroes every EFI register (with no EFI display to tear down)
        #[test]
        fn off_zeroes_registers() {
            let t = dev();
            t.device.displays.lock().efi = EfiRegisters {
                enabled: 1,
                start: 2,
                length: 3,
                depth: 4,
                mode: 5,
                stride: 6,
            };
            t.device.set_efi_display(0);
            let d = t.device.displays.lock();
            assert_eq!(d.efi, EfiRegisters::default());
            assert!(d.efi_display.is_none());
        }

        /// any non-zero value makes the EFI display and is stored
        #[test]
        fn on_stores_value() {
            let t = dev();
            t.device.set_efi_display(7);
            assert_eq!(t.device.displays.lock().efi.enabled, 7);
            assert!(t.device.displays.lock().efi_display.is_some());
            t.device.set_efi_display(0);
            assert!(t.device.displays.lock().efi_display.is_none());
        }

        /// A display parked on port 0 whose end has not detached it yet: the device's
        /// `display_client[0]` still names it. Its own host is a recorder, so its detach never
        /// reaches the device and the slot stays as it is in that window.
        fn parked_on_port_0(t: &TestDevice) -> Arc<Display<FakePlatform>> {
            let (d, _other) = real_display(t);
            assert!(t.device.attach_display(Arc::downgrade(&d), 0));
            d
        }

        /// With — a live parked port-0 display is bound (`objc_storeWeak` of a live object): the
        /// EFI mode registers read its modes.
        #[test]
        fn binds_the_parked_port_0_display() {
            use crate::display::DisplayCoord;
            use crate::display::mode::DisplayMode;
            let t = dev();
            let d = parked_on_port_0(&t);
            d.set_mode_list(&[DisplayMode {
                size_in_pixels: DisplayCoord { x: 1024, y: 768 },
                refresh_rate_in_hz: 60.0,
                supports_edr: false,
                supports_10bpc: false,
            }])
            .unwrap();
            t.device.set_efi_display(1);
            assert_eq!(t.device.displays.lock().efi.enabled, 1);
            assert_eq!(t.device.get_efi_mode_count(), 1);
            drop(d);
            // A dead display loads as nil.
            assert_eq!(t.device.get_efi_mode_count(), 0);
        }

        /// The host's last release of the parked port-0 display has begun and its end has not yet
        /// reached the detach when the guest enables EFI: `set_efi_display` reads
        /// `display_client[0]` and binds the EFI display to a display whose teardown has begun,
        /// which ends the process. The EFI display was made and stored first; `efi.enabled` is
        /// never written.
        #[test]
        fn binding_a_parked_port_0_display_in_teardown_dies() {
            let t = dev();
            let d = parked_on_port_0(&t);
            let weak = Arc::downgrade(&d);
            drop(d);
            assert_eq!(weak.strong_count(), 0);
            assert!(t.device.displays.lock().display_client[0].is_some());
            assert_eq!(
                catch_death(|| t.device.set_efi_display(1)),
                Err(DeathKind::EfiBindDeallocating.site())
            );
            assert!(t.device.displays.lock().efi_display.is_some());
            assert_eq!(
                t.device.displays.atomics().enabled.load(Ordering::Relaxed),
                0
            );
        }

        /// No parked client (the port is free, or has a guest display): `nil` is stored, which
        /// never aborts.
        #[test]
        fn no_parked_client_stores_nil() {
            let t = dev();
            t.device.set_efi_display(1);
            assert_eq!(t.device.displays.lock().efi.enabled, 1);
            assert_eq!(t.device.get_efi_mode_count(), 0);
        }
    }

    mod metal_device_ref {
        use super::*;
        use crate::platform::Handle;

        /// The displays' sends to the device's Metal device borrow the device's field: its own
        /// `MTLDevice`, with no retain taken for the send and none released after it.
        #[test]
        fn the_displays_borrow_the_ivar_unretained() {
            let t = device(Opts::default());
            let mtl = t.platform.device();
            let before = mtl.retain_count();
            let lent = t.device.mtl_device_ref();
            assert_eq!(lent.id(), mtl.id());
            assert_eq!(mtl.retain_count(), before);
            // A display retains it only where it keeps it (the pipeline cache).
            let retained = lent.clone_ref();
            assert_eq!(mtl.retain_count(), before + 1);
            drop(retained);
            assert_eq!(mtl.retain_count(), before);
        }
    }

    mod mapper_service {
        use super::*;

        /// without the mapper flag there is no lookup and no service
        #[test]
        fn not_using_mapper() {
            let t = device(Opts {
                using_mapper_service: false,
                ..Opts::default()
            });
            assert!(t.device.mapper_service().is_none());
            assert!(t.device.mapper_service.lock().is_none());
        }

        /// The device's mapper service is an unretained pointer, and
        /// `retain_surface_with_mapping_id` through it holds no reference. A caller parked in that
        /// call's sync on the host device's hardware queue does not keep the host's object alive:
        /// the host's last release ends it at once on the releasing thread (its teardown line
        /// logged while the caller still waits), and the caller's return ends nothing of its own. A
        /// `Weak` cache would hold the host's object for the call, so both teardown lines would
        /// come from the caller after its call returned.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: two host threads, a real timeout
        fn the_hosts_last_release_ends_it_on_its_own_thread() {
            use crate::iosurface::descriptor::IoSurfaceHostDeviceDescriptor;
            use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
            use std::sync::mpsc::channel;
            let t = device(Opts {
                using_mapper_service: true,
                ..Opts::default()
            });
            let mut map = MemoryMapDescriptor::new();
            for r in t.host.ram().ranges() {
                map.add_range(GuestPhysicalRange {
                    physical_address: r.physical_address,
                    physical_length: r.region.len(),
                    host_address: r.host(),
                });
            }
            let mut d = IoSurfaceHostDeviceDescriptor::new();
            d.memory_map_descriptor = Some(map);
            let logs = crate::testkit::logs::capture();
            let host_dev = IoSurfaceHostDevice::new(&t.platform, &d).expect("host device");
            // The first call fills the unretained cache.
            assert!(t.device.mapper_service().is_some());
            // The host device's lines so far.
            let lines = crate::sync::Mutex::new(Vec::<String>::new());
            let logged = |pattern: &str| {
                let mut lines = lines.lock();
                lines.extend(
                    logs.take()
                        .into_iter()
                        .filter(|l| l.1.ends_with("::iosurface::device"))
                        .map(|l| l.2),
                );
                lines.iter().filter(|l| l.contains(pattern)).count()
            };
            let deallocs = || logged(" tearing down") + logged(" torn down");
            // A bounded poll on a logged effect: no hook shows a thread parked in a sync.
            let poll = |done: &dyn Fn() -> bool| {
                let start = std::time::Instant::now();
                while !done() && start.elapsed().as_secs() < 30 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            };
            // The source's cancel callout frees the handler, and what it holds, on the queue after
            // the teardown's block: kept to wait it out.
            let hw_queue = host_dev.hw_queue.clone();
            // Park `_hwQueue` behind a gate.
            let (gate_tx, gate_rx) = channel::<()>();
            let (parked_tx, parked_rx) = channel::<()>();
            host_dev.hw_queue.exec_async(move || {
                let _ = parked_tx.send(());
                let _ = gate_rx.recv();
            });
            parked_rx.recv().unwrap();
            // Thread A: `retain_surface_with_mapping_id` through the mapper service, which waits in
            // the sync behind the gate.
            let dev = t.device.clone();
            let a = std::thread::spawn(move || {
                dev.mapper_service()
                    .and_then(|m| m.retain_surface_with_mapping_id(1))
                    .is_none()
            });
            // A logs "retain surface 1" with the answer in hand, then enters the sync: A holds its
            // reference before B's release, and cannot drop it while the gate is shut.
            poll(&|| logged("retain surface 1") != 0);
            assert_eq!(logged("retain surface 1"), 1, "A's send began");
            // Thread B: the host's last release.
            let b = std::thread::spawn(move || drop(host_dev));
            poll(&|| deallocs() != 0);
            // B's teardown began while A still waits (the gate is shut: A can log nothing).
            assert_eq!(deallocs(), 1, "the teardown began on the releaser");
            assert!(!b.is_finished(), "B's teardown waits in its sync");
            gate_tx.send(()).unwrap();
            assert!(a.join().unwrap(), "no surface under mapping 1");
            b.join().unwrap();
            // Exactly two lines, both B's: A's return ended nothing.
            assert_eq!(deallocs(), 2);
            // Every block queued before this one has run: the cancel callout among them.
            hw_queue.sync(|| ());
            drop(hw_queue);
            // The cache still names the IOSurface device, now gone; a later call answers nil.
            assert!(t.device.mapper_service.lock().is_some());
            assert!(t.device.mapper_service().is_none());
        }
    }

    /// The port's client is unretained (`display_client[port]`, the guest display's client): opcode
    /// 1's and page 0's `set_guest_display` on it hold no reference under the displays lock, so the
    /// host's last release runs the display's end on the host's thread, where it waits in the
    /// detach for the displays lock.
    mod shared_state_page_unretained_client {
        use super::*;
        use crate::device::test_support::TestDevice;
        use crate::display::DisplayCoord;
        use crate::display::mode::DisplayMode;
        use crate::display::page as sp;
        use crate::host::HostQueue;
        use crate::testkit::host::{FakeHostQueue, QueueMode};
        use std::thread::ThreadId;
        use std::time::Duration;

        /// The guest page the tests give port 1's guest display.
        const PAGE: u32 = 0x10;

        /// A display of the device's own on port 1: with no guest display yet, it parks.
        fn parked_display(t: &TestDevice) -> DisplayHandle<FakePlatform> {
            let queue: Arc<dyn HostQueue> = FakeHostQueue::new(QueueMode::Manual);
            let mut desc = DisplayDescriptor::sdk_defaults();
            desc.queue = Some(queue);
            let d = t.device.new_display(&desc, 1, 0x1234).expect("display");
            assert!(t.device.displays.lock().display_client[1].is_some());
            d
        }

        /// Records the thread the display's end runs on.
        fn end_threads(d: &Display<FakePlatform>) -> Arc<crate::sync::Mutex<Vec<ThreadId>>> {
            let threads = Arc::new(crate::sync::Mutex::new(Vec::new()));
            let t = threads.clone();
            d.core().drop_probe.lock().replace(Box::new(move || {
                t.lock().push(std::thread::current().id());
            }));
            threads
        }

        /// Waits (10 s at most) until `f` holds.
        fn wait_until(what: &str, f: impl Fn() -> bool) {
            for _ in 0..10_000 {
                if f() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            panic!("timed out waiting for {what}");
        }

        /// The in-flight frame's completion.
        fn finish_frame(core: &DisplayCore<FakePlatform>) {
            let _g = core.display_mutex.lock();
            core.in_flight_frame_count.store(0, Ordering::SeqCst);
            core.in_flight_condition.notify_all();
        }

        /// A byte of the guest display's shared page in guest RAM.
        fn page_bytes<const N: usize>(t: &TestDevice, off: u64) -> [u8; N] {
            t.platform.fake_vm().sync();
            let mut b = [0u8; N];
            assert!(
                t.host.ram().read(
                    u64::from(PAGE)
                        .wrapping_mul(crate::guest::records::APV_PAGE_SIZE)
                        .wrapping_add(off),
                    &mut b
                )
            );
            b
        }

        fn page_u32(t: &TestDevice, off: u64) -> u32 {
            u32::from_le_bytes(page_bytes(t, off))
        }

        /// Opcode 1 for port 1, whose parked display has a frame in flight: `set_guest_display`
        /// waits for it under the display's mutex while the device holds its displays lock. The
        /// host's release then is the last one: the display's end runs on the host's thread and
        /// waits in the detach until opcode 1 is done, never on the FIFO's thread under the lock
        /// its detach takes.
        #[test]
        fn host_release_during_opcode_1s_set_guest_display_ends_it_on_the_host_thread() {
            let t = dev().with_root_page();
            let logs = crate::testkit::logs::capture();
            let destroyed = |logs: &crate::testkit::logs::Capture| {
                logs.take()
                    .iter()
                    .any(|l| l.2.starts_with("display on port 1 destroyed"))
            };
            let d = parked_display(&t);
            let threads = end_threads(&d);
            d.in_flight_frame_count.store(1, Ordering::SeqCst);
            let weak = d.downgrade();
            let core = Arc::downgrade(d.core());
            std::thread::scope(|sc| {
                let fifo = sc.spawn(|| t.device.set_shared_display_state_page(PAGE, 1, false));
                // Setting the display's guest display waits for the frame in flight.
                assert!(
                    t.platform
                        .virtual_clock()
                        .wait_for_waiters(1, Duration::from_secs(10))
                );
                let host = sc.spawn(move || {
                    drop(d);
                    std::thread::current().id()
                });
                wait_until("the host's release", || weak.strong_count() == 0);
                std::thread::sleep(Duration::from_millis(50));
                // The display's end waits in the detach; opcode 1 is not done.
                assert!(!host.is_finished());
                assert!(!destroyed(&logs));
                finish_frame(&core.upgrade().unwrap());
                fifo.join().unwrap().unwrap();
                let host = host.join().unwrap();
                // Every release point of the display's end ran on the host's thread.
                assert_eq!(*threads.lock(), vec![host]);
            });
            assert!(destroyed(&logs));
            let guest_display = t.device.get_guest_display_at_port(1).unwrap();
            assert!(guest_display.client().is_none());
            assert_eq!(t.device.displays.lock().display_port_busy_mask & 0b10, 0);
            assert!(t.device.display_core(&weak).is_none());
        }

        /// Page 0 for port 1 while its display has a frame in flight: the client's
        /// `set_guest_display(None)` waits for it, and the host's release meanwhile runs the
        /// display's end on the host's thread. Its detach, once page 0 has re-parked the client,
        /// reads the display's own guest display, none.
        #[test]
        fn host_release_during_page_0s_set_guest_display_ends_it_on_the_host_thread() {
            let t = dev().with_root_page();
            let logs = crate::testkit::logs::capture();
            let destroyed = |logs: &crate::testkit::logs::Capture| {
                logs.take()
                    .iter()
                    .any(|l| l.2.starts_with("display on port 1 destroyed"))
            };
            let d = parked_display(&t);
            {
                {
                    t.device
                        .set_shared_display_state_page(PAGE, 1, false)
                        .unwrap();
                }
            }
            assert!(d.guest_display().is_some());
            let threads = end_threads(&d);
            d.in_flight_frame_count.store(1, Ordering::SeqCst);
            let weak = d.downgrade();
            let core = Arc::downgrade(d.core());
            std::thread::scope(|sc| {
                let fifo = sc.spawn(|| t.device.set_shared_display_state_page(0, 1, false));
                assert!(
                    t.platform
                        .virtual_clock()
                        .wait_for_waiters(1, Duration::from_secs(10))
                );
                let host = sc.spawn(move || {
                    drop(d);
                    std::thread::current().id()
                });
                wait_until("the host's release", || weak.strong_count() == 0);
                std::thread::sleep(Duration::from_millis(50));
                assert!(!host.is_finished());
                finish_frame(&core.upgrade().unwrap());
                fifo.join().unwrap().unwrap();
                let host = host.join().unwrap();
                assert_eq!(*threads.lock(), vec![host]);
            });
            assert!(destroyed(&logs));
            let dl = t.device.displays.lock();
            assert!(dl.guest_display[1].is_none());
            assert!(dl.display_client[1].is_none());
            assert_eq!(dl.display_port_busy_mask & 0b10, 0);
        }

        /// A parked display with a mode list whose end waits in the detach is still intact, and
        /// opcode 1's unretained `set_guest_display` on the client hands the guest display its
        /// modes, then the guest display's `set_client` connects it (state 1, interrupt 4). The
        /// detach's `set_client(None)` then finds the connect unacked, and leaves it.
        #[test]
        fn a_parked_display_released_during_opcode_1_connects() {
            let t = dev().with_root_page();
            let logs = crate::testkit::logs::capture();
            t.platform.fake_vm().sync();
            assert!(t.host.ram().write_u32(
                u64::from(PAGE) * crate::guest::records::APV_PAGE_SIZE + sp::ENABLED_INTERRUPTS,
                4,
            ));
            t.platform.fake_vm().sync();
            let d = parked_display(&t);
            d.set_mode_list(&[DisplayMode {
                size_in_pixels: DisplayCoord { x: 1024, y: 768 },
                refresh_rate_in_hz: 60.0,
                supports_edr: false,
                supports_10bpc: false,
            }])
            .unwrap();
            let weak = d.downgrade();
            std::thread::scope(|sc| {
                // Opcode 1 stops at the root node's lock, holding `_displayMutex`, before it sets
                // the client's guest display.
                let root = t.device.root_node.lock();
                let fifo = sc.spawn(|| t.device.set_shared_display_state_page(PAGE, 1, false));
                wait_until("opcode 1's lock", || t.device.displays.try_lock().is_none());
                let host = sc.spawn(move || drop(d));
                wait_until("the host's release", || weak.strong_count() == 0);
                std::thread::sleep(Duration::from_millis(50));
                assert!(!host.is_finished());
                drop(root);
                fifo.join().unwrap().unwrap();
                host.join().unwrap();
            });
            assert_eq!(page_u32(&t, sp::SERIAL_NUM), 0x1234);
            assert_eq!(u16::from_le_bytes(page_bytes(&t, sp::MODE_LIST_COUNT)), 1);
            assert_eq!(page_u32(&t, sp::EXPECTED_ACK), 1);
            assert_eq!(page_bytes::<1>(&t, sp::ACK_PENDING), [1]);
            assert_eq!(page_u32(&t, sp::STATE), 1);
            assert_eq!(page_u32(&t, sp::RAISED_INTERRUPTS), 4);
            assert_ne!(t.device.interrupt_display.load(Ordering::SeqCst) & 0b10, 0);
            let lines: Vec<String> = logs.take().into_iter().map(|l| l.2).collect();
            assert!(
                lines.contains(
                    &"guest display on port 1: state 1 with an ack still pending, left as it is"
                        .to_owned()
                )
            );
            assert!(
                lines
                    .iter()
                    .any(|l| l.starts_with("display on port 1 destroyed"))
            );
            assert!(
                t.device
                    .get_guest_display_at_port(1)
                    .unwrap()
                    .client()
                    .is_none()
            );
        }
    }

    /// `set_efi_display` and `teardown_displays` read `guest_display[port]`, the EFI display and
    /// `display_client[0]` through their unlocked view, so neither waits behind a holder of the
    /// displays lock (page 0's 2 s `wait_for_pending_frames`, a detach behind a present), and each
    /// sees the slot as it is at that moment.
    mod unlocked_display_slots {
        use super::*;
        use std::sync::mpsc;
        use std::time::Duration;

        /// With `_displayMutex` held elsewhere, `-setEFIDisplay:` makes the EFI display, binds it
        /// to `_displayClient[0]` as it is now (nil: the page-0 holder has not re-parked its client
        /// yet) and returns at once. The holder's later store does not reach the EFI display: it
        /// stays bound to no display.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
        fn set_efi_display_never_waits_on_display_mutex() {
            let t = dev();
            let (d, _other) = real_display(&t);
            // The page-0 holder, inside the client's `set_guest_display(None)`.
            let mut held = t.device.displays.lock();
            let dev = t.device.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                dev.set_efi_display(1);
                let _ = tx.send(());
            });
            rx.recv_timeout(Duration::from_secs(3))
                .expect("-setEFIDisplay: waited on _displayMutex (it takes no lock)");
            // The holder re-parks its client and lets go.
            held.display_port_busy_mask |= 1;
            held.display_client[0] = Some(Arc::downgrade(&d));
            drop(held);
            let efi = t.device.displays.atomics().efi_display().expect("efi");
            assert!(efi.state.lock().display.is_none());
            assert_eq!(
                t.device.displays.atomics().enabled.load(Ordering::Relaxed),
                1
            );
            // The EFI mode count reads the same field.
            assert!(t.device.displays.lock().efi_display.is_some());
            // Off: torn down and cleared without the lock too.
            let held = t.device.displays.lock();
            let dev = t.device.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                dev.set_efi_display(0);
                let _ = tx.send(());
            });
            rx.recv_timeout(Duration::from_secs(3))
                .expect("-setEFIDisplay:0 waited on _displayMutex (it takes no lock)");
            drop(held);
            assert!(!t.device.displays.atomics().has_efi_display());
            assert!(t.device.displays.lock().efi_display.is_none());
        }

        /// A client parked under the lock is seen by `-setEFIDisplay:` from the store on
        /// (`_displayClient[0] = display`), even while the storing thread still holds
        /// `_displayMutex`.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
        fn set_efi_display_sees_a_parked_client_at_its_store() {
            let t = dev();
            let (d, _other) = real_display(&t);
            assert!(t.device.attach_display(Arc::downgrade(&d), 0));
            let held = t.device.displays.lock();
            let dev = t.device.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                dev.set_efi_display(1);
                let _ = tx.send(());
            });
            rx.recv_timeout(Duration::from_secs(3))
                .expect("-setEFIDisplay: waited on _displayMutex (it takes no lock)");
            drop(held);
            let efi = t.device.displays.atomics().efi_display().expect("efi");
            let bound = efi.state.lock().display.clone().expect("bound");
            assert!(Weak::ptr_eq(&bound, &Arc::downgrade(&d)));
        }

        /// No guest display: `-teardownDisplays` reads the slots and returns without the lock. With
        /// a guest display it reads the slot unlocked, then its page 0 waits for `_displayMutex`;
        /// when the holder emptied the slot meanwhile, page 0 raises that port 1 is not in use, and
        /// the teardown's handler swallows it.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time wait
        fn teardown_displays_reads_the_guest_display_slots_without_the_lock() {
            let t = dev().with_root_page();
            let held = t.device.displays.lock();
            let dev = t.device.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                dev.teardown_displays(Hold::NONE);
                let _ = tx.send(());
            });
            rx.recv_timeout(Duration::from_secs(3))
                .expect("-teardownDisplays waited on _displayMutex (it takes no lock)");
            drop(held);
            // A guest display on port 1.
            {
                {
                    t.device
                        .set_shared_display_state_page(0x10, 1, false)
                        .unwrap();
                }
            }
            let mut held = t.device.displays.lock();
            let dev = t.device.clone();
            let teardown = std::thread::spawn(move || dev.teardown_displays(Hold::NONE));
            // Its unlocked check has passed; its page 0 waits for the lock.
            std::thread::sleep(Duration::from_millis(200));
            assert!(!teardown.is_finished());
            // Another page 0 on the port empties the slot first.
            let guest_display = held.guest_display[1].take();
            drop(held);
            // Its page 0 raised that port 1 is not in use, which the teardown's handler swallowed.
            teardown.join().unwrap();
            drop(guest_display);
        }
    }

    /// The page change opens no pool of its own: what the host's callbacks autorelease in it (the
    /// guest display's page `unmapMemory` at page 0's release of the guest display) drains in the
    /// caller's pool, the FIFO command's or the API entry's, and a raise leaves nothing pushed.
    mod shared_state_page_pools {
        use super::*;
        use crate::device::root::rig::rig_with;
        use crate::rt::pool::test_native::{events, note, with_test_native};

        #[test]
        fn the_page_change_runs_in_the_callers_pool() {
            let armed = Arc::new(std::sync::atomic::AtomicBool::new(true));
            let arm = armed.clone();
            let r = rig_with(|b| {
                let inner = b.unmap_memory.clone().expect("unmap");
                b.unmap_memory = Some(Arc::new(move |t, off, len| {
                    if arm.load(Ordering::SeqCst) {
                        note("host unmap");
                    }
                    inner(t, off, len)
                }));
            });
            r.device.set_root_page(Hold::NONE, 1);
            let mut got = (Vec::new(), Vec::new());
            with_test_native(|| {
                r.device
                    .set_shared_display_state_page(0x10, 0, false)
                    .unwrap();
                got.0 = events();
                r.device.set_shared_display_state_page(0, 0, false).unwrap();
                got.1 = events();
                armed.store(false, Ordering::SeqCst);
            });
            let (page, zero) = got;
            assert!(page.is_empty(), "{page:?}");
            // Page 0: the guest display's page is unmapped at its release, in no pool of the
            // page change's.
            assert_eq!(zero, ["host unmap"]);
        }

        /// A throw before the lock and one under it push nothing.
        #[test]
        fn a_throw_leaves_nothing_pushed() {
            let t = dev();
            let mut got = (Vec::new(), Vec::new());
            with_test_native(|| {
                assert!(t.device.set_shared_display_state_page(0, 9, false).is_err());
                got.0 = events();
                assert!(t.device.set_shared_display_state_page(0, 1, false).is_err());
                got.1 = events();
            });
            assert!(got.0.is_empty(), "{:?}", got.0);
            assert!(got.1.is_empty(), "{:?}", got.1);
        }
    }

    /// The EFI display holding a display's last reference (testing.md 7.3): a host release of
    /// the display while a present, or a mode-count read, holds it leaves the display's end to
    /// that holder, and the end's detach waits for the device's display mutex. Each wait here is
    /// bounded; the threads that hang stay behind, and the test's process ends with them.
    mod efi_display_holding_the_last_reference {
        use super::*;
        use crate::host::HostQueue;
        use crate::testkit::host::{FakeHostQueue, QueueMode};
        use std::sync::mpsc;
        use std::thread::ThreadId;
        use std::time::Duration;

        /// How long a hung call is given to return before the test calls it hung.
        const HUNG: Duration = Duration::from_secs(1);
        /// How long a step the test waits for may take.
        const STEP: Duration = Duration::from_secs(10);

        /// A display of the device's own on port 0, the EFI display's port, parked (no guest
        /// display).
        fn parked_on_port_0(t: &TestDevice) -> DisplayHandle<FakePlatform> {
            let queue: Arc<dyn HostQueue> = FakeHostQueue::new(QueueMode::Manual);
            let mut desc = DisplayDescriptor::sdk_defaults();
            desc.queue = Some(queue);
            let d = t.device.new_display(&desc, 0, 0x1234).expect("display");
            assert!(t.device.displays.lock().display_client[0].is_some());
            d
        }

        /// The thread the display's end begins on, sent as it begins.
        fn end_begins(d: &DisplayHandle<FakePlatform>) -> mpsc::Receiver<ThreadId> {
            let (tx, rx) = mpsc::channel();
            d.core().drop_probe.lock().replace(Box::new(move || {
                let _ = tx.send(std::thread::current().id());
            }));
            rx
        }

        /// Waits (`STEP` at most, in 1 ms steps) until `f` holds.
        fn wait_until(what: &str, f: impl Fn() -> bool) {
            for _ in 0..STEP.as_millis() {
                if f() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            panic!("timed out waiting for {what}");
        }

        /// A present holds the display when the host releases it, while the guest's EFI mode
        /// write holds the display mutex and waits for the present queue (its mode rebuild
        /// stops presentation there): the present's end of the display waits for the mutex, and
        /// neither returns.
        #[test]
        fn a_present_ending_the_display_under_a_register_write_deadlocks() {
            let t = dev();
            let host_display = parked_on_port_0(&t);
            t.device.set_efi_display(1);
            let efi = t.efi_display().expect("EFI display");
            let (loaded_tx, loaded_rx) = mpsc::channel();
            let (go_tx, go_rx) = mpsc::channel::<()>();
            efi.loaded_probe.lock().replace(Box::new(move || {
                let _ = loaded_tx.send(std::thread::current().id());
                let _ = go_rx.recv_timeout(STEP);
            }));
            let ended = end_begins(&host_display);
            // A present, on the present queue.
            efi.source.merge(1);
            let presenter = loaded_rx
                .recv_timeout(STEP)
                .expect("the present loads the display");
            // The host's release: the present holds the last reference now.
            drop(host_display);
            // The guest's mode write, on its vCPU's thread.
            let (wrote_tx, wrote_rx) = mpsc::channel();
            let device = Arc::clone(&t.device);
            #[allow(clippy::disallowed_methods)] // test scaffolding: the guest's vCPU thread
            std::thread::spawn(move || {
                device.set_efi_framebuffer_mode(0x0010_0010);
                let _ = wrote_tx.send(());
            });
            wait_until("the mode write to hold the display mutex", || {
                t.device.displays.try_lock().is_none()
            });
            // The present goes on, and the display's end begins on the present queue.
            go_tx.send(()).expect("the present waits");
            assert_eq!(ended.recv_timeout(STEP), Ok(presenter));
            assert_eq!(
                wrote_rx.recv_timeout(HUNG),
                Err(mpsc::RecvTimeoutError::Timeout),
                "the mode write returned"
            );
            assert!(t.device.displays.try_lock().is_none());
        }

        /// A mode-count read holds the display mutex around its load of the display: a host
        /// release meanwhile leaves the end to the read, whose detach waits for the mutex its
        /// own thread holds, and the read never returns.
        #[test]
        fn a_mode_count_read_ending_the_display_deadlocks() {
            let t = dev();
            let host_display = parked_on_port_0(&t);
            t.device.set_efi_display(1);
            let efi = t.efi_display().expect("EFI display");
            let ended = end_begins(&host_display);
            // The host's release, inside the read's load.
            efi.loaded_probe
                .lock()
                .replace(Box::new(move || drop(host_display)));
            let (read_tx, read_rx) = mpsc::channel();
            let device = Arc::clone(&t.device);
            #[allow(clippy::disallowed_methods)] // test scaffolding: the guest's vCPU thread
            let reader = std::thread::spawn(move || {
                let _ = read_tx.send(device.get_efi_mode_count());
            });
            assert_eq!(ended.recv_timeout(STEP), Ok(reader.thread().id()));
            assert_eq!(
                read_rx.recv_timeout(HUNG),
                Err(mpsc::RecvTimeoutError::Timeout),
                "the mode-count read returned"
            );
            assert!(t.device.displays.try_lock().is_none());
        }
    }
}

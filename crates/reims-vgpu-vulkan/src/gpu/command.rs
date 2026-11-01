//! `id<MTLCommandQueue>` and `id<MTLCommandBuffer>`: core's `CommandQueue` and `CommandBuffer`
//! over the runtime's.
//!
//! - **Handlers** run on the runtime's handler threads, in the queue's order. A completed
//!   handler reads its buffer's `status` and `error` when it runs, as on the Metal platform, and
//!   runs marked as ours, so a panic in it aborts.
//! - **`encodeSignalEventScheduled:value:`** has no runtime call: it is a scheduled handler that
//!   sets the event's value, which is what Metal's does when the buffer is scheduled.
//! - **A nil event**, which the runtime's calls cannot take, dies at the send.
//! - **An open encoder** kills the command buffer's sends that Metal asserts at while one is
//!   ("with uncommitted encoder"): a new encoder, an event, `setProtectionOptions:`, `commit`.
//! - **Where a command buffer is** ([`Progress`]) is kept for what changes once the GPU has run
//!   it: a rate map's reset is taken again once the passes that took it completed, the indirect
//!   command buffers its commands use keep their numbers until then, and what is kept beside
//!   their commands moves once its blits ran ([`crate::gpu::icb`]). A completed handler of this
//!   platform's keeps it, added the first time an encoder needs it, so before the handlers core
//!   adds once it has encoded.
//! - **The records of an indirect command buffer an execution runs** are checked by the runtime
//!   as they stand at commit, or, where the GPU may have written them, once the work before the
//!   execution completed. A refusal at commit dies there naming the runtime's refusal, and says
//!   it is of the records where the runtime names an execution's ([`committing`]). The later one
//!   is a host step's, which fails the command buffer with `CommandBufferError::Refused`, as does
//!   one of an indirect dispatch's arguments: Metal runs those commands and fails no command
//!   buffer so, and it dies naming the runtime's refusal (`CommandBuffer::refusal`), never a
//!   failure the guest could see. It dies as the command buffer completes, before any handler of
//!   core's learns of it, at the selector of its first execution of an indirect command buffer
//!   ([`Ending`]), and at `commit` where its status or error is read first ([`failure`]). An
//!   isolated task's command buffer that completes after the task ended dies at neither, as
//!   core's handlers of the task run none then.
//! - **An execution's commands that decide a death** (a draw of points under a state whose points
//!   Metal sizes none of; a command whose state can encode indirect commands in an encoder that
//!   made resident for writing a buffer whose ID is not the runtime's) are read at the commit, as
//!   the runtime runs them, and die there naming the execution and the command; an execution of a
//!   buffer whose records the GPU may have written dies there whatever they hold, as the runtime
//!   reads them only once the work before it completed ([`executions`]). An encoder the
//!   command buffer's failure runs nothing of is not asked.
//! - **A draw's or dispatch's pipeline the driver did not make**, which the runtime makes while
//!   the command goes on and reports as the command buffer's failure
//!   (`CommandBufferError::PipelineCreation`), dies
//!   naming the driver's error ([`failed_pipeline`]) before anything of the failure reaches
//!   core: at `commit` where its status or error is read first, which a completed handler's
//!   does before it runs core's block ([`failure`]), and at `waitUntilCompleted` as the wait
//!   returns, as the runtime runs the completed handlers before its waiters return. Metal makes
//!   a state's pipelines as it makes the state, and fails no command buffer so.
//! - **An indexed indirect draw's indices past the command buffer's index pool**
//!   (`CommandBufferError::IndexPoolOverflow`) die there too ([`pool_overflow`]): the runtime
//!   checks such a draw's indices, or rewrites a strip's that restarts at another index than the
//!   all-ones one, into a pool before the render pass, a sixty-fourth of the device's largest
//!   device-local heap, and draws nothing of a draw past it, where Metal
//!   draws it.
//! - **A sample through a cubic sampler an argument buffer holds**
//!   (`CommandBufferError::CubicArgumentSampler`) dies there too ([`cubic_argument_sampler`]):
//!   the runtime does not filter bicubically through an argument buffer's sampler, and fails a
//!   command buffer whose function reached one, where Metal samples and fails none. A function
//!   that never reaches one runs as Metal's.
//! - **A command whose argument samplers the runtime could not list as its command buffer
//!   committed, or as a host step recorded it** (`CommandBufferError::UnlistedArgumentSampler`)
//!   dies there too ([`unlisted_argument_sampler`]): more sampler states live than the command's
//!   set-1 sampler arrays hold (at most 65536 elements each, fewer where the device's limits
//!   are) or than its sampler tables hold, or one of a shape they refuse, such as a cubic state
//!   beside a fragment stage. A command a host step records after the commit, of an indirect
//!   command buffer whose records the GPU writes, is in no commit's listing: it fails its command
//!   buffer as the step records it, where more states live than its set-1 sampler arrays hold.
//!   The runtime fails such a command buffer whether or not its functions reach the state left
//!   out, where Metal's reach every state by its ID and fail none. The runtime's announcement of
//!   the first (`unlisted-argument-sampler`) is relayed, as the failure that follows it dies.
//! - **A host step's Vulkan object that nothing made before it**
//!   (`CommandBufferError::HostStepObject`) dies there too, naming the object's kind
//!   ([`host_step_object`]): the runtime records a render encoder's execution of an indirect
//!   command buffer the GPU may write, or of a range read from a buffer, in a host step, which
//!   makes no image, render pass, framebuffer or neutral binding. Where a command's draw needs
//!   one that nothing made before the step, the runtime fails the command buffer, and runs
//!   nothing of the batch the step recorded for or of those after it, where Metal draws the
//!   command and fails none.
//! - **A command whose buffer over the guest's pages was rebound before its command buffer was
//!   submitted** (`CommandBufferError::EncodedUseRebound`) dies there too
//!   ([`encoded_use_rebound`]): the runtime resolves a command's use of a buffer over the
//!   caller's host pages as the command is encoded, so where the buffer is rebound (a backing
//!   replaced, `crate::gpu::backing`) before the command buffer's first submission, it runs
//!   nothing of each submission holding the command and fails the command buffer, where Metal
//!   resolves the use as it submits and runs the command over the pages the buffer lies in then.
//!   The runtime announces the first such failure (`EncodedUseRebound`, [`crate::answer::DIES`])
//!   on the thread that makes the submission, where it dies first, ending the process.
//! - **Every other failure of the runtime's own**, where Metal fails no command buffer, dies
//!   there too, naming it ([`answered`]): an emulated mesh draw past the command buffer's mesh
//!   records pool (`MeshRecordsOverflow`), which it draws nothing of; a sparse mapping update it
//!   applied none of (`SparseHeapSplit`, `SparseMappingArguments`, `SparseMappingSplit`); a
//!   Vulkan failure of a submission other than a lost device or memory
//!   (`Internal`); and any other the runtime gives, whatever origin it states, until one is
//!   reviewed here. Metal's own failures, the seven the runtime states as Metal's
//!   (`FailureOrigin::Metal`) and this platform lists, reach core as Metal's: those this platform
//!   commits a command buffer with as Metal's GPU fails it (below), a lost device, memory, an
//!   address fault, a render pass without attachments past 16384, a detached buffer's use, and
//!   the driver's relay of one of Metal's codes.
//! - **A failure of the GPU's work that the driver relayed** (`CommandBufferError::Execution`, and
//!   `PageFault` and `DeviceLost` of the same kind): MoltenVK runs the runtime's batches as Metal
//!   command buffers, and the runtime gives a failed one's Metal code to the command buffer whose
//!   batch it was, the device kept, as Metal keeps it. Core sees that code as Metal's failure of
//!   the command buffer where it is one of Metal's ([`METAL_CODES`]); any other dies where the
//!   failure is first read, as a pipeline's does ([`relayed`]). Metal's description of the
//!   failure, as the driver's message gave it, goes beside the code in the error's description,
//!   which core's log lines of a failed command buffer print, and in the death's line
//!   ([`command_buffer_error`]).
//! - **A wait the runtime refuses** as one that would not return hangs where Metal's never
//!   returns, and dies where Metal's answer is not measured, by the cause the runtime names
//!   ([`refused_wait`]).
//! - **Command predication**'s units of compute encoders end at a blit or render encoder, an
//!   event and the commit ([`predication::boundary`]).
//! - **How Metal's GPU fails a command buffer for what its encoders did** is injected at its
//!   commit ([`rt::CommandBuffer::commit_failed`]): Metal's code, the first encoder that runs
//!   nothing, every one after it running nothing too, and whether its event signals fire
//!   ([`Failing`]). Two different ways at once, which nothing measured, die at the commit.
//!   Measured on the M3 Max (macOS 26.7):
//!   - a render encoder's draw of primitive type 9 hangs Metal's GPU, which fails the command
//!     buffer with `MTLCommandBufferErrorTimeout` from that encoder, its signals firing
//!     ([`crate::gpu::encode_render`]);
//!   - `setProtectionOptions:` is kept here, as the runtime has no protected content
//!     ([`CommandBufferObject::made_encoder`]). An encoder made under options other than 0 runs
//!     nothing, nor does any after it, and fails the command buffer with code 16 ("Protection
//!     Violation") below 0x10 and `MTLCommandBufferErrorInternal` ("Invalid Input") from 0x10
//!     up, none of its signals firing; options from 0x10 up fail it so with no encoder made
//!     under them too, every encoder running. An encoder made before the options runs. Under
//!     1, 3 and 7 a compute or render encoder's creation, and a blit encoder's first command,
//!     die, as Metal ends the process there.
//!
//!   A hang reaches the queue's later command buffers too ([`Hangs`]). Measured on the M3 Max
//!   (macOS 26.7, `metal_runs_after_a_type_9_hang_here`, 11 hangs): Metal fails the hang's
//!   command buffer 30 to 37 ms after its commit, and runs nothing of a command buffer of the
//!   same queue submitted meanwhile, committed at once or up to 10 ms on, completing it without
//!   an error (status `Completed`), its event signals firing and its completed handlers seeing
//!   that. One of another queue runs, and so does one of the same queue whose event wait held it
//!   until the hang completed, and one committed after it. The runtime completes no command
//!   buffer so: one committed on a queue while a hang of that queue is in flight here dies at
//!   its commit. The hang stays in flight here as long as Metal's: its command buffer holds what
//!   comes after its hanging encoder (its later signals, its completion) until [`HANG_TIME`]
//!   after the GPU reached it ([`CommandBufferObject::hold_after_hang`]). A protection failure
//!   reaches no later command buffer: one committed on its queue while it is in flight runs,
//!   on Metal (measured alike) as here.
//! - **A submission that uses a detached buffer or heap** fails with
//!   `MTLCommandBufferErrorOutOfMemory` (8), whose description is Metal's ("Insufficient
//!   Memory"), as Metal fails it. Metal splits a command buffer into submissions, and decides
//!   each as it submits it, once the event waits before it are met (measured on the M3 Max,
//!   macOS 26.7, `tests/detach_backing.rs`):
//!   - every render pass is a submission of its own;
//!   - a run of consecutive blit encoders is one, and so is a run of consecutive compute
//!     encoders;
//!   - an event wait or signal ends a run.
//!
//!   The runtime splits a command buffer so, and checks each submission as it submits it, once
//!   its event waits are met ([`backing::submission_check`]): a wait another command buffer's
//!   GPU signal meets, once that signal landed, as Metal checks it once the signal happened. One
//!   that uses a detached object, or one released detached since the command buffer was made,
//!   runs nothing; every other submission runs, the event signals fire, the scheduled and
//!   completed handlers run, and the command buffer completes in its queue's order with status
//!   `Error`. The use can be a binding, a `useResource:` or `useHeap:`, a blit's operand, a
//!   command of an indirect command buffer it executes, or a texture over the buffer, a view of
//!   that texture, or a resource placed in the heap, as the runtime lists a submission's uses.
//!   This platform differs from Metal where:
//!   - the object is replaced after the command was encoded and before its command buffer's
//!     first submission (committed after the replace, or committed before it and held behind an
//!     event wait): Metal runs the submission over the new pages. The runtime resolved the
//!     command's use as it encoded it, and fails the command buffer
//!     (`CommandBufferError::EncodedUseRebound`), which dies ([`encoded_use_rebound`]). The
//!     runtime decides each submission so as it submits it, after the check: one the check
//!     fails, a use detached then, keeps code 8, as Metal's, and a later one rebound so dies all
//!     the same, the command buffer keeping that code;
//!   - a submission that executes an indirect command buffer the GPU may write is checked again
//!     once the runtime recorded what its commands reach, and failing then, only what had not
//!     run yet runs nothing. Metal decides it once, before any of it runs.
//!
//!   A command that reaches a detached object only through an address it loads names nothing the
//!   check sees, so its submission runs, as Metal's does, reading 0 and dropping the write, on
//!   every host: the runtime's detach leaves the object in no memory, replaced or never replaced,
//!   and out of what such a command reaches until a command buffer uses it after a replace
//!   (`backing`'s docs).
//!
//!   A texture over a buffer or placed in a heap, and a buffer placed in a heap, hold the buffer
//!   or heap, as Metal's do, so a detached one is released only with the last of them, and a use
//!   through one fails until then ([`crate::gpu::resource::Under`]).
//!
//!   A command buffer Metal's GPU fails for what its encoders did is decided at its commit, with
//!   no check: one that uses a detached object then dies there, as one failing two ways does.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use metal2vulkan::metal::DispatchType;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::command::{
    CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo,
};
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
use reims_vgpu_core::platform::values::ComputePassDesc;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::tasks::isolation::TaskLife;

use crate::answer::{
    Policy, get, hang, metal_aborts, plain, refusal, refused, send, unrepresentable,
};
use crate::gpu::backing::{self, Opened};
use crate::gpu::desc::VulkanRenderPassDescriptor;
use crate::gpu::device::VulkanDevice;
use crate::gpu::encode_blit::{EncoderObject, VulkanBlitEncoder};
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::encode_render::VulkanRenderEncoder;
use crate::gpu::event::VulkanSharedEvent;
use crate::gpu::icb::VulkanIcb;
use crate::gpu::metal_space::{self, FirstUse};
use crate::gpu::predication::{self, Predication};
use crate::gpu::timing;
use crate::platform::{Obj, VulkanPlatform};

pub(crate) mod executions;

/// A runtime command queue, with the device it came from.
#[derive(Debug)]
pub struct QueueObject {
    /// The runtime's queue.
    pub raw: rt::CommandQueue,
    /// The device that made it.
    pub device: VulkanDevice,
    /// Its command buffers Metal's GPU hangs at that are in flight ([`Hangs`]).
    hangs: Arc<Hangs>,
}

impl QueueObject {
    /// The queue of `raw`, made by `device`.
    pub(crate) fn new(raw: rt::CommandQueue, device: VulkanDevice) -> QueueObject {
        QueueObject {
            raw,
            device,
            hangs: Arc::default(),
        }
    }
}

/// How many of a queue's command buffers that Metal's GPU hangs at (a draw of primitive type 9,
/// [`HANG`]) are in flight: from the commit until [`HANG_TIME`] after the GPU reached the hang
/// ([`CommandBufferObject::hold_after_hang`]). Metal runs nothing of a command buffer of the
/// queue submitted meanwhile, and completes it without an error (the module docs), which the
/// runtime has no form for: one committed meanwhile dies at its commit ([`Hangs::committing`]).
#[derive(Debug, Default)]
struct Hangs(AtomicUsize);

impl Hangs {
    /// Dies at `commit` of a command buffer of the queue while one of its hangs is in flight.
    fn committing(&self) {
        let hangs = self.0.load(Ordering::SeqCst);
        if hangs != 0 {
            unrepresentable(
                "commit",
                "a command buffer committed on a queue while a command buffer of that queue \
                 that Metal's GPU hangs at (a draw of primitive type 9) is in flight: Metal runs \
                 none of it and completes it without an error, its event signals firing, and \
                 the runtime completes no command buffer so",
            );
        }
    }
}

/// How long a command buffer that Metal's GPU hangs at stays in flight once the GPU reached the
/// hang: Metal fails it 30 to 37 ms after its commit on the M3 Max (11 hangs, the module docs),
/// so every command buffer of its queue Metal runs nothing of is committed within it.
pub(crate) const HANG_TIME: std::time::Duration = std::time::Duration::from_millis(40);

/// `id<MTLCommandQueue>`.
pub type VulkanCommandQueue = Obj<QueueObject>;

/// A runtime command buffer, with the device it came from.
#[derive(Debug)]
pub struct CommandBufferObject {
    /// The runtime's command buffer.
    pub raw: rt::CommandBuffer,
    /// The device of its queue.
    pub device: VulkanDevice,
    /// How many of its encoders are open: `setProtectionOptions:`, which the runtime answers
    /// whatever is open, dies while one is.
    pub open_encoders: Arc<AtomicUsize>,
    /// Its progress, once an encoder needed it ([`CommandBufferObject::progress`]).
    progress: Mutex<Option<Arc<Progress>>>,
    /// Its command predication.
    pub(crate) predication: Arc<Predication>,
    /// Its place among the releases of detached objects until it commits, which its submission
    /// check then holds ([`backing::submission_check`]).
    opened: Mutex<Option<Opened>>,
    /// How many runtime encoders it has made: each encoder of Metal's, or a part of one where
    /// [`predication`] ends a compute encoder's runtime encoder and makes another. As no two are
    /// open at once, each one's count before it is its place among the encoders the runtime
    /// ended, which [`rt::CommitFailure::encoders`] counts.
    made: AtomicUsize,
    /// How Metal's GPU fails it for what its encoders did ([`Failing`]), which its commit fails
    /// it with ([`CommandBufferObject::failure`]).
    failing: Mutex<Vec<Failing>>,
    /// The protection options `setProtectionOptions:` set, which the runtime does not hold.
    protection: Mutex<Protection>,
    /// Whether it holds work for Metal's tiler ([`CommandBufferObject::tiler_work`]), which its
    /// submission handler reads ([`tiler_handler`]).
    tiler: Arc<AtomicBool>,
    /// Its queue's hangs in flight ([`Hangs`]).
    hangs: Arc<Hangs>,
    /// The event that holds what comes after the first encoder Metal's GPU hangs at, where one
    /// did ([`CommandBufferObject::hold_after_hang`]).
    hold: Mutex<Option<rt::SharedEvent>>,
    /// The executions of indirect command buffers whose commands its commit reads, each with its
    /// encoder's place ([`CommandBufferObject::checked_at_commit`]).
    executions: Mutex<Vec<(usize, executions::Execution)>>,
}

/// A new command buffer's submission handler: Metal's allocation for its work for the tiler,
/// where it holds some ([`CommandBufferObject::tiler_work`]). Metal takes its own allocation for
/// the first such work in the process as it submits it to the GPU
/// ([`FirstUse::TilerSubmission`]; `vaprobe rendersubmit`, plain Metal, M3 Max: none of it at the
/// commit of a render pass behind an event wait, all of it by the command buffer's completion).
/// The runtime runs the handler as the command buffer's last batch reaches its queue, on the
/// thread that submits it, before the command buffer reads scheduled, before
/// `waitUntilScheduled` returns and before any scheduled or completed handler runs. Every command
/// buffer this platform commits carries a submission check ([`backing::submission_check`]), which
/// the runtime runs on its scheduler thread, so that thread submits it, once its waits are met
/// and the work ahead of it on its queue is submitted: after `commit` returns, as Metal submits
/// on a thread of its own and tells the program only through scheduled (the runtime's
/// measurement on an M4). A range placed between the commit and the submission lies by the threads'
/// timing, on Metal as here. A command buffer the runtime fails before its submission takes
/// nothing.
fn tiler_handler(tiler: &Arc<AtomicBool>) -> rt::Handler {
    let tiler = Arc::clone(tiler);
    Box::new(move |_| {
        if tiler.load(Ordering::SeqCst) {
            metal_space::first_use(FirstUse::TilerSubmission);
        }
    })
}

/// The end of a hang held by `hold` ([`CommandBufferObject::hold_after_hang`]), committed by
/// `sel`: once the GPU reached the hang (`hold` at 1), [`HANG_TIME`] on, on a listener's thread
/// of its own, the queue's hang is no longer in flight (`hangs`), then the hold lets what comes
/// after the hang go (`hold` to 2), as Metal's GPU fails the command buffer then.
fn hang_ends(sel: &'static str, hold: rt::SharedEvent, hangs: Arc<Hangs>) {
    let listener = get(sel, rt::SharedEventListener::new);
    let end: rt::SharedEventNotification = Box::new(move |hold, _| {
        std::thread::sleep(HANG_TIME);
        hangs.0.fetch_sub(1, Ordering::SeqCst);
        hold.set_signaled_value(2);
    });
    hold.notify_listener(&listener, 1, end);
}

/// `MTLCommandBufferErrorInternal`, which Metal's GPU fails a command buffer with under
/// protection options with a bit from 0x10 up ("Invalid Input").
pub(crate) const INVALID_INPUT: u64 = 1;

/// `MTLCommandBufferErrorTimeout`, which Metal's GPU fails a command buffer with where a render
/// encoder drew primitive type 9 ("Caused GPU Hang Error").
pub(crate) const HANG: u64 = 2;

/// The error code Metal's GPU fails a command buffer with where an encoder was made under
/// protection options below 0x10 ("Protection Violation"), which no public
/// `MTLCommandBufferError` names.
pub(crate) const PROTECTION_VIOLATION: u64 = 16;

/// The codes Metal fails a command buffer with: each `MTLCommandBufferError` but `None`, and
/// [`PROTECTION_VIOLATION`].
pub(crate) const METAL_CODES: [u64; 11] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, PROTECTION_VIOLATION];

/// A way Metal's GPU fails a command buffer for what one of its encoders did: with `code`, from
/// the encoder at `from` on, its event signals firing where `signals` says (the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Failing {
    /// The `MTLCommandBufferError` code.
    pub code: u64,
    /// The first encoder that runs nothing, in the order the runtime ended them.
    pub from: usize,
    /// Whether its event signals fire.
    pub signals: bool,
}

/// What `setProtectionOptions:` left on a command buffer.
#[derive(Debug, Default, Clone, Copy)]
struct Protection {
    /// The options set last: 0 until one is set.
    options: u64,
    /// Whether an encoder was made under options other than 0.
    used: bool,
}

/// Whether protection options end the process where an encoder is made under them: 1, 3 and 7,
/// at a compute or render encoder's creation and a blit encoder's first command (measured on
/// the M3 Max: SIGABRT, no message).
fn aborts_encoders(options: u64) -> bool {
    matches!(options, 1 | 3 | 7)
}

#[cfg(test)]
impl CommandBufferObject {
    /// Commits it as failed with `failure`, as `commit` commits it: the runtime's extension for
    /// error injection ([`rt::CommandBuffer::commit_failed`]), which fails it with no GPU fault.
    pub(crate) fn commit_failed(&self, failure: rt::CommitFailure) {
        if let Some(progress) = &*self.progress.lock() {
            progress.committing();
        }
        self.raw.commit_failed(failure).unwrap();
    }
}

impl CommandBufferObject {
    /// Work for Metal's tiler encoded, `first`: a render encoder made, or a blit's reset or copy
    /// of an indirect command buffer. Metal's own allocations for it the first time, and the
    /// command buffer's submission a first use too ([`tiler_handler`]).
    pub(crate) fn tiler_work(&self, first: FirstUse) {
        metal_space::first_use(first);
        self.tiler.store(true, Ordering::SeqCst);
    }

    /// Dies at `sel`, the creation of a compute or render encoder, under protection options that
    /// end the process there ([`aborts_encoders`]).
    pub(crate) fn protected_creation(&self, sel: &'static str) {
        let options = plain(|| self.protection.lock().options);
        if aborts_encoders(options) {
            metal_aborts(
                sel,
                &format!("an encoder made under protection options {options}"),
            );
        }
    }

    /// A runtime encoder it made, for an encoder of Metal's (`whole`) or a part of one: its place
    /// among the encoders the runtime ends. An encoder of Metal's made under protection options
    /// other than 0 fails the command buffer from it, with
    /// [`PROTECTION_VIOLATION`] below 0x10 and [`INVALID_INPUT`] from there up, its event
    /// signals firing none; under options that end the process at its first command (a blit
    /// encoder's, [`aborts_encoders`]) it answers them, which that command dies at. Nothing here
    /// stops a thread of a task that ended: the encoder is not in its handle yet, and dropped
    /// it would abandon the command buffer.
    pub(crate) fn made_encoder(&self, whole: bool) -> (usize, Option<u64>) {
        let at = self.made.fetch_add(1, Ordering::SeqCst);
        if !whole {
            return (at, None);
        }
        let options = {
            let mut protection = self.protection.lock();
            if protection.options != 0 {
                protection.used = true;
            }
            protection.options
        };
        if options == 0 {
            return (at, None);
        }
        if aborts_encoders(options) {
            return (at, Some(options));
        }
        self.fail(Failing {
            code: protection_code(options),
            from: at,
            signals: false,
        });
        (at, None)
    }

    /// Records `failing`, which its commit fails it with ([`CommandBufferObject::failure`]).
    pub(crate) fn fail(&self, failing: Failing) {
        self.failing.lock().push(failing);
    }

    /// `execution`, a command of the encoder it made last, is checked at its commit, where its
    /// commands' records stand as the runtime runs them ([`executions`]).
    pub(crate) fn checked_at_commit(&self, execution: executions::Execution) {
        let at = self.made.load(Ordering::SeqCst).saturating_sub(1);
        plain(|| self.executions.lock().push((at, execution)));
    }

    /// Checks at its commit the executions [`CommandBufferObject::checked_at_commit`] took, in
    /// their order, but those of an encoder `failed` runs nothing of; each whose records it read
    /// for a death is held as read until it ends ([`executions::reads`]).
    fn check_executions(&self, failed: Option<&rt::FailedEncoders>, progress: Option<&Progress>) {
        let taken = std::mem::take(&mut *self.executions.lock());
        let runs = |at: usize| match failed {
            Some(rt::FailedEncoders::From(from)) => at < *from,
            Some(rt::FailedEncoders::Set(set)) => !set.contains(&at),
            _ => true,
        };
        let read: Vec<_> = taken
            .into_iter()
            .filter(|(at, _)| runs(*at))
            .filter_map(|(_, execution)| execution.check())
            .collect();
        if let Some(progress) = progress {
            for icb in &read {
                executions::reads(icb, progress);
            }
        }
    }

    /// After `sel`, the end of an encoder Metal's GPU hangs at (a draw of primitive type 9),
    /// what comes after it waits as on Metal, whose GPU fails the command buffer [`HANG_TIME`]
    /// after it reaches the hang: a signal of an event of its own to 1, which fires once the work
    /// before the hang ran, then a wait for 2, which its commit sets [`HANG_TIME`] after the
    /// signal ([`hang_ends`]). So its later event signals and its completion come then, and its
    /// queue's hang is in flight until then ([`Hangs`]). The first such encoder alone holds it,
    /// as nothing after it runs.
    pub(crate) fn hold_after_hang(&self, sel: &'static str) {
        let mut hold = plain(|| self.hold.lock());
        if hold.is_some() {
            return;
        }
        let event = get(sel, || self.device.raw.new_shared_event());
        let at = || rt::EventRef::Shared(&event);
        get(sel, || self.raw.encode_signal_event(at(), 1));
        get(sel, || self.raw.encode_wait_for_event(at(), 2));
        *hold = Some(event);
    }

    /// How Metal's GPU fails it, at its commit (`sel`): what its encoders did ([`Failing`]),
    /// and protection options with a bit from 0x10 up, which fail it with [`INVALID_INPUT`]
    /// though no encoder was made under them (measured), all of its encoders running. Where it
    /// fails so, a use of a detached object at the commit (`detached`, asked only then) is
    /// another way. Two different ways at once, which nothing measured, die. A use of a
    /// detached object alone fails only its submissions ([`backing::submission_check`]).
    fn failure(
        &self,
        sel: &'static str,
        detached: impl FnOnce() -> bool,
    ) -> Option<rt::CommitFailure> {
        let mut ways = plain(|| self.failing.lock().clone());
        let options = plain(|| self.protection.lock().options);
        if protection_code(options) == INVALID_INPUT {
            ways.push(Failing {
                code: INVALID_INPUT,
                from: self.made.load(Ordering::SeqCst),
                signals: false,
            });
        }
        let first = *ways.first()?;
        if detached() {
            ways.push(Failing {
                code: backing::DETACHED_USE,
                from: 0,
                signals: true,
            });
        }
        if ways
            .iter()
            .any(|w| (w.code, w.signals) != (first.code, first.signals))
        {
            unrepresentable(
                sel,
                &format!("a command buffer Metal's GPU fails in more than one way: {ways:?}"),
            );
        }
        let mut failure = rt::CommitFailure::new(first.code);
        let from = ways.iter().map(|w| w.from).min().unwrap_or(first.from);
        failure.encoders = rt::FailedEncoders::From(from);
        failure.signals = first.signals;
        Some(failure)
    }

    /// Its progress ([`Progress`]), for `sel`, an encoder's command: made the first time one asks,
    /// with the completed handler that ends it, which then comes before every handler core adds
    /// once it has encoded. An encoder's command comes before the commit, which refuses an open
    /// encoder, so the handler's refusal after the commit never comes.
    pub(crate) fn progress(&self, sel: &'static str) -> Arc<Progress> {
        let mut kept = plain(|| self.progress.lock());
        if let Some(progress) = &*kept {
            return Arc::clone(progress);
        }
        let progress = Arc::new(Progress::default());
        let ending = Ending {
            progress: Some(Arc::clone(&progress)),
            task: TaskLife::current(),
            #[cfg(test)]
            slot: reims_vgpu_core::testkit::capture::current_slot(),
        };
        let handler: rt::Handler = Box::new(move |cb: &rt::CommandBuffer| ending.run(cb));
        plain(|| self.raw.add_completed_handler(handler)).unwrap_or_else(|e| refused(sel, &e));
        *kept = Some(Arc::clone(&progress));
        progress
    }
}

/// Where a command buffer is, as [`Progress`] sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Stage {
    /// Not committed yet.
    #[default]
    Encoding,
    /// Committed, not yet complete: what the GPU has run of it is unknown.
    Committed,
    /// Completed: the GPU ran it all.
    Completed,
    /// Failed: what the GPU ran of it is unknown, for good.
    Failed,
    /// Gone without a commit: the GPU ran none of it.
    Dropped,
}

/// What this platform keeps of one command buffer until the GPU has run it: its [`Stage`], what
/// its commands use until then, and what changes once it completed or failed.
#[derive(Default)]
pub struct Progress {
    state: Mutex<ProgressState>,
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress")
            .field("stage", &self.stage())
            .finish_non_exhaustive()
    }
}

/// A command buffer's [`Progress`], under its lock.
#[derive(Default)]
struct ProgressState {
    stage: Stage,
    /// What its commands use, held until it completes, fails or goes, as Metal's command buffer
    /// retains them.
    held: Vec<Box<dyn std::any::Any + Send + Sync>>,
    /// What changes once it completed, in the order its commands were encoded.
    changes: Vec<Box<dyn FnOnce() + Send>>,
    /// What changes once it failed, in the order its commands were encoded.
    failures: Vec<Box<dyn FnOnce() + Send>>,
    /// The indirect command buffers its commands execute, with the executions' selectors, held as
    /// `held` is.
    executions: Vec<(&'static str, VulkanIcb)>,
}

impl Progress {
    /// Where the command buffer is now.
    pub(crate) fn stage(&self) -> Stage {
        plain(|| self.state.lock().stage)
    }

    /// Whether the GPU may still run some of it: it is not committed, or committed and not
    /// complete.
    pub(crate) fn running(&self) -> bool {
        matches!(self.stage(), Stage::Encoding | Stage::Committed)
    }

    /// `object` is held until the command buffer completes, fails or goes.
    pub(crate) fn hold(&self, object: impl std::any::Any + Send + Sync) {
        plain(|| self.state.lock().held.push(Box::new(object)));
    }

    /// An execution `sel` of `icb`, which is held until the command buffer completes, fails or
    /// goes.
    pub(crate) fn executes(&self, sel: &'static str, icb: VulkanIcb) {
        plain(|| self.state.lock().executions.push((sel, icb)));
    }

    /// The selector of its first execution of an indirect command buffer, if its commands
    /// execute one.
    fn first_execution(&self) -> Option<&'static str> {
        self.state.lock().executions.first().map(|(sel, _)| *sel)
    }

    /// `change` runs once the command buffer completed, after the changes added before it and
    /// before its stage is set; it is dropped unrun when the command buffer fails or goes.
    pub(crate) fn at_completion(&self, change: impl FnOnce() + Send + 'static) {
        plain(|| self.state.lock().changes.push(Box::new(change)));
    }

    /// `change` runs once the command buffer failed, as [`Progress::at_completion`]'s once it
    /// completed; it is dropped unrun when the command buffer completes or goes.
    pub(crate) fn at_failure(&self, change: impl FnOnce() + Send + 'static) {
        plain(|| self.state.lock().failures.push(Box::new(change)));
    }

    /// The commit begins: no thread stop can come between this and the runtime's commit, so a
    /// command buffer never committed never shows as committed.
    fn committing(&self) {
        let mut state = self.state.lock();
        if state.stage == Stage::Encoding {
            state.stage = Stage::Committed;
        }
    }

    /// The command buffer ends at `stage`: its changes run if it completed, its failures' if it
    /// failed, what it held goes, then the stage is set. It runs on the runtime's thread, or
    /// where the command buffer goes, and sends nothing, so it checks no task's end.
    fn end(&self, stage: Stage) {
        let (changes, failures, held, executions) = {
            let mut state = self.state.lock();
            (
                std::mem::take(&mut state.changes),
                std::mem::take(&mut state.failures),
                std::mem::take(&mut state.held),
                std::mem::take(&mut state.executions),
            )
        };
        let ran = match stage {
            Stage::Completed => {
                drop(failures);
                changes
            }
            Stage::Failed => {
                drop(changes);
                failures
            }
            Stage::Encoding | Stage::Committed | Stage::Dropped => {
                drop((changes, failures));
                Vec::new()
            }
        };
        ran.into_iter().for_each(|change| change());
        drop((held, executions));
        self.state.lock().stage = stage;
    }
}

/// The completed handler that ends a command buffer's [`Progress`]; dropped unrun, as the runtime
/// drops the handlers of a command buffer that goes uncommitted, it ends it as gone.
struct Ending {
    /// The progress it ends, until it has.
    progress: Option<Arc<Progress>>,
    /// The isolated task whose code made the progress, if one's did: the command buffer is the
    /// task's, and its death in the handler is the task's code ([`as_task`]).
    task: Option<Arc<TaskLife>>,
    /// The test's death slot, which a death in the handler lands in.
    #[cfg(test)]
    slot: Option<Arc<reims_vgpu_core::testkit::capture::DeathSlot>>,
}

impl Ending {
    /// The handler, on the runtime's handler thread, marked ours: the command buffer completed or
    /// failed.
    fn run(mut self, cb: &rt::CommandBuffer) {
        let _env = BlockEnv::foreign();
        let stage = if cb.status() == rt::CommandBufferStatus::Error {
            Stage::Failed
        } else {
            Stage::Completed
        };
        let die = |sel| {
            refused_step(sel, cb);
        };
        let refused = cb.error() == Some(rt::CommandBufferError::Refused);
        self.end(stage, refused.then_some(die));
    }

    /// Ends the progress at `stage`, after `refused`'s death at the selector of its first
    /// execution of an indirect command buffer where a host step refused the command buffer and
    /// its commands execute one. The death is the task's code, so none comes once the task ended.
    fn end(&mut self, stage: Stage, refused: Option<impl Fn(&'static str)>) {
        let Some(progress) = self.progress.take() else {
            return;
        };
        let task = self.task.take();
        let end = || {
            if let Some(refused) = refused
                && let Some(sel) = progress.first_execution()
            {
                as_task(task.as_ref(), || refused(sel));
            }
            progress.end(stage);
        };
        #[cfg(test)]
        reims_vgpu_core::testkit::capture::with_slot(self.slot.take(), end);
        #[cfg(not(test))]
        end();
    }
}

/// Runs `f` as the code of `task`, the isolated task a command buffer is of (none outside
/// isolation), as core carries its completed handlers (`TaskLife::carry`): not at all once the
/// task ended, so that a task's command buffer that ends after it kills nothing. A platform death
/// in `f` still ends the process, as on the Metal platform a crash inside Metal does.
fn as_task(task: Option<&Arc<TaskLife>>, f: impl FnOnce()) {
    match task {
        Some(life) => {
            let _ = life.run(f);
        }
        None => f(),
    }
}

/// Runs the commit `call`: a refusal of the records of an indirect command buffer the command
/// buffer executes ([`of_records`]), which the runtime checks as they stand at commit, dies at
/// the commit, saying so and naming the runtime's refusal. Any other refusal is the caller's to
/// decide, which dies at the commit naming the runtime's refusal too ([`send`]): of the command
/// buffer's state, of the heaps' or function tables' entries, or of what else the runtime records
/// then.
fn committing(call: impl FnOnce() -> Result<(), rt::Error>) -> Result<(), rt::Error> {
    match call() {
        Err(e) if of_records(&e) => fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: "commit",
            },
            &format!(
                "the Vulkan runtime refused the records of an indirect command buffer the \
                 command buffer executes, which it checks as they stand at commit: {e}"
            ),
        ),
        made => made,
    }
}

/// Whether `e`, a refusal of a commit, is of the records of an indirect command buffer the
/// command buffer executes: the runtime names the method a refusal is of by its Metal selector,
/// and those of the records by an execution's (`execute_commands_in_buffer` and its indirect
/// form). A refusal it gives no method (a runtime gap, a compile) is named alone.
fn of_records(e: &rt::Error) -> bool {
    match e {
        rt::Error::Invalid(invalid) => invalid.call.starts_with("execute_commands_in_buffer"),
        _ => false,
    }
}

impl Drop for Ending {
    fn drop(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.end(Stage::Dropped);
        }
    }
}

/// Runs `call`, the command buffer's `sel`, as [`send`] does, but for an encoder still open:
/// Metal asserts there ("with uncommitted encoder": `exec_open_encoder_at_commit`,
/// `render_segments_open_twice`), so it dies whatever handler is live.
fn encoding<T>(
    sel: &'static str,
    policy: Policy,
    call: impl FnOnce() -> Result<T, rt::Error>,
) -> RaiseResult<T> {
    send(sel, policy, || match call() {
        Err(rt::Error::Invalid(invalid)) if matches!(invalid.problem, rt::Problem::EncoderOpen) => {
            uncommitted_encoder(sel)
        }
        made => made,
    })
}

/// The death of `sel` while an encoder of its command buffer is open, as Metal asserts.
fn uncommitted_encoder(sel: &'static str) -> ! {
    metal_aborts(
        sel,
        "a command buffer with an open encoder (\"with uncommitted encoder\")",
    )
}

/// Runs `call`, the wait `sel` for a command buffer (or for a new one of a queue), as [`send`]
/// does with `Dies`, but for a wait the runtime refuses as one that would not return
/// (`WaitWouldDeadlock`), decided by [`refused_wait`].
fn waiting<T>(sel: &'static str, call: impl FnOnce() -> Result<T, rt::Error>) -> RaiseResult<T> {
    send(sel, Policy::Dies, || match call() {
        Err(rt::Error::WaitWouldDeadlock(cause)) => {
            match refused_wait(cause, rt::process_exiting) {
                RefusedWait::Hangs => hang(),
                RefusedWait::Unmeasured => unmeasured_wait(sel, cause),
            }
        }
        made => made,
    })
}

/// The death of the wait `sel` the runtime refused for `cause`, as [`RefusedWait::Unmeasured`].
fn unmeasured_wait(sel: &'static str, cause: rt::DeadlockCause) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        &format!(
            "the Vulkan runtime refused the wait, as {cause}, where Metal's answer is not \
             measured"
        ),
    )
}

/// What a wait the runtime refuses does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefusedWait {
    /// It hangs, as Metal's never returns there: this thread parks for good ([`hang`]).
    Hangs,
    /// It dies: Metal's answer there is not measured.
    Unmeasured,
}

/// What a wait the runtime refuses for `cause` does, the process `exiting` or not:
/// - a handler's wait for its own command buffer or a later one of its queue, and a new command
///   buffer in the queue's own handler while all 64 are held (`OwnQueue`): Metal's never
///   returns there (measured), and a hang stays a hang;
/// - any wait for a command buffer not yet finished once the process is exiting (`Exiting`), on
///   any thread: Metal's goes on waiting, and the exit ends the thread with the process, so the
///   thread parks (the library registers no exit handler that waits);
/// - a handler's wait for another queue's command buffer that would close a cycle of handlers'
///   waits (`Cycle`), or leave the device's handler threads none free (`HandlerThreadCap`, the
///   runtime's cap, past the 256 such waits measured on Metal), and any cause the runtime adds:
///   Metal's answer is not measured, so it dies, unless the process is exiting, when it parks
///   as an exiting process's wait does. Only these ask whether the process is exiting.
fn refused_wait(cause: rt::DeadlockCause, exiting: impl FnOnce() -> bool) -> RefusedWait {
    match cause {
        rt::DeadlockCause::OwnQueue | rt::DeadlockCause::Exiting => RefusedWait::Hangs,
        _ if exiting() => RefusedWait::Hangs,
        _ => RefusedWait::Unmeasured,
    }
}

/// `id<MTLCommandBuffer>`.
pub type VulkanCommandBuffer = Obj<CommandBufferObject>;

/// The code Metal's GPU fails a command buffer with where an encoder was made under protection
/// `options`, not 0: [`INVALID_INPUT`] where a bit from 0x10 up is set, else
/// [`PROTECTION_VIOLATION`] (measured on the M3 Max over every option below 0x10 but 1, 3 and 7,
/// and 0x10 to 0x100, 0x11, 0x13, `u32::MAX`, `1 << 63`, `u64::MAX - 1` and `u64::MAX`). 0 for
/// none.
fn protection_code(options: u64) -> u64 {
    match options {
        0 => 0,
        o if o >= 0x10 => INVALID_INPUT,
        _ => PROTECTION_VIOLATION,
    }
}

impl CommandQueue<VulkanPlatform> for VulkanCommandQueue {
    /// Blocks while the queue holds 64 uncompleted command buffers, as Metal's does, and hangs
    /// where none can complete first ([`waiting`]).
    fn command_buffer(&self, _on: OnFault) -> RaiseResult<Option<VulkanCommandBuffer>> {
        let raw = waiting("commandBuffer", || self.raw.command_buffer())?;
        let predication = Predication::new(&self.device);
        let tiler = Arc::default();
        get("commandBuffer", || {
            raw.add_submission_handler(tiler_handler(&tiler))
        });
        let command_buffer = Obj::new(CommandBufferObject {
            raw,
            device: self.device.clone_ref(),
            open_encoders: Arc::default(),
            progress: Mutex::new(None),
            predication,
            opened: Mutex::new(Some(Opened::new())),
            made: AtomicUsize::new(0),
            failing: Mutex::new(Vec::new()),
            protection: Mutex::new(Protection::default()),
            tiler,
            hangs: Arc::clone(&self.hangs),
            hold: Mutex::new(None),
            executions: Mutex::new(Vec::new()),
        });
        // Its progress now, so its completed handler is the first: a change an encoder adds
        // (`BlitEncoder::at_completion`) then runs before every handler core adds, even one core
        // added before it encoded.
        command_buffer.progress("commandBuffer");
        Ok(Some(command_buffer))
    }

    fn set_label(&self, label: &str) {
        plain(|| self.raw.set_label(label));
    }
}

/// Why `cb` failed, as the runtime says: a host step's refusal dies at `commit`, whose commands
/// it recorded, naming it, as Metal runs the command the runtime refused; so does a draw's
/// pipeline the driver did not make ([`failed_pipeline`]), and every other failure that is none
/// of Metal's ([`answered`]).
fn failure(cb: &rt::CommandBuffer) -> Option<rt::CommandBufferError> {
    let e = cb.error()?;
    match e {
        rt::CommandBufferError::Refused => refused_step("commit", cb),
        rt::CommandBufferError::PipelineCreation(_) => failed_pipeline("commit", cb),
        e => answered("commit", e, Some(cb)),
    }
    Some(e)
}

/// Dies at `sel`, where a command buffer's failure `e` is read first, unless `e` is one of
/// Metal's failures, which core then sees as Metal's: a failure this platform committed the
/// command buffer with as Metal's GPU fails it ([`Failing`], a detached object's use), a lost
/// device's, memory's, an address fault, a render pass without attachments past 16384, a
/// detached buffer's use, a render pass without attachments whose `defaultRasterSampleCount`'s
/// low 8 bits are 16 to 127, each with Metal's code, and one the driver relayed with one of
/// Metal's codes ([`relayed`]). Those are the eight variants the runtime states as Metal's
/// (`rt::FailureOrigin::Metal`), each listed here, so that one the runtime adds later, whatever
/// origin it states, dies until it is reviewed ([`unanswered`]). Every other is the runtime's own
/// (`rt::FailureOrigin::Runtime`), where Metal fails no command buffer, and dies naming it: an
/// indirect draw's indices the index pool could not hold ([`pool_overflow`]), a sample through
/// a cubic sampler an argument buffer holds ([`cubic_argument_sampler`]), argument samplers the
/// runtime could not list ([`unlisted_argument_sampler`]), a host step's Vulkan object that
/// nothing made before it ([`host_step_object`]), a command's buffer rebound before the command
/// buffer was submitted ([`encoded_use_rebound`]), an emulated mesh draw past the mesh records pool
/// ([`mesh_records_overflow`]), a sparse mapping update the runtime applied none of
/// ([`sparse_heap_split`], [`sparse_mapping_arguments`], [`sparse_mapping_split`]), a Vulkan
/// failure of a submission ([`vulkan_failure`]), and any other ([`unanswered`]). A host step's
/// refusal and a draw's pipeline die before, where `cb` names them ([`refused_step`],
/// [`failed_pipeline`]). A relayed failure's death names Metal's description of it, which `cb`,
/// the command buffer that failed, keeps.
fn answered(sel: &'static str, e: rt::CommandBufferError, cb: Option<&rt::CommandBuffer>) {
    match e {
        rt::CommandBufferError::Injected(_)
        | rt::CommandBufferError::DeviceLost
        | rt::CommandBufferError::OutOfMemory
        | rt::CommandBufferError::PageFault
        | rt::CommandBufferError::RenderTargetSize
        | rt::CommandBufferError::DetachedBacking
        | rt::CommandBufferError::DefaultRasterSampleCount => {}
        rt::CommandBufferError::Execution(code) => {
            relayed(sel, code, cb.and_then(rt::CommandBuffer::error_description))
        }
        rt::CommandBufferError::IndexPoolOverflow => pool_overflow(sel),
        rt::CommandBufferError::CubicArgumentSampler => cubic_argument_sampler(sel),
        rt::CommandBufferError::UnlistedArgumentSampler => unlisted_argument_sampler(sel),
        rt::CommandBufferError::HostStepObject(object) => host_step_object(sel, object),
        rt::CommandBufferError::EncodedUseRebound => encoded_use_rebound(sel),
        rt::CommandBufferError::MeshRecordsOverflow(overflow) => {
            mesh_records_overflow(sel, overflow)
        }
        rt::CommandBufferError::SparseHeapSplit => sparse_heap_split(sel),
        rt::CommandBufferError::SparseMappingArguments(problem) => {
            sparse_mapping_arguments(sel, problem)
        }
        rt::CommandBufferError::SparseMappingSplit => sparse_mapping_split(sel),
        rt::CommandBufferError::Internal(code) => vulkan_failure(sel, code),
        e => unanswered(sel, e),
    }
}

/// The death at `sel` of a command buffer an emulated mesh draw failed, naming the first such
/// draw: its records did not fit the rest of the command buffer's mesh records pool, which is
/// the device's budget, or its vertices passed a `u32`, and the runtime drew nothing of it,
/// where Metal draws it and fails no command buffer.
fn mesh_records_overflow(sel: &'static str, overflow: rt::MeshRecordsOverflow) -> ! {
    unrepresentable(
        sel,
        &format!(
            "mesh draw {} of the command buffer ({} object and {} mesh threadgroups), whose \
             records passed its mesh records pool of {} bytes or whose vertices passed a u32, \
             which the runtime drew nothing of, failing the command buffer, where Metal draws it",
            overflow.draw,
            overflow.object_threadgroups,
            overflow.mesh_threadgroups,
            overflow.budget
        ),
    )
}

/// The death at `sel` of a command buffer whose sparse mapping filled its heap inside one of the
/// device's sparse units of several tiles, of a texture whose texels no copy writes (a
/// multisampled texture): the runtime binds such a texture's whole units, so it applied none of
/// the batch's mappings and ran nothing of the command buffer from there, where Metal maps the
/// tiles that fit and fails no command buffer.
fn sparse_heap_split(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a sparse mapping that filled its heap inside one of the device's sparse units of a \
         texture whose texels no copy writes, which the runtime applied nothing of the batch of \
         and ran nothing of the command buffer from, where Metal maps the tiles that fit",
    )
}

/// The death at `sel` of a command buffer whose indirect sparse mapping update's arguments, as
/// the host read them, are what Metal's validation stops (`problem`: a level, slice or region
/// past the texture's, or arguments past the buffer): the runtime applied none of the batch's
/// mappings and ran nothing of the command buffer from there, where what Metal maps without
/// validation is not measured.
fn sparse_mapping_arguments(sel: &'static str, problem: rt::SparseProblem) -> ! {
    unrepresentable(
        sel,
        &format!(
            "an indirect sparse mapping update whose arguments Metal's validation stops \
             ({problem:?}), which the runtime applied nothing of the batch of and ran nothing of \
             the command buffer from, where what Metal maps without validation is not measured"
        ),
    )
}

/// The death at `sel` of a command buffer whose indirect sparse mapping update's regions, as the
/// host read them, bind or unbind part of one of the device's sparse units of a texture whose
/// texels no copy writes: the runtime applied none of the batch's mappings and ran nothing of
/// the command buffer from there, where Metal maps them tile by tile and fails no command buffer.
fn sparse_mapping_split(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "an indirect sparse mapping update whose regions bind or unbind part of one of the \
         device's sparse units of a texture whose texels no copy writes, which the runtime \
         applied nothing of the batch of and ran nothing of the command buffer from, where Metal \
         maps them tile by tile",
    )
}

/// The death at `sel` of a command buffer whose submission Vulkan failed with `code`, other than
/// a lost device or memory (or that the runtime failed so for its own broken invariant, as
/// `VK_ERROR_UNKNOWN`): no failure of Metal's answers it.
fn vulkan_failure(sel: &'static str, code: rt::VkResultCode) -> ! {
    unrepresentable(
        sel,
        &format!(
            "a submission Vulkan failed with VkResult {}, which no failure of Metal's command \
             buffers answers",
            code.0
        ),
    )
}

/// The death at `sel` of a command buffer the runtime failed in a way nothing here matched to
/// a failure of Metal's: one the runtime adds after this platform was written, of either origin
/// it states, until it is reviewed here.
fn unanswered(sel: &'static str, e: rt::CommandBufferError) -> ! {
    unrepresentable(
        sel,
        &format!(
            "the runtime's failure of the command buffer ({e:?}), which nothing here matched to \
             a failure of Metal's"
        ),
    )
}

/// The death at `sel` of a command buffer a function failed by sampling, gathering or querying
/// through a sampler an argument buffer holds whose min or mag filter is cubic: the runtime
/// does not filter bicubically there, and fails the command buffer where its function reached
/// one, where Metal samples it and fails no command buffer.
fn cubic_argument_sampler(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a sample through a cubic sampler an argument buffer holds, which the runtime does not \
         filter bicubically there and fails the command buffer of, where Metal samples it",
    )
}

/// The death at `sel` of a command buffer the runtime failed as it committed, as a command's plan
/// could not list the sampler states live then (more than its set-1 sampler arrays or its
/// sampler tables hold, or one of a shape they refuse), or as a host step recorded a command
/// after the commit whose plan could not list them (more than its set-1 sampler arrays hold),
/// whether or not its functions reach the state left out;
/// Metal's functions reach every live state by its resource ID, and Metal fails no command
/// buffer so.
fn unlisted_argument_sampler(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a command whose argument samplers the runtime could not all list as its command buffer \
         committed or a host step recorded it, which it fails the command buffer of, where \
         Metal's functions reach every live sampler state by its ID",
    )
}

/// The death at `sel` of a command buffer a host step failed, as a command it recorded needed a
/// Vulkan object of `object`'s kind that nothing made before the step, which makes none (an
/// image, a render pass, a framebuffer, or a neutral binding of a slot nothing is bound to):
/// Metal draws the command and fails no command buffer.
fn host_step_object(sel: &'static str, object: rt::HostStepObject) -> ! {
    unrepresentable(
        sel,
        &format!(
            "a command a host step recorded, whose draw needed a Vulkan object nothing made \
             before the step ({object:?}), which the step makes none of and fails the command \
             buffer of, where Metal draws it"
        ),
    )
}

/// The death at `sel` of a command buffer the runtime failed as it was first submitted, as a
/// command of it named a buffer over the guest's pages that was rebound after the command was
/// encoded: the runtime resolved the use as the command was encoded, and ran nothing of each
/// submission holding such a command, where Metal resolves it as it submits the command buffer
/// and runs the command over the pages the buffer lies in then.
fn encoded_use_rebound(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a command naming a buffer over the guest's pages that was rebound after the command was \
         encoded and before its command buffer was submitted, whose use the runtime resolved as \
         it was encoded and ran nothing of, failing the command buffer, where Metal runs it over \
         the pages the buffer lies in as it submits",
    )
}

/// The death at `sel` of a command buffer an indexed indirect draw failed: the runtime keeps
/// such a draw's indices, checked or rewritten by a pass before the render pass, in the command
/// buffer's index pool, a sixty-fourth of the device's largest device-local heap, and draws
/// nothing of one past it, where Metal draws the draw and fails no command buffer.
fn pool_overflow(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "the indices of an indexed indirect draw past the command buffer's index pool, which the \
         runtime drew nothing of where Metal draws them",
    )
}

/// Dies at `sel` unless `code`, a failure of the GPU's work the driver relayed, is one of
/// Metal's codes ([`METAL_CODES`]), which core then sees as Metal's failure of the command
/// buffer. The death's line names Metal's `description` of the failure, where the driver's
/// message gave one ([`relayed_line`]).
fn relayed(sel: &'static str, code: u64, description: Option<String>) {
    if !METAL_CODES.contains(&code) {
        unrepresentable(sel, &relayed_line(code, description.as_deref()));
    }
}

/// The line of a relayed failure's death ([`relayed`]): its code, and Metal's `description` of
/// it where the driver's message gave one, not empty.
fn relayed_line(code: u64, description: Option<&str>) -> String {
    let line = format!(
        "a failure of the GPU's work the driver relayed with code {code}, which no failure of \
         Metal's command buffers has"
    );
    match description.filter(|d| !d.is_empty()) {
        Some(description) => format!("{line} (Metal's description: {description})"),
        None => line,
    }
}

/// The death at `sel` of `cb`, which failed as the driver did not make the `VkPipeline` of a draw
/// or a dispatch it recorded, naming the driver's error and the command, by its encoder's place
/// among the command buffer's ended encoders and its order there: the runtime makes a draw's or
/// a dispatch's pipeline while the command goes on, where Metal makes a state's pipelines as it
/// makes the state, and fails no command buffer so.
fn failed_pipeline(sel: &'static str, cb: &rt::CommandBuffer) -> ! {
    match cb.failed_pipeline() {
        Some(failed) => {
            let (command, encoder) = match failed.state {
                rt::FailedState::Compute(_) => ("dispatch", "compute"),
                _ => ("draw", "render"),
            };
            fatal_with_detail(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: sel,
                },
                &format!(
                    "{} (the pipeline of {command} {} of the command buffer's {encoder} encoder \
                     {})",
                    refusal(&failed.error),
                    failed.draw,
                    failed.encoder
                ),
            )
        }
        None => unrepresentable(
            sel,
            "a draw's or dispatch's pipeline the runtime did not keep, which failed",
        ),
    }
}

/// The death at `sel` of `cb`, which a host step refused, naming the refusal.
fn refused_step(sel: &'static str, cb: &rt::CommandBuffer) -> ! {
    match cb.refusal() {
        Some(refusal) => refused(sel, &refusal),
        None => unrepresentable(sel, "a host step's refusal the runtime did not keep"),
    }
}

/// The `NSError` of a failed command buffer: Metal's domain and code for the runtime's error,
/// and Metal's description of a use of a detached buffer or heap. A failure the driver relayed
/// (`Execution`, and the `PageFault` and `DeviceLost` the runtime maps codes 3 and 11 to) names
/// the runtime's error and, beside it, Metal's `relayed` description of the failure, as the
/// driver's message gave it (`rt::CommandBuffer::error_description`), where not empty. Core
/// prints the description in its log lines of a failed command buffer; nothing else reads it.
fn command_buffer_error(e: rt::CommandBufferError, relayed: Option<String>) -> NsErrorText {
    let description = match (e, relayed.filter(|d| !d.is_empty())) {
        (rt::CommandBufferError::Injected(backing::DETACHED_USE), _) => {
            "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".to_owned()
        }
        (e, Some(metal)) => {
            format!(
                "the Vulkan runtime's command buffer failed: {e:?} (Metal's description: {metal})"
            )
        }
        (e, None) => format!("the Vulkan runtime's command buffer failed: {e:?}"),
    };
    NsErrorText {
        domain: "MTLCommandBufferErrorDomain".to_owned(),
        code: i64::try_from(e.metal_code()).unwrap_or(i64::MAX),
        description,
    }
}

/// The `NSError` of `cb`, once it failed with one of Metal's failures ([`failure`],
/// [`command_buffer_error`]).
fn failure_error(cb: &rt::CommandBuffer) -> Option<NsErrorText> {
    let e = failure(cb)?;
    Some(command_buffer_error(e, cb.error_description()))
}

/// What a completed handler learns of `cb`: its `status` and `error` as they are now.
fn completed_info(cb: &rt::CommandBuffer) -> CompletedInfo {
    let error = failure_error(cb);
    CompletedInfo {
        status: CommandBufferStatus(cb.status().raw()),
        error,
    }
}

/// The runtime's form of an event core passes; a nil one dies at `sel`.
fn event<'a>(sel: &'static str, event: Option<EventRef<'a, VulkanPlatform>>) -> rt::EventRef<'a> {
    match event {
        Some(EventRef::Event(e)) => rt::EventRef::Event(e),
        Some(EventRef::Shared(e)) => rt::EventRef::Shared(e),
        None => unrepresentable(sel, "a nil event"),
    }
}

impl CommandBuffer<VulkanPlatform> for VulkanCommandBuffer {
    /// Refused once the buffer is committed, as Metal's validation refuses it.
    fn add_completed_handler(&self, _on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        #[cfg(test)]
        let slot = reims_vgpu_core::testkit::capture::current_slot();
        let handler: rt::Handler = Box::new(move |cb: &rt::CommandBuffer| {
            // A callout on the runtime's thread: marked ours, so a panic here aborts.
            let _env = BlockEnv::foreign();
            let run = || block(completed_info(cb));
            #[cfg(test)]
            reims_vgpu_core::testkit::capture::with_slot(slot, run);
            #[cfg(not(test))]
            run();
        });
        send("addCompletedHandler:", Policy::Dies, || {
            self.raw.add_completed_handler(handler)
        })
    }

    /// Refused while another encoder is open and after the commit, as Metal's validation
    /// refuses them. Its creation ends the exact `uniqueIdentifier` of later indirect command
    /// buffers ([`crate::gpu::icb::encoder_made`]), as every encoder creator's does.
    fn blit_encoder(&self, _on: OnFault) -> RaiseResult<Option<VulkanBlitEncoder>> {
        crate::gpu::icb::encoder_made();
        predication::boundary(&self.predication, "blitCommandEncoder");
        let raw = encoding("blitCommandEncoder", Policy::Dies, || {
            self.raw.blit_command_encoder()
        })?;
        Ok(Some(Obj::new(EncoderObject::new(raw, self))))
    }

    /// Refused while an encoder is open or after a commit, as Metal's validation refuses it.
    /// Every refusal dies naming the runtime's, and one of the records of an indirect command
    /// buffer it executes says so ([`committing`]). Its progress is committed past the send's
    /// check of the task's end, where a thread of an ended task stops with the command buffer
    /// never committed. One that Metal's GPU fails for what its encoders did is committed failed
    /// ([`CommandBufferObject::failure`]); any other is committed with its submission check,
    /// which fails a submission that uses a detached buffer or heap (module docs). A slow commit
    /// and a late completion are debug lines ([`timing::commit`]). One that holds work for
    /// Metal's tiler takes Metal's allocation for it as the runtime submits it
    /// ([`tiler_handler`]). One committed while a hang of its queue is in flight dies
    /// ([`Hangs`]); one that hangs Metal's GPU is in flight until [`HANG_TIME`] after the GPU
    /// reached its hang ([`hang_ends`]).
    fn commit(&self, _on: OnFault) -> RaiseResult<()> {
        predication::boundary(&self.predication, "commit");
        let progress = self.progress.lock().clone();
        let hold = plain(|| self.hold.lock().take());
        let committed = encoding("commit", Policy::Dies, || {
            self.hangs.committing();
            let opened = self.opened.lock().take();
            let failure = self.failure("commit", || {
                opened.as_ref().is_some_and(|opened| {
                    backing::uses_detached(|| self.raw.used_resources(), opened)
                })
            });
            self.check_executions(failure.as_ref().map(|f| &f.encoders), progress.as_deref());
            let check = match (&failure, opened) {
                (None, Some(opened)) => Some(backing::submission_check(opened.commit())),
                _ => None,
            };
            if let Some(progress) = &progress {
                progress.committing();
            }
            committing(|| {
                timing::commit(&self.raw, || match (failure, check) {
                    (Some(failure), _) => {
                        if hold.is_some() {
                            self.hangs.0.fetch_add(1, Ordering::SeqCst);
                        }
                        self.raw.commit_failed(failure)
                    }
                    (None, Some(check)) => self
                        .raw
                        .set_submission_check(check)
                        .and_then(|()| self.raw.commit()),
                    (None, None) => self.raw.commit(),
                })
            })
        });
        if let (Ok(()), Some(hold)) = (&committed, hold) {
            hang_ends("commit", hold, Arc::clone(&self.hangs));
        }
        committed
    }

    /// A serial encoder, refused as `blitCommandEncoder` is; under protection options 1, 3 and 7
    /// it dies ([`CommandBufferObject::protected_creation`]). A first use of Metal's device
    /// ([`FirstUse::ComputeEncoder`]).
    fn compute_encoder(&self, _on: OnFault) -> RaiseResult<Option<VulkanComputeEncoder>> {
        crate::gpu::icb::encoder_made();
        self.protected_creation("computeCommandEncoder");
        let raw = encoding("computeCommandEncoder", Policy::Dies, || {
            self.raw.compute_command_encoder()
        })?;
        metal_space::first_use(FirstUse::ComputeEncoder);
        Ok(Some(predication::encoder(raw, self)))
    }

    fn device(&self) -> Option<VulkanDevice> {
        Some(self.device.clone_ref())
    }

    fn encode_signal(
        &self,
        event: Option<EventRef<'_, VulkanPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeSignalEvent:value:";
        let event = self::event(sel, event);
        predication::boundary(&self.predication, sel);
        encoding(sel, Policy::Dies, || {
            self.raw.encode_signal_event(event, value)
        })
    }

    /// A scheduled handler that sets the event's value: refused after the commit, as every
    /// handler is.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&VulkanSharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeSignalEventScheduled:value:";
        let event = signal_event_scheduled
            .map(|e| rt::SharedEvent::clone(e))
            .unwrap_or_else(|| unrepresentable(sel, "a nil event"));
        let handler: rt::Handler = Box::new(move |_| event.set_signaled_value(value));
        encoding(sel, Policy::Dies, || {
            self.raw.add_scheduled_handler(handler)
        })
    }

    fn encode_wait(
        &self,
        event: Option<EventRef<'_, VulkanPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeWaitForEvent:value:";
        let event = self::event(sel, event);
        predication::boundary(&self.predication, sel);
        encoding(sel, Policy::Dies, || {
            self.raw.encode_wait_for_event(event, value)
        })
    }

    fn error(&self) -> Option<NsErrorText> {
        plain(|| failure_error(&self.raw))
    }

    /// An encoder of the pass, nil where Metal answers nil (`encode_render::open`).
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &VulkanRenderPassDescriptor,
    ) -> RaiseResult<Option<VulkanRenderEncoder>> {
        predication::boundary(&self.predication, "renderCommandEncoderWithDescriptor:");
        crate::gpu::encode_render::open(self, on, render_pass_descriptor)
    }

    /// Kept here, as the runtime has no protected content: the encoders made under them decide
    /// how the command buffer fails ([`CommandBufferObject::made_encoder`]). While an encoder is
    /// open it dies, as Metal asserts (`exec_protection_envelope_open_encoder`), and so do other
    /// options once an encoder was made under some, or after options from 0x10 up, which
    /// nothing measured.
    fn set_protection_options(&self, _on: OnFault, options: u64) -> RaiseResult<()> {
        let sel = "setProtectionOptions:";
        if self.open_encoders.load(Ordering::SeqCst) != 0 {
            uncommitted_encoder(sel);
        }
        send(sel, Policy::Dies, || {
            self.raw
                .set_protection_options(rt::DEFAULT_PROTECTION_OPTIONS)
        })?;
        let was = plain(|| *self.protection.lock());
        let kept = was.used || protection_code(was.options) == INVALID_INPUT;
        if kept && options != was.options {
            unrepresentable(
                sel,
                &format!(
                    "protection options {options:#x} after {:#x}, which nothing measured",
                    was.options
                ),
            );
        }
        plain(|| self.protection.lock().options = options);
        Ok(())
    }

    /// Scheduled or completed once its submission handler ran ([`tiler_handler`]).
    fn status(&self) -> CommandBufferStatus {
        let status = plain(|| {
            let status = self.raw.status();
            failure(&self.raw);
            status
        });
        CommandBufferStatus(status.raw())
    }

    /// Blocks. A wait the runtime knows would never return (a handler's for its own buffer)
    /// hangs, as Metal's does, and one whose answer on Metal is not measured dies ([`waiting`]).
    /// A command buffer a draw's pipeline failed, or a host step's refusal, or any failure
    /// that is none of Metal's, dies as the wait returns ([`failed_pipeline`],
    /// [`refused_step`], [`answered`]), where no completed handler read its failure first.
    fn wait_until_completed(&self, _on: OnFault) -> RaiseResult<()> {
        const SEL: &str = "waitUntilCompleted";
        waiting(SEL, || self.raw.wait_until_completed())?;
        match plain(|| self.raw.error()) {
            Some(rt::CommandBufferError::PipelineCreation(_)) => failed_pipeline(SEL, &self.raw),
            Some(rt::CommandBufferError::Refused) => refused_step(SEL, &self.raw),
            Some(e) => answered(SEL, e, Some(&self.raw)),
            None => {}
        }
        Ok(())
    }

    /// Blocks, and hangs or dies, as `wait_until_completed` does. It returns once the command
    /// buffer's submission handler ran ([`tiler_handler`]).
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        waiting("waitUntilScheduled", || self.raw.wait_until_scheduled())
    }

    /// The pass's dispatch type, serial when unset as Metal's descriptor's is. Metal makes an
    /// encoder of any type, and runs every type but concurrent as serial (measured, macOS 26.7,
    /// M3 Max): an encoder of type 2, 3, `0x80000000` or `0xffffffff` orders 8 dependent
    /// dispatches over 4M threads as a serial one does, where a concurrent one's race, and skips
    /// a nil entry of `useResources:count:usage:`, where a concurrent one ends the process. So
    /// such a type is serial here; a type past 32 bits, which the guest's record cannot hold
    /// and nothing measured, dies at `setDispatchType:`.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<VulkanComputeEncoder>> {
        crate::gpu::icb::encoder_made();
        let dispatch_type = match desc.dispatch_type {
            None => DispatchType::Serial,
            Some(raw) => match u32::try_from(raw) {
                Ok(raw) => DispatchType::try_from(raw).unwrap_or(DispatchType::Serial),
                Err(_) => unrepresentable(
                    "setDispatchType:",
                    &format!("dispatch type {raw:#x}, past 32 bits, which nothing measured"),
                ),
            },
        };
        self.protected_creation("computeCommandEncoderWithDescriptor:");
        let raw = encoding("computeCommandEncoderWithDescriptor:", Policy::Dies, || {
            self.raw
                .compute_command_encoder_with_dispatch_type(dispatch_type)
        })?;
        metal_space::first_use(FirstUse::ComputeEncoder);
        Ok(Some(predication::encoder(raw, self)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A fresh buffer of the queue's device
    fn buffer(queue: &VulkanCommandQueue) -> VulkanCommandBuffer {
        queue.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// commandBuffer makes a fresh buffer, not enqueued, of the queue's device; setLabel: sets
    /// the queue's label
    #[test]
    fn queue_command_buffer_and_label() {
        let q = dev().new_command_queue().unwrap();
        let (a, b) = (buffer(&q), buffer(&q));
        assert_ne!(a.id(), b.id());
        assert_eq!(a.status(), CommandBufferStatus::NOT_ENQUEUED);
        assert_eq!(a.device().unwrap().id(), q.device.id());
        q.set_label("reims-vgpu.exec");
        assert_eq!(q.raw.label().as_deref(), Some("reims-vgpu.exec"));
    }

    /// An empty buffer commits and completes: its completed handler sees Completed and no
    /// error, after the scheduled signal set its event; a handler added after the commit, and a
    /// second commit, die
    #[test]
    fn commit_completes_and_runs_handlers() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let scheduled = d.new_shared_event().unwrap();
        cb.encode_signal_when_scheduled(Some(&scheduled), 5)
            .unwrap();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(
            OnFault::Return,
            Box::new(move |info| tx.send(info).unwrap()),
        )
        .unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let info = rx.recv().unwrap();
        assert_eq!(info.status, CommandBufferStatus::COMPLETED);
        assert_eq!(info.error, None);
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(cb.error(), None);
        assert_eq!(scheduled.signaled_value(), 5);
        // A handler after the commit, and a second commit, die: Metal raises at neither.
        let late = catch_death(|| {
            let _ = cb.add_completed_handler(OnFault::Return, Box::new(|_| {}));
        });
        assert_eq!(late.unwrap_err().file, "addCompletedHandler:");
        for on in [OnFault::Return, OnFault::Die] {
            let site = catch_death(|| {
                let _ = cb.commit(on);
            })
            .unwrap_err();
            assert_eq!(site.file, "commit");
        }
    }

    /// Metal takes its own allocation for a command buffer's first work for its tiler as it
    /// submits that work (`vaprobe rendersubmit`): behind an event wait, once the wait is met,
    /// not at the commit
    #[test]
    fn tiler_work_behind_a_wait_is_taken_once_the_wait_is_met() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work(FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
    }

    /// With no event wait and nothing ahead of it on its queue, a command buffer's first work
    /// for the tiler is taken as the runtime submits it, after its commit and before it reads
    /// scheduled, as Metal submits it on a thread of its own: a blit's work, and a render
    /// encoder's made through the platform's `renderCommandEncoderWithDescriptor:` (a render pass
    /// that clears and stores a texture)
    #[test]
    fn tiler_work_with_no_wait_is_taken_as_it_is_submitted() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        cb.tiler_work(FirstUse::TilerBlit);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();

        // A render encoder's work, from the platform's creation, is the command buffer's, which
        // its submission takes too (the process's first use was the blit's, above).
        let p = crate::platform::tests::platform();
        let target = d
            .new_texture(
                OnFault::Die,
                &p.texture_descriptor(&TextureDesc {
                    pixel_format: 80,
                    width: 16,
                    height: 16,
                    usage: 4,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                }),
            )
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(2),
            store_action: Some(1),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let cb = buffer(&q);
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        assert!(metal_space::process().took(FirstUse::RenderEncoder));
        assert!(cb.tiler.load(Ordering::SeqCst));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// A wait met before the commit holds nothing back: the runtime submits the command buffer
    /// at once, and its work for the tiler is taken before it reads scheduled
    #[test]
    fn tiler_work_behind_a_met_wait_is_taken_as_it_is_submitted() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        event.set_signaled_value(1).unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work(FirstUse::TilerBlit);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// An event wait after the work holds the command buffer's submission back until the wait
    /// is met, and its work for the tiler with it
    #[test]
    fn tiler_work_before_a_wait_is_taken_at_the_submission() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.tiler_work(FirstUse::TilerBlit);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
    }

    /// A command buffer with no wait of its own, committed behind one on its queue whose wait
    /// holds it, takes its work for the tiler as the runtime submits it behind the earlier one
    #[test]
    fn tiler_work_behind_a_waiting_command_buffer_is_taken_at_its_submission() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let first = buffer(&q);
        first
            .encode_wait(Some(EventRef::Shared(&event)), 1)
            .unwrap();
        first.commit(OnFault::Return).unwrap();
        let cb = buffer(&q);
        cb.tiler_work(FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        first.wait_until_completed(OnFault::Return).unwrap();
    }

    /// No call finds a command buffer scheduled before its work for the tiler is taken: `status`
    /// answers committed while its wait holds it, and scheduled or completed only with the
    /// allocation taken, as the runtime runs the submission handler first
    #[test]
    fn tiler_work_is_taken_before_the_command_buffer_reads_scheduled() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work(FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMMITTED);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        // At most 10 s: the event's value is met, so the runtime submits it at once.
        let mut polls = 0;
        while cb.status() == CommandBufferStatus::COMMITTED {
            polls += 1;
            assert!(
                polls < 10_000,
                "the command buffer was not scheduled within 10 s"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// A buffer signals an event when it completes, and a later buffer waits for it; both
    /// kinds of event are taken, and a nil event dies
    #[test]
    fn events_signal_and_wait() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let shared = d.new_shared_event().unwrap();
        let event = d.new_event(OnFault::Return).unwrap().unwrap();
        let first = buffer(&q);
        first
            .encode_signal(Some(EventRef::Shared(&shared)), 1)
            .unwrap();
        first
            .encode_signal(Some(EventRef::Event(&event)), 1)
            .unwrap();
        let second = buffer(&q);
        second
            .encode_wait(Some(EventRef::Event(&event)), 1)
            .unwrap();
        first.commit(OnFault::Return).unwrap();
        second.commit(OnFault::Return).unwrap();
        second.wait_until_completed(OnFault::Return).unwrap();
        assert!(shared.wait_until_signaled_value(1, 1000));
        let cb = buffer(&q);
        let site = catch_death(|| cb.encode_wait(None, 1)).unwrap_err();
        assert_eq!(site.file, "encodeWaitForEvent:value:");
    }

    /// A completed handler's wait for its own command buffer, which Metal never returns from,
    /// hangs: the handler neither returns nor dies
    #[test]
    fn a_handlers_wait_for_its_own_buffer_hangs() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        let own = cb.clone_ref();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(
            OnFault::Return,
            Box::new(move |_| {
                tx.send("waiting").unwrap();
                let waited = own.wait_until_completed(OnFault::Return);
                tx.send(if waited.is_ok() { "returned" } else { "raised" })
                    .unwrap();
            }),
        )
        .unwrap();
        cb.commit(OnFault::Return).unwrap();
        assert_eq!(rx.recv().unwrap(), "waiting");
        // Still blocked: no answer, and the handler still holds its sender (a death would have
        // dropped it).
        let after = rx.recv_timeout(std::time::Duration::from_millis(300));
        assert_eq!(after, Err(mpsc::RecvTimeoutError::Timeout));
    }

    /// A wait the runtime refuses hangs where Metal's never returns, a handler's wait in its own
    /// queue, and any once the process is exiting, and dies for the causes where Metal's answer
    /// is not measured while it is not; only those ask whether the process is exiting. A live
    /// process is not exiting
    #[test]
    fn refused_waits_hang_only_where_metal_does() {
        use rt::DeadlockCause::{Cycle, Exiting, HandlerThreadCap, OwnQueue};
        let asked = &std::cell::Cell::new(0);
        let exiting = |answer| {
            move || {
                asked.set(asked.get() + 1);
                answer
            }
        };
        for answer in [false, true] {
            for cause in [OwnQueue, Exiting] {
                assert_eq!(refused_wait(cause, exiting(answer)), RefusedWait::Hangs);
            }
        }
        assert_eq!(asked.get(), 0);
        for cause in [Cycle, HandlerThreadCap] {
            assert_eq!(refused_wait(cause, exiting(false)), RefusedWait::Unmeasured);
            assert_eq!(refused_wait(cause, exiting(true)), RefusedWait::Hangs);
        }
        assert_eq!(asked.get(), 4);
        assert!(!rt::process_exiting());
    }

    /// Two queues' completed handlers that wait for each other's command buffer close a cycle the
    /// runtime refuses: the wait dies at its selector, as Metal's answer there is not measured
    #[test]
    fn a_cycle_of_handlers_waits_dies() {
        use reims_vgpu_core::raise::death::DeathSite;
        use reims_vgpu_core::testkit::capture::expect_death;

        let site = DeathSite {
            kind: DeathKind::Abort,
            file: "waitUntilCompleted",
            line: 0,
        };
        // The process's device, made before `expect_death`'s clock starts, which times the cycle
        // alone: making it compiles what the runtime's shared caches hold, which can take longer
        // than the clock allows.
        let d = dev();
        expect_death(site, move || {
            let (qa, qb) = (
                d.new_command_queue().unwrap(),
                d.new_command_queue().unwrap(),
            );
            let event = d.new_shared_event().unwrap();
            let (a, b) = (buffer(&qa), buffer(&qb));
            b.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            let (to_a, to_b) = (a.clone_ref(), b.clone_ref());
            let (tx, rx) = mpsc::channel();
            a.add_completed_handler(
                OnFault::Return,
                Box::new(move |_| {
                    tx.send(()).unwrap();
                    let _ = to_b.wait_until_completed(OnFault::Return);
                }),
            )
            .unwrap();
            b.add_completed_handler(
                OnFault::Return,
                Box::new(move |_| {
                    let _ = to_a.wait_until_completed(OnFault::Return);
                }),
            )
            .unwrap();
            a.commit(OnFault::Return).unwrap();
            b.commit(OnFault::Return).unwrap();
            rx.recv().unwrap();
            event.set_signaled_value(1).unwrap();
            a.wait_until_completed(OnFault::Return).unwrap();
        });
    }

    /// Runs `body` in a child copy of this test binary, its deaths ending it as in production,
    /// and answers how the child ended: what an exit does is the process's, which only a child
    /// can show.
    fn in_child(test: &str, body: impl FnOnce()) -> std::process::ExitStatus {
        if std::env::var("RVG_CHILD").as_deref() == Ok(test) {
            reims_vgpu_core::testkit::capture::set_capture_enabled(false);
            body();
            std::process::exit(0);
        }
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("RVG_CHILD", test)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
    }

    /// An exit while a completed handler waits for another queue's command buffer, which the
    /// runtime refuses once the process is exiting, leaves the handler parked, as Metal's wait
    /// never returns there: the process exits as asked, and dies at nothing
    #[test]
    fn an_exit_parks_a_handlers_wait_for_another_queue() {
        /// An exit handler, registered before the runtime's, that keeps the process past the
        /// runtime's refusal of the blocked wait (within 50 ms of its exit handler).
        extern "C" fn linger() {
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        unsafe extern "C" {
            fn atexit(f: extern "C" fn()) -> std::ffi::c_int;
        }
        let status = in_child(
            "gpu::command::tests::an_exit_parks_a_handlers_wait_for_another_queue",
            || {
                // SAFETY: `linger` takes nothing and never unwinds.
                assert_eq!(unsafe { atexit(linger) }, 0);
                let d = dev();
                let (exec_queue, paging_queue) = (
                    d.new_command_queue().unwrap(),
                    d.new_command_queue().unwrap(),
                );
                let never = d.new_shared_event().unwrap();
                let exec = buffer(&exec_queue);
                exec.encode_wait(Some(EventRef::Shared(&never)), 1).unwrap();
                exec.commit(OnFault::Return).unwrap();
                let paging = buffer(&paging_queue);
                let waited = exec.clone_ref();
                let (tx, rx) = mpsc::channel();
                paging
                    .add_completed_handler(
                        OnFault::Return,
                        Box::new(move |_| {
                            tx.send(()).unwrap();
                            let _ = waited.wait_until_completed(OnFault::Die);
                        }),
                    )
                    .unwrap();
                paging.commit(OnFault::Return).unwrap();
                rx.recv().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
            },
        );
        assert_eq!(status.code(), Some(0), "{status:?}");
    }

    /// A host step's refusal of a command buffer that executes an indirect command buffer dies
    /// as the command buffer ends, at the execution's selector, outside isolation and while the
    /// command buffer's task runs; once the task ended, as core's handlers of it run none, it
    /// dies at nothing, and the command buffer's progress ends all the same
    #[test]
    fn a_refused_command_buffer_of_an_ended_task_dies_at_nothing() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let d = dev();
        let desc = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        let icb = d.new_indirect_command_buffer(&desc, 1, 0).unwrap().unwrap();
        let sel = "executeCommandsInBuffer:withRange:";
        let ending = |task| {
            let progress = Arc::new(Progress::default());
            progress.executes(sel, icb.clone_ref());
            let ending = Ending {
                progress: Some(Arc::clone(&progress)),
                task,
                slot: None,
            };
            (progress, ending)
        };
        let refused = |sel| {
            unrepresentable(sel, "a host step's refusal");
        };
        let life = TaskLife::new();
        for task in [None, Some(Arc::clone(&life))] {
            let (_, mut e) = ending(task);
            let site = catch_death(|| e.end(Stage::Failed, Some(refused))).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        }
        let (progress, mut e) = ending(Some(Arc::clone(&life)));
        assert!(life.exit());
        e.end(Stage::Failed, Some(refused));
        assert_eq!(progress.stage(), Stage::Failed);
    }

    /// A thread of an isolated task that ended stops at commit before the runtime's commit, and
    /// the command buffer's progress stays uncommitted: what it uses is not held as the GPU's
    #[test]
    fn an_ended_tasks_commit_leaves_its_command_buffer_uncommitted() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        let progress = cb.progress("test");
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            cb.commit(OnFault::Return)
        });
        assert!(stopped.is_none());
        assert_eq!(progress.stage(), Stage::Encoding);
        assert_eq!(cb.status(), CommandBufferStatus::NOT_ENQUEUED);
    }

    /// A failure of the GPU's work the driver relayed reaches core with Metal's code where its
    /// code is one of Metal's, and dies where it is none
    #[test]
    fn relayed_failures_take_metals_codes_or_die() {
        for code in METAL_CODES {
            relayed("commit", code, None);
            let e = command_buffer_error(rt::CommandBufferError::Execution(code), None);
            assert_eq!(
                (e.domain.as_str(), e.code),
                ("MTLCommandBufferErrorDomain", i64::try_from(code).unwrap())
            );
        }
        for code in [0, 5, 6, 13, 15, 17, u64::MAX] {
            let site = catch_death(|| relayed("waitUntilCompleted", code, Some("x".to_owned())))
                .unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "waitUntilCompleted"),
                "{code}"
            );
        }
    }

    /// A relayed failure's description and death line carry Metal's description beside the
    /// runtime's error, as the driver's message gave it (MoltenVK 1.4.1's forms), and the error
    /// alone where the message gave none; a detached object's use keeps Metal's own description
    #[test]
    fn relayed_failures_name_metals_description() {
        use rt::CommandBufferError as E;
        let metal = "Discarded (victim of GPU error/recovery)";
        let e = command_buffer_error(E::Execution(9), Some(metal.to_owned()));
        assert_eq!(
            (e.code, e.description.as_str()),
            (
                9,
                "the Vulkan runtime's command buffer failed: Execution(9) (Metal's description: \
                 Discarded (victim of GPU error/recovery))"
            )
        );
        let fault =
            command_buffer_error(E::PageFault, Some("Caused GPU Address Fault Error".into()));
        assert_eq!(
            (fault.code, fault.description.as_str()),
            (
                3,
                "the Vulkan runtime's command buffer failed: PageFault (Metal's description: \
                 Caused GPU Address Fault Error)"
            )
        );
        for none in [None, Some(String::new())] {
            let e = command_buffer_error(E::Execution(9), none);
            assert_eq!(
                e.description,
                "the Vulkan runtime's command buffer failed: Execution(9)"
            );
        }
        let detached = command_buffer_error(E::Injected(backing::DETACHED_USE), Some(metal.into()));
        assert_eq!(
            detached.description,
            "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        );
        assert!(relayed_line(13, Some(metal)).ends_with(
            "with code 13, which no failure of Metal's command buffers has (Metal's description: \
             Discarded (victim of GPU error/recovery))"
        ));
        assert!(
            relayed_line(13, Some("")).ends_with("which no failure of Metal's command buffers has")
        );
    }

    /// A command buffer's failure that is one of Metal's reaches core; every failure of the
    /// runtime's own dies at the selector that reads it first, at the commit's and at the wait,
    /// its line naming it: a sparse mapping update the runtime applied none of, a Vulkan failure
    /// of a submission, one nothing here matched to Metal's, and an indirect draw's indices past
    /// the index pool. The eight this platform lists as Metal's are those the runtime states as
    /// Metal's, and the others it names its own: a change of either at a new pin fails here, for
    /// review
    #[test]
    fn failures_metal_has_none_of_die_named() {
        use rt::CommandBufferError as E;
        let metals = [
            E::Injected(8),
            E::DeviceLost,
            E::OutOfMemory,
            E::PageFault,
            E::RenderTargetSize,
            E::DetachedBacking,
            E::DefaultRasterSampleCount,
            E::Execution(1),
        ];
        for e in metals {
            assert_eq!(e.origin(), rt::FailureOrigin::Metal, "{e:?}");
            answered("commit", e, None);
        }
        let named = [
            (
                E::SparseHeapSplit,
                "filled its heap inside one of the device's sparse units",
            ),
            (
                E::SparseMappingArguments(rt::SparseProblem::Lengths),
                "Metal's validation stops (Lengths)",
            ),
            (E::SparseMappingSplit, "regions bind or unbind part of one"),
            (E::Internal(rt::VkResultCode(-13)), "VkResult -13"),
            (E::IndexPoolOverflow, "index pool"),
        ];
        for (e, _) in named {
            assert_eq!(e.origin(), rt::FailureOrigin::Runtime, "{e:?}");
        }
        for sel in ["commit", "waitUntilCompleted"] {
            for (e, names) in named {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(|| answered(sel, e, None)).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                assert!(line.contains(names), "{e:?}: {line}");
            }
        }
    }

    /// A refusal of a commit dies at the commit naming the runtime's refusal; only one the
    /// runtime names an execution's, of the records of an indirect command buffer, says so
    #[test]
    fn a_commit_blames_the_records_only_for_their_refusal() {
        use rt::{Invalid, Problem, RuntimeGap};

        let records = |call| {
            rt::Error::Invalid(Invalid::new(
                call,
                Problem::IndirectCommandBuffer(rt::IndirectCommandBufferProblem::UnstridedBuffer {
                    index: 0,
                }),
            ))
        };
        for call in [
            "execute_commands_in_buffer",
            "execute_commands_in_buffer_indirect",
        ] {
            assert!(of_records(&records(call)));
            let site = catch_death(|| committing(|| Err(records(call)))).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
        }
        for other in [
            rt::Error::Invalid(Invalid::new("commit", Problem::AlreadyCommitted)),
            rt::Error::NotImplemented(RuntimeGap::SparseHeaps),
            rt::Error::DeviceLost,
        ] {
            assert!(!of_records(&other));
            assert_eq!(committing(|| Err(other.clone())), Err(other));
        }
    }

    /// A Shared buffer of 16 zero bytes, and a blit encoder of `cb` that fills it with `value`
    /// and ends.
    fn filled(cb: &VulkanCommandBuffer, value: u8) -> crate::gpu::resource::VulkanBuffer {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::types::NsRange;
        let b = crate::gpu::resource::tests::shared(16);
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let range = NsRange::new(0, 16);
        let fill = BlitCmd::FillBuffer {
            buffer: &b,
            range,
            value,
        };
        blit.encode(OnFault::Return, fill).unwrap();
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        b
    }

    /// The 16 bytes of a Shared buffer.
    fn sixteen(b: &crate::gpu::resource::VulkanBuffer) -> Vec<u8> {
        use reims_vgpu_core::platform::resource::Buffer as _;
        // SAFETY: a Shared buffer of 16 bytes, read after its command buffer completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), 16) }.to_vec()
    }

    /// `cb` committed and completed: its status and its error's code.
    fn completed(cb: &VulkanCommandBuffer) -> (u64, Option<i64>) {
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        (cb.status().0, cb.error().map(|e| e.code))
    }

    /// Protection options fail a command buffer as Metal's GPU does: an encoder made under
    /// options below 0x10 runs nothing and fails it with code 16, one made before them runs,
    /// and none of its event signals fires; options from 0x10 up fail it with code 1, with an
    /// encoder or without; options below 0x10 with no encoder under them, or 0 again before
    /// one, complete it
    #[test]
    fn protection_options_fail_command_buffers_as_metals_gpu_does() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let event = d.new_shared_event().unwrap();
        let before = filled(&cb, 0x5a);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        let under = filled(&cb, 0x6b);
        let signal = Some(EventRef::Shared(&event));
        cb.encode_signal(signal, 7).unwrap();
        assert_eq!(completed(&cb), (5, Some(16)));
        assert_eq!(
            (sixteen(&before), sixteen(&under)),
            (vec![0x5a; 16], vec![0; 16])
        );
        assert_eq!(event.signaled_value(), 0);
        for options in [0x10, u64::MAX] {
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            assert_eq!(completed(&cb), (5, Some(1)), "{options:#x}, no encoder");
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            let under = filled(&cb, 0x6b);
            assert_eq!(completed(&cb), (5, Some(1)), "{options:#x}");
            assert_eq!(sixteen(&under), vec![0; 16]);
        }
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        assert_eq!(completed(&cb), (4, None), "options 2, no encoder");
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        cb.set_protection_options(OnFault::Return, 0).unwrap();
        let after = filled(&cb, 0x7c);
        assert_eq!(completed(&cb), (4, None), "options 2, then 0");
        assert_eq!(sixteen(&after), vec![0x7c; 16]);
    }

    /// Under protection options 1, 3 and 7 a compute or render encoder's creation dies, and so
    /// does a blit encoder's first command, as Metal ends the process there; with no encoder the
    /// command buffer completes. Other options once an encoder was made under some, or after
    /// options from 0x10 up, die
    #[test]
    fn protection_options_that_end_the_process_die() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        let q = dev().new_command_queue().unwrap();
        for options in [1, 3, 7] {
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            let site = catch_death(|| cb.compute_encoder(OnFault::Return)).unwrap_err();
            assert_eq!(site.file, "computeCommandEncoder");
            let pass = crate::platform::tests::platform().render_pass_descriptor();
            let site = catch_death(|| cb.render_encoder(OnFault::Return, &pass)).unwrap_err();
            assert_eq!(site.file, "renderCommandEncoderWithDescriptor:");
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            let end = || blit.encode(OnFault::Return, BlitCmd::EndEncoding);
            assert_eq!(catch_death(end).unwrap_err().file, "endEncoding");
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            assert_eq!(completed(&cb), (4, None), "options {options}, no encoder");
        }
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        filled(&cb, 0x11);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        let site = catch_death(|| cb.set_protection_options(OnFault::Return, 0)).unwrap_err();
        assert_eq!(site.file, "setProtectionOptions:");
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 0x10).unwrap();
        cb.set_protection_options(OnFault::Return, 0x10).unwrap();
        let site = catch_death(|| cb.set_protection_options(OnFault::Return, 0)).unwrap_err();
        assert_eq!(site.file, "setProtectionOptions:");
    }

    /// A command buffer Metal's GPU fails in one way fails so at its commit, from the first
    /// encoder that fails, and one that only uses a detached object is left to its submissions;
    /// one it would fail in two, a use of a detached object and a hang, dies at its commit, as
    /// nothing measured it
    #[test]
    fn a_command_buffer_failing_two_ways_dies_at_its_commit() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        assert!(
            cb.failure("commit", || true).is_none(),
            "a detached use alone"
        );
        for from in [3, 1] {
            cb.fail(Failing {
                code: HANG,
                from,
                signals: true,
            });
        }
        let failure = cb.failure("commit", || false).unwrap();
        assert_eq!(
            (failure.code, failure.encoders, failure.signals),
            (HANG, rt::FailedEncoders::From(1), true)
        );
        let site = catch_death(|| cb.failure("commit", || true)).unwrap_err();
        assert_eq!(site.file, "commit");
    }

    /// A command buffer a host step failed for a Vulkan object nothing made before it
    /// (`CommandBufferError::HostStepObject`) dies at the selector where its failure is read, at
    /// the commit's (a completed handler's, its status or error) and at the wait, its line naming
    /// the object's kind and that Metal draws the command
    #[test]
    fn a_host_steps_object_dies_naming_its_kind() {
        use rt::HostStepObject as O;
        for sel in ["commit", "waitUntilCompleted"] {
            for object in [O::Neutral, O::Image, O::RenderPass, O::Framebuffer] {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(|| {
                    host_step_object(sel, object);
                })
                .unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                let named = format!("nothing made before the step ({object:?})");
                assert!(line.contains(&named), "{line}");
                assert!(line.ends_with("where Metal draws it"), "{line}");
            }
        }
    }

    /// A command buffer the runtime failed for a command's buffer rebound before its first
    /// submission (`CommandBufferError::EncodedUseRebound`) dies at the selector where its
    /// failure is read, at the commit's (a completed handler's, its status or error) and at the
    /// wait, its line naming the rebind and where Metal runs the command
    #[test]
    fn an_encoded_use_rebound_dies_naming_the_rebind() {
        for sel in ["commit", "waitUntilCompleted"] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| {
                encoded_use_rebound(sel);
            })
            .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains("rebound after the command was encoded"),
                "{line}"
            );
            assert!(line.ends_with("the buffer lies in as it submits"), "{line}");
        }
    }

    /// While an encoder is open, what Metal asserts at dies whatever the handler: another
    /// encoder, an event, setProtectionOptions:, commit; once it ends, each is taken
    #[test]
    fn an_open_encoder_kills_what_metal_asserts_at() {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let event = d.new_shared_event().unwrap();
        let dies = |sel: &str, call: &dyn Fn()| {
            let site = catch_death(call).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        };
        dies("computeCommandEncoder", &|| {
            let _ = cb.compute_encoder(OnFault::Return);
        });
        dies("encodeSignalEvent:value:", &|| {
            let _ = cb.encode_signal(Some(EventRef::Shared(&event)), 1);
        });
        dies("setProtectionOptions:", &|| {
            let _ = cb.set_protection_options(OnFault::Return, 0);
        });
        dies("commit", &|| {
            let _ = cb.commit(OnFault::Return);
        });
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.set_protection_options(OnFault::Return, 0).unwrap();
        cb.encode_signal(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // An encoder that goes without endEncoding is open no more.
        let cb = buffer(&q);
        drop(cb.blit_encoder(OnFault::Return).unwrap().unwrap());
        cb.set_protection_options(OnFault::Return, 0).unwrap();
    }
}

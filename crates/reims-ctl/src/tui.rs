//! The interactive interface: the VMs, their details, and what can be done with them.
//!
//! Slow work (starting, stopping, creating, a bug report) runs on a thread of its own, which
//! reports its steps; the list shows each VM's step until it ends. Each VM's state is read
//! again every second, so VMs started or stopped elsewhere (the CLI, a printed script, closing
//! a VM's window) show as they are.

use crate::cli::{self, delete_vm, describe, set_installer};
use crate::create::{self, DiskSource, ImportMode, Spec};
use crate::form::{self, Field, Form, opts};
use crate::gpu::GpuInfo;
use crate::host::{Host, Os, Profile};
use crate::update::Channel;
use crate::vm::{Registry, Vm, format_mib, parse_resolution, parse_size_mib};
use crate::{install, logs, paths, reims_util, runtime, update};
use anyhow::{Context, Result, bail};
use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState, Wrap};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

pub fn run() -> Result<()> {
    let host = Host::current();
    let mut app = App::new(host)?;
    let mut terminal = ratatui::init();
    let result = app.run(&mut terminal);
    ratatui::restore();
    result?;
    if app.update_requested {
        return run_update(app.update_channel);
    }
    Ok(())
}

/// Runs the update in the plain terminal, then the new reims-ctl in this process's place.
fn run_update(channel: Option<Channel>) -> Result<()> {
    println!("Updating reims-ctl and its QEMU…\n");
    update::update(channel)?;
    println!("\nUpdated. Starting the new reims-ctl…");
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe()?;
    let err = std::process::Command::new(&exe).exec();
    Err(err).with_context(|| format!("run {}", exe.display()))
}

struct VmRow {
    name: String,
    dir: PathBuf,
    vm: Result<Vm, String>,
    pid: Option<i32>,
}

enum Msg {
    Progress(String, String),
    Done(String, Result<String, String>),
    Gpus(Vec<GpuInfo>),
}

enum Overlay {
    Form(FormKind, Form),
    Text {
        title: String,
        lines: Vec<String>,
        scroll: usize,
        kind: TextKind,
    },
    Confirm {
        title: String,
        body: String,
        options: Vec<(&'static str, Action)>,
        choice: usize,
    },
}

#[derive(Clone)]
enum FormKind {
    Create,
    Edit(String),
}

#[derive(Clone)]
enum TextKind {
    Command { vm: String, script: String },
    Log { vm: String, serial: bool },
    Help,
}

#[derive(Clone)]
enum Action {
    None,
    Delete { vm: String, keep_files: bool },
    ForceStop(String),
    Update(Option<Channel>),
    Quit,
}

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Info,
    Ok,
    Warn,
    Error,
}

struct App {
    host: Host,
    rows: Vec<VmRow>,
    table: TableState,
    overlay: Option<Overlay>,
    status: Option<(String, Level)>,
    activity: BTreeMap<String, String>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    gpus: Option<Vec<GpuInfo>>,
    qemu: Option<String>,
    last_state: Instant,
    last_registry: Instant,
    quit: bool,
    update_requested: bool,
    update_channel: Option<Channel>,
    pending_select: Option<String>,
    /// A VM's logs for the run loop to show in less, which needs the terminal.
    pending_page: Option<(String, Vec<PathBuf>)>,
    /// A VM for the run loop to show in the terminal viewer, which needs the terminal.
    pending_view: Option<Vm>,
    /// The macOS versions reims-util downloads, read when the first x86-64 VM's form opens.
    macos_versions: Option<Result<Vec<reims_util::Version>, String>>,
}

impl App {
    fn new(host: Host) -> Result<App> {
        let (tx, rx) = channel();
        let qemu = host
            .default_profile()
            .and_then(|p| install::qemu_binary(p).ok())
            .map(|q| {
                install::qemu_version(&q)
                    .map(|v| v.replace("QEMU emulator version ", "QEMU "))
                    .unwrap_or_else(|| paths::tilde(&q))
            });
        let mut app = App {
            host,
            rows: Vec::new(),
            table: TableState::default(),
            overlay: None,
            status: None,
            activity: BTreeMap::new(),
            tx,
            rx,
            gpus: None,
            qemu,
            last_state: Instant::now(),
            last_registry: Instant::now(),
            quit: false,
            update_requested: false,
            update_channel: None,
            pending_select: None,
            pending_page: None,
            pending_view: None,
            macos_versions: None,
        };
        app.reload()?;
        if !app.rows.is_empty() {
            app.table.select(Some(0));
        }
        if host.os == Os::Linux {
            let tx = app.tx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(Msg::Gpus(crate::gpu::list()));
            });
        }
        if host.default_profile().is_none() {
            app.say(
                &format!("reims-ctl runs no VMs on {host} yet"),
                Level::Error,
            );
        } else if app.qemu.is_none() {
            app.say(
                "QEMU is not installed: run scripts/install-ctl.sh in your reims-vgpu checkout",
                Level::Warn,
            );
        }
        Ok(app)
    }

    fn say(&mut self, s: &str, level: Level) {
        self.status = Some((s.to_string(), level));
    }

    fn reload(&mut self) -> Result<()> {
        let reg = Registry::load()?;
        let selected = self.selected_name();
        self.rows = reg
            .vms
            .iter()
            .map(|e| {
                let vm = Vm::load(&e.dir).map_err(|err| format!("{err:#}"));
                let pid = vm.as_ref().ok().and_then(runtime::pid);
                VmRow {
                    name: e.name.clone(),
                    dir: e.dir.clone(),
                    vm,
                    pid,
                }
            })
            .collect();
        let idx = selected.and_then(|n| self.rows.iter().position(|r| r.name == n));
        self.table.select(match (idx, self.rows.is_empty()) {
            (_, true) => None,
            (Some(i), _) => Some(i),
            (None, _) => Some(self.table.selected().unwrap_or(0).min(self.rows.len() - 1)),
        });
        self.last_registry = Instant::now();
        self.last_state = Instant::now();
        Ok(())
    }

    fn refresh_state(&mut self) {
        for r in &mut self.rows {
            r.pid = r.vm.as_ref().ok().and_then(runtime::pid);
        }
        self.last_state = Instant::now();
    }

    fn selected_name(&self) -> Option<String> {
        self.table
            .selected()
            .and_then(|i| self.rows.get(i))
            .map(|r| r.name.clone())
    }

    fn selected(&self) -> Option<&VmRow> {
        self.table.selected().and_then(|i| self.rows.get(i))
    }

    fn selected_vm(&mut self) -> Option<Vm> {
        let row = self.selected()?;
        match &row.vm {
            Ok(vm) => Some(vm.clone()),
            Err(e) => {
                let e = e.clone();
                self.say(&e, Level::Error);
                None
            }
        }
    }

    fn spawn<F>(&mut self, vm: &str, first: &str, work: F)
    where
        F: FnOnce(&mut dyn FnMut(&str)) -> Result<String> + Send + 'static,
    {
        if self.activity.contains_key(vm) {
            self.say(&format!("{vm} is busy: {}", self.activity[vm]), Level::Warn);
            return;
        }
        self.activity.insert(vm.to_string(), first.to_string());
        let tx = self.tx.clone();
        let name = vm.to_string();
        std::thread::spawn(move || {
            let t = tx.clone();
            let n = name.clone();
            let mut progress = move |s: &str| {
                let _ = t.send(Msg::Progress(n.clone(), s.to_string()));
            };
            let r = work(&mut progress).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Done(name, r));
        });
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.quit {
            terminal.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(250))?
                && let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
            {
                self.key(k);
            }
            if let Some((vm, files)) = self.pending_page.take() {
                self.page(terminal, &vm, &files)?;
            }
            if let Some(vm) = self.pending_view.take() {
                self.show_screen(terminal, &vm)?;
            }
            while let Ok(m) = self.rx.try_recv() {
                self.message(m);
            }
            if self.last_registry.elapsed() > Duration::from_secs(5) {
                let _ = self.reload();
            } else if self.last_state.elapsed() > Duration::from_secs(1) {
                self.refresh_state();
            }
        }
        Ok(())
    }

    /// Shows a VM's logs in less, on the plain terminal: its own selection then copies their
    /// text, a long line whole. Without less, the QEMU log shows in a view of reims-ctl's own.
    fn page(&mut self, terminal: &mut DefaultTerminal, vm: &str, files: &[PathBuf]) -> Result<()> {
        ratatui::restore();
        let status = run_less(files);
        enable_raw_mode()?;
        execute!(std::io::stdout(), EnterAlternateScreen)?;
        terminal.clear()?;
        match status {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => self.open_log(vm, false),
            Err(e) => self.say(&format!("less: {e}"), Level::Error),
        }
        Ok(())
    }

    /// Shows a running VM's screen in the terminal (view.rs) until its Control-].
    fn show_screen(&mut self, terminal: &mut DefaultTerminal, vm: &Vm) -> Result<()> {
        ratatui::restore();
        let result = crate::view::run(vm, &crate::view::Options::default());
        enable_raw_mode()?;
        execute!(std::io::stdout(), EnterAlternateScreen)?;
        terminal.clear()?;
        match result {
            Ok(Some(m)) => self.say(&m, Level::Info),
            Ok(None) => {}
            Err(e) => self.say(&format!("{}: {e:#}", vm.name()), Level::Error),
        }
        let _ = self.reload();
        Ok(())
    }

    fn message(&mut self, m: Msg) {
        match m {
            Msg::Progress(vm, s) => {
                self.status = Some((format!("{vm}: {s}"), Level::Info));
                self.activity.insert(vm, s);
            }
            Msg::Done(vm, r) => {
                self.activity.remove(&vm);
                match r {
                    Ok(s) => self.say(&s, Level::Ok),
                    Err(e) => self.say(&format!("{vm}: {e}"), Level::Error),
                }
                let _ = self.reload();
            }
            Msg::Gpus(list) => {
                let options = gpu_options(&list);
                self.gpus = Some(list);
                if let Some(Overlay::Form(_, f)) = &mut self.overlay {
                    f.set_options("gpu", options);
                }
            }
        }
    }

    // ------------------------------------------------------------ keys

    fn key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c')
            && k.modifiers.contains(KeyModifiers::CONTROL)
            && self.overlay.is_none()
        {
            self.try_quit();
            return;
        }
        match self.overlay.take() {
            Some(Overlay::Form(kind, mut f)) => match f.handle_key(k) {
                form::Event::None => {
                    display_help(&mut f);
                    self.overlay = Some(Overlay::Form(kind, f))
                }
                form::Event::Cancel => {}
                form::Event::Submit => match self.submit(&kind, &f) {
                    Ok(()) => {}
                    Err(e) => {
                        f.error = Some(format!("{e:#}"));
                        self.overlay = Some(Overlay::Form(kind, f));
                    }
                },
            },
            Some(Overlay::Text {
                title,
                lines,
                scroll,
                kind,
            }) => self.text_key(k, title, lines, scroll, kind),
            Some(Overlay::Confirm {
                title,
                body,
                options,
                choice,
            }) => match k.code {
                KeyCode::Left | KeyCode::Up | KeyCode::BackTab => {
                    let c = (choice + options.len() - 1) % options.len();
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice: c,
                    });
                }
                KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
                    let c = (choice + 1) % options.len();
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice: c,
                    });
                }
                KeyCode::Enter => self.act(options[choice].1.clone()),
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') => {}
                _ => {
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice,
                    })
                }
            },
            None => self.main_key(k),
        }
    }

    fn text_key(
        &mut self,
        k: KeyEvent,
        title: String,
        lines: Vec<String>,
        scroll: usize,
        kind: TextKind,
    ) {
        // Scrolling counts the rows the lines wrap to in the terminal as it is now.
        let (w, h) = ratatui::crossterm::terminal::size().unwrap_or((120, 40));
        let (width, height) = text_inner(Rect::new(0, 0, w, h));
        let max = wrap_rows(&lines, width).len().saturating_sub(height);
        let page = height.max(1);
        let mut scroll = scroll.min(max);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return,
            KeyCode::Down | KeyCode::Char('j') => scroll = (scroll + 1).min(max),
            KeyCode::Up | KeyCode::Char('k') => scroll = scroll.saturating_sub(1),
            KeyCode::PageDown | KeyCode::Char(' ') => scroll = (scroll + page).min(max),
            KeyCode::PageUp => scroll = scroll.saturating_sub(page),
            KeyCode::Home | KeyCode::Char('g') => scroll = 0,
            KeyCode::End | KeyCode::Char('G') => scroll = max,
            KeyCode::Char('w') => {
                if let TextKind::Command { vm, script } = &kind {
                    let r = self
                        .rows
                        .iter()
                        .find(|r| &r.name == vm)
                        .map(|r| r.dir.join("launch.sh"));
                    if let Some(p) = r {
                        use std::os::unix::fs::PermissionsExt;
                        let res = std::fs::write(&p, script).and_then(|_| {
                            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))
                        });
                        match res {
                            Ok(()) => self.say(&format!("wrote {}", paths::tilde(&p)), Level::Ok),
                            Err(e) => self.say(&format!("{}: {e}", p.display()), Level::Error),
                        }
                    }
                }
            }
            KeyCode::Char('y') => {
                if let TextKind::Command { script, .. } = &kind {
                    match copy_to_clipboard(script) {
                        Some(tool) => self.say(&format!("copied the command ({tool})"), Level::Ok),
                        None => self.say("no clipboard tool found (pbcopy, wl-copy, xclip or xsel): press w to write it to a file", Level::Warn),
                    }
                }
            }
            KeyCode::Char('s') => {
                if let TextKind::Log { vm, serial } = &kind {
                    let vm = vm.clone();
                    let serial = !*serial;
                    self.open_log(&vm, serial);
                    return;
                }
            }
            KeyCode::Char('r') => {
                if let TextKind::Log { vm, serial } = &kind {
                    let (vm, serial) = (vm.clone(), *serial);
                    self.open_log(&vm, serial);
                    return;
                }
            }
            _ => {}
        }
        self.overlay = Some(Overlay::Text {
            title,
            lines,
            scroll,
            kind,
        });
    }

    fn try_quit(&mut self) {
        if self.activity.is_empty() {
            self.quit = true;
            return;
        }
        let busy = self
            .activity
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n");
        self.overlay = Some(Overlay::Confirm {
            title: "Quit while work is running?".into(),
            body: format!(
                "{busy}\n\nQuitting stops this work where it is: a guest asked to shut down still does, \
                 but is no longer powered off if it does not; a VM being created is left unfinished, \
                 though a download of macOS picks up where it stopped when you create it again."
            ),
            options: vec![("Keep working", Action::None), ("Quit", Action::Quit)],
            choice: 0,
        });
    }

    fn main_key(&mut self, k: KeyEvent) {
        let n = self.rows.len();
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => self.try_quit(),
            KeyCode::Down | KeyCode::Char('j') if n > 0 => self
                .table
                .select(Some((self.table.selected().unwrap_or(0) + 1).min(n - 1))),
            KeyCode::Up | KeyCode::Char('k') if n > 0 => self
                .table
                .select(Some(self.table.selected().unwrap_or(0).saturating_sub(1))),
            KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::F(1) => self.overlay = Some(help()),
            KeyCode::Char('n') => self.open_create(),
            KeyCode::Char('s') | KeyCode::Enter => self.start(),
            KeyCode::Char('x') => self.stop(false),
            KeyCode::Char('K') => self.ask_force_stop(),
            KeyCode::Char('e') => self.open_edit(),
            KeyCode::Char('c') => self.show_command(),
            KeyCode::Char('v') => self.view_screen(),
            KeyCode::Char('l') => self.toggle_logs(),
            KeyCode::Char('L') => self.view_logs(),
            KeyCode::Char('b') => self.report(),
            KeyCode::Char('d') | KeyCode::Delete => self.ask_delete(),
            KeyCode::Char('u') => self.ask_update(),
            KeyCode::Char('r') => {
                let _ = self.reload();
                self.say("reloaded", Level::Info);
            }
            _ => {}
        }
    }

    fn act(&mut self, a: Action) {
        match a {
            Action::None => {}
            Action::Quit => self.quit = true,
            Action::Update(channel) => {
                self.update_requested = true;
                self.update_channel = channel;
                self.quit = true;
            }
            Action::Delete { vm, keep_files } => match delete_vm(&vm, keep_files) {
                Ok(s) => {
                    self.say(&s, Level::Ok);
                    let _ = self.reload();
                }
                Err(e) => self.say(&format!("{e:#}"), Level::Error),
            },
            Action::ForceStop(name) => {
                let Some(vm) = self
                    .rows
                    .iter()
                    .find(|r| r.name == name)
                    .and_then(|r| r.vm.as_ref().ok())
                    .cloned()
                else {
                    return;
                };
                self.activity.remove(&name);
                self.spawn(&name, "powering off", move |p| {
                    runtime::stop(&vm, true, Duration::ZERO, |s| p(s))?;
                    Ok(format!("{}: powered off", vm.name()))
                });
            }
        }
    }

    // ------------------------------------------------------------ actions

    fn start(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        if runtime::running(&vm) {
            self.say(&format!("{} is already running", vm.name()), Level::Info);
            return;
        }
        // What runtime::warnings finds shows in the VM's details, not here.
        let host = self.host;
        self.spawn(&vm.config.name.clone(), "starting", move |_| {
            let pid = runtime::start(&vm, host)?;
            Ok(format!(
                "{}: running (pid {pid}); its window is open",
                vm.name()
            ))
        });
    }

    fn stop(&mut self, force: bool) {
        let Some(vm) = self.selected_vm() else { return };
        if !runtime::running(&vm) {
            self.say(&format!("{} is not running", vm.name()), Level::Info);
            return;
        }
        let timeout = Duration::from_secs(vm.config.stop_timeout_secs);
        let first = if force {
            "powering off"
        } else {
            "shutting down"
        };
        self.spawn(&vm.config.name.clone(), first, move |p| {
            runtime::stop(&vm, force, timeout, |s| p(s))?;
            Ok(format!("{}: stopped", vm.name()))
        });
    }

    fn ask_force_stop(&mut self) {
        let Some(row) = self.selected() else { return };
        if row.pid.is_none() {
            let n = row.name.clone();
            self.say(&format!("{n} is not running"), Level::Info);
            return;
        }
        let name = row.name.clone();
        self.overlay = Some(Overlay::Confirm {
            title: format!("Power off {name}?"),
            body: "Powering off is pulling the plug: whatever the guest has not written to its disk is lost. \
                   Stop (x) asks the guest to shut down first."
                .into(),
            options: vec![("Cancel", Action::None), ("Power off", Action::ForceStop(name))],
            choice: 0,
        });
    }

    fn ask_delete(&mut self) {
        let Some(row) = self.selected() else { return };
        let name = row.name.clone();
        if row.pid.is_some() {
            self.say(&format!("{name} is running: stop it first"), Level::Warn);
            return;
        }
        let dir = paths::tilde(&row.dir);
        self.overlay = Some(Overlay::Confirm {
            title: format!("Delete {name}?"),
            body: format!(
                "Deleting the VM and its files removes {dir}, with its disks, for good. Keeping its \
                 files only takes it off this list. A disk used in place, outside that directory, stays either way."
            ),
            options: vec![
                ("Cancel", Action::None),
                (
                    "Delete it and its files",
                    Action::Delete {
                        vm: name.clone(),
                        keep_files: false,
                    },
                ),
                (
                    "Delete it, keep its files",
                    Action::Delete {
                        vm: name,
                        keep_files: true,
                    },
                ),
            ],
            choice: 0,
        });
    }

    fn ask_update(&mut self) {
        let Some(i) = install::Install::load() else {
            self.say(
                "no install record: run scripts/install-ctl.sh in your reims-vgpu checkout once",
                Level::Error,
            );
            return;
        };
        let running = self.rows.iter().filter(|r| r.pid.is_some()).count();
        let mut note = if running > 0 {
            format!("\n\n{running} VM(s) running keep their current QEMU until they restart.")
        } else {
            String::new()
        };
        if !self.activity.is_empty() {
            let busy = self
                .activity
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join("\n");
            note.push_str(&format!(
                "\n\nThe restart stops this work where it is, as quitting does (a download of \
                 macOS picks up where it stopped):\n{busy}"
            ));
        }
        let current = update::current_branch(&i.repo)
            .as_deref()
            .and_then(Channel::of);
        let mut options = vec![("Cancel", Action::None), ("Update", Action::Update(None))];
        for c in Channel::ALL.into_iter().filter(|&c| Some(c) != current) {
            let label = match c {
                Channel::Stable => "Switch to stable",
                Channel::Development => "Switch to development",
            };
            options.push((label, Action::Update(Some(c))));
        }
        self.overlay = Some(Overlay::Confirm {
            title: "Update reims-ctl?".into(),
            body: format!(
                "This pulls {}, which follows {}, and rebuilds reims-ctl, the library and QEMU (a few minutes; the build's \
                 output shows here), then restarts reims-ctl.\n\nA switch checks out that channel's branch first, and \
                 later updates stay on it: stable is master, and development gets changes first.{note}",
                paths::tilde(&i.repo),
                update::following(&i.repo)
            ),
            options,
            choice: 1,
        });
    }

    fn toggle_logs(&mut self) {
        let Some(mut vm) = self.selected_vm() else {
            return;
        };
        vm.config.logs.enabled = !vm.config.logs.enabled;
        match vm.save() {
            Ok(()) => {
                let on = vm.config.logs.enabled;
                let when = if runtime::running(&vm) {
                    " from its next start"
                } else {
                    ""
                };
                self.say(
                    &format!(
                        "{}: logs {}{when}{}",
                        vm.name(),
                        if on { "on" } else { "off" },
                        if on {
                            " (b makes a bug report of them)"
                        } else {
                            ""
                        }
                    ),
                    Level::Ok,
                );
                let _ = self.reload();
            }
            Err(e) => self.say(&format!("{e:#}"), Level::Error),
        }
    }

    fn report(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let host = self.host;
        let logs_on = vm.config.logs.enabled;
        self.spawn(&vm.config.name.clone(), "writing a bug report", move |_| {
            let p = logs::bundle(&vm, host)?;
            let mut s = format!("wrote {}", paths::tilde(&p));
            if !logs_on {
                s.push_str(
                    " · logs were off: turn them on (l), reproduce the problem, and report again",
                );
            }
            Ok(s)
        });
    }

    fn show_command(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        match runtime::invocation_for_print(&vm, self.host) {
            Ok(inv) => {
                let script = inv.shell_script(&cli::script_comment(&vm));
                self.overlay = Some(Overlay::Text {
                    title: format!(
                        "{}: the QEMU command  (w write launch.sh · y copy · Esc close)",
                        vm.name()
                    ),
                    lines: script.lines().map(String::from).collect(),
                    scroll: 0,
                    kind: TextKind::Command {
                        vm: vm.config.name.clone(),
                        script,
                    },
                });
            }
            Err(e) => self.say(&format!("{e:#}"), Level::Error),
        }
    }

    /// The selected VM's QEMU log, then its serial console, for the run loop to show in less.
    fn view_screen(&mut self) {
        let Some(r) = self.selected() else { return };
        if r.pid.is_none() {
            let name = r.name.clone();
            self.say(&format!("{name} is not running: s starts it"), Level::Warn);
            return;
        }
        self.pending_view = self.selected_vm();
    }

    fn view_logs(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let files: Vec<PathBuf> = [vm.qemu_log(), vm.serial_log()]
            .into_iter()
            .filter(|f| f.exists())
            .collect();
        if files.is_empty() {
            let p = paths::tilde(&vm.qemu_log());
            self.say(
                &format!("{}: {p} does not exist yet", vm.name()),
                Level::Info,
            );
            return;
        }
        self.pending_page = Some((vm.config.name.clone(), files));
    }

    fn open_log(&mut self, name: &str, serial: bool) {
        let Some(vm) = self
            .rows
            .iter()
            .find(|r| r.name == name)
            .and_then(|r| r.vm.as_ref().ok())
            .cloned()
        else {
            return;
        };
        let file = if serial {
            vm.serial_log()
        } else {
            vm.qemu_log()
        };
        let text = if file.exists() {
            runtime::tail(&file, 2000)
        } else {
            format!("{} does not exist yet", paths::tilde(&file))
        };
        let lines: Vec<String> = text.lines().map(String::from).collect();
        // Its end: drawing and the keys clamp this to the last screenful.
        let scroll = usize::MAX;
        let which = if serial { "serial console" } else { "QEMU log" };
        self.overlay = Some(Overlay::Text {
            title: format!(
                "{name}: {which}  (s {} · r reload · Esc close)",
                if serial { "QEMU log" } else { "serial console" }
            ),
            lines,
            scroll,
            kind: TextKind::Log {
                vm: name.to_string(),
                serial,
            },
        });
    }

    // ------------------------------------------------------------ forms

    fn gpu_field(&self, current: &crate::vm::Gpu) -> Field {
        let mut options = gpu_options(self.gpus.as_deref().unwrap_or(&[]));
        let selected = gpu_id(current);
        if !options.iter().any(|o| o.0 == selected) {
            options.push((
                selected.clone(),
                current.name.clone().unwrap_or_else(|| selected.clone()),
            ));
        }
        let help = if self.gpus.is_none() {
            "Looking for the host's GPUs…"
        } else {
            "The GPU the library draws on. Automatic takes the first discrete GPU, then the first integrated one."
        };
        Field::choice("gpu", "GPU", options, &selected, help)
    }

    fn open_create(&mut self) {
        let Some(profile) = self.host.default_profile() else {
            self.say(
                &format!("reims-ctl runs no VMs on {} yet", self.host),
                Level::Error,
            );
            return;
        };
        let name = match profile {
            Profile::Q35X86_64 => Field::text(
                "name",
                "Name",
                "",
                "Letters, digits, '-', '_' and '.'. Empty names a VM of downloaded macOS after it.",
            )
            .placeholder("the macOS version's, such as tahoe"),
            Profile::VmappleArm64 => {
                Field::text("name", "Name", "", "Letters, digits, '-', '_' and '.'.")
            }
        };
        let mut fields = vec![name];
        match profile {
            Profile::Q35X86_64 => {
                if !matches!(self.macos_versions, Some(Ok(_))) {
                    self.macos_versions =
                        Some(reims_util::versions().map_err(|e| format!("{e:#}")));
                }
                let (options, version_help) = match &self.macos_versions {
                    Some(Ok(list)) if !list.is_empty() => (
                        list.iter()
                            .map(|v| (v.slug.clone(), v.name.clone()))
                            .collect(),
                        "The macOS to download from Apple. x86-64 macOS 26 is the one we've tested.".to_string(),
                    ),
                    Some(Ok(_)) => (opts(&[("", "none")]), "reims-util lists no versions.".to_string()),
                    Some(Err(e)) => (opts(&[("", "unavailable")]), e.clone()),
                    None => (opts(&[("", "unavailable")]), String::new()),
                };
                let first = options.first().map(|o| o.0.clone()).unwrap_or_default();
                fields.extend([
                    Field::choice("source", "macOS from", opts(&[("download", "Download it from Apple"), ("installer", "An installer image of my own"), ("disk", "A disk that has macOS already")]), "download", "reims-ctl downloads the installer, makes the VM's OpenCore image and a new disk, and the VM boots into the installer."),
                    Field::choice("macos", "macOS version", options, &first, &version_help).show_if("source", &["download"]),
                    Field::choice("installer_kind", "Installer", opts(&[("online", "Online, about 1 GB"), ("offline", "Offline, 13 to 20 GB")]), "online", "Online is Apple's recovery image: macOS downloads the newest release as it installs. Offline holds one release whole, and installs with no network (macOS 11 and later). Either is kept for the next VM, and a stopped download picks up where it stopped.").show_if("source", &["download"]),
                    Field::text("release", "Release", "", "Optional: an exact release of that macOS, such as 26.6.2; empty for the newest. reims-util offline-installer list lists them.").placeholder("the newest").show_if("installer_kind", &["offline"]),
                    Field::path("installer", "Installer image", "", "An installer to boot (Tab completes): a raw image, such as a recovery image's BaseSystem.img, or a DMG, converted to a raw image in the VM's directory. Eject it once macOS is installed (e).").show_if("source", &["installer"]),
                    Field::text("disk_size", "Disk size", "128G", "The new disk's size; it takes only what the guest writes.").show_if("source", &["download", "installer"]),
                    Field::path("disk_path", "Disk image", "", "A qcow2 or raw image (Tab completes).").show_if("source", &["disk"]),
                    Field::choice("disk_mode", "Use the disk by", opts(&[("copy", "Copying it (instant where the file system reflinks)"), ("in-place", "Using it in place")]), "copy", "A copy leaves the image as it was; in place, the VM writes the image itself.")
                        .show_if("source", &["disk"]),
                    Field::path("opencore", "OpenCore image", "", "Empty makes one for this VM, with OVMF and a Mac identity of its own. Or an OpenCore image of yours to copy, with OVMF_CODE.fd and OVMF_VARS.fd beside it, such as the one a disk of macOS booted from, which keeps its Mac's identity.")
                        .placeholder("made for this VM"),
                ]);
            }
            Profile::VmappleArm64 => {
                fields.extend([
                    Field::choice("source", "macOS from", opts(&[("ipsw", "Install from an IPSW"), ("import", "Import a Virtualization.framework VM")]), "ipsw", "Install a fresh macOS, or bring a VM made with macosvm or tart."),
                    Field::path("ipsw", "IPSW", "latest", "An .ipsw restore image (Tab completes), or latest to download the newest this Mac supports (about 15 GB).").show_if("source", &["ipsw"]),
                    Field::text("disk_size", "Disk size", "64G", "The new disk's size; it takes only what the guest writes.").show_if("source", &["ipsw"]),
                    Field::path("bundle", "VM directory", "", "macosvm's directory (disk.img, aux.img, vm.json) or tart's (disk.img, nvram.bin, config.json).").show_if("source", &["import"]),
                    Field::choice("bundle_mode", "Use it by", opts(&[("copy", "Cloning it (instant on the same APFS volume)"), ("in-place", "Using it in place")]), "copy", "A clone leaves the original as it was.").show_if("source", &["import"]),
                ]);
            }
        }
        fields.extend([
            Field::path(
                "dir",
                "Storage directory",
                "",
                "Where its disks and settings go (Tab completes). Empty for the default.",
            )
            .placeholder(&format!(
                "{}/<name>",
                paths::tilde(&paths::default_vms_dir())
            )),
            Field::text(
                "cpus",
                "CPU cores",
                &create::default_cpus().to_string(),
                &format!("vCPUs; this host has {}.", create::host_cpus()),
            ),
            Field::text(
                "memory",
                "Memory",
                &format_mib(create::default_memory_mib()),
                &format!(
                    "Such as 8G; this host has {}.",
                    format_mib(create::host_memory_mib() / 1024 * 1024)
                ),
            ),
        ]);
        if profile.gpu_choice(self.host) {
            fields.push(self.gpu_field(&crate::vm::Gpu::default()));
        }
        fields.extend([
            Field::text(
                "displays",
                "Displays",
                "1",
                "1 to 8, each in a window of its own.",
            ),
            Field::text("resolution", "Resolution", "1920x1080", RESOLUTION_HELP),
            Field::toggle("hidpi", "HiDPI", false, HIDPI_HELP),
            Field::toggle(
                "logs",
                "Logs",
                false,
                "Keep the library's log and the serial console, for bug reports.",
            ),
            Field::text(
                "ssh_port",
                "SSH port",
                "",
                "Optional: a port on 127.0.0.1 forwarded to the guest's SSH (Remote Login).",
            )
            .placeholder("optional"),
        ]);
        let mut form = Form::new("New VM", "create", fields);
        display_help(&mut form);
        self.overlay = Some(Overlay::Form(FormKind::Create, form));
    }

    fn open_edit(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let c = &vm.config;
        let mut fields = vec![
            Field::text(
                "cpus",
                "CPU cores",
                &c.cpus.to_string(),
                &format!("vCPUs; this host has {}.", create::host_cpus()),
            ),
            Field::text("memory", "Memory", &format_mib(c.memory_mib), "Such as 8G."),
            Field::text(
                "displays",
                "Displays",
                &c.display.count.to_string(),
                "1 to 8, each in a window of its own.",
            ),
            Field::text(
                "resolution",
                "Resolution",
                &c.display.resolution,
                RESOLUTION_HELP,
            ),
            Field::toggle("hidpi", "HiDPI", c.display.hidpi, HIDPI_HELP),
            Field::toggle(
                "borderless",
                "Borderless",
                c.display.borderless,
                "Windows without borders that fill their screens.",
            ),
            Field::text(
                "refresh",
                "Refresh rate",
                &c.display.refresh.map(|r| r.to_string()).unwrap_or_default(),
                "Hz offered to the guest; empty for the host screen's.",
            )
            .placeholder("the host screen's"),
        ];
        if c.profile.gpu_choice(self.host) {
            fields.push(self.gpu_field(&c.gpu));
        }
        fields.extend([
            Field::toggle(
                "audio",
                "Audio",
                c.audio.enabled,
                "A USB audio device on the host's sound.",
            ),
            Field::toggle(
                "logs",
                "Logs",
                c.logs.enabled,
                "The library's log and the serial console, in the VM's logs/.",
            ),
            Field::text(
                "log_level",
                "Log level",
                &c.logs.level,
                "REIMS_VGPU_LOG's filter: error, warn, info, debug, trace, or target=level pairs.",
            )
            .show_if("logs", &["yes"]),
            Field::toggle(
                "network",
                "Network",
                c.network.enabled,
                "User-mode networking (NAT).",
            ),
            Field::text(
                "ssh_port",
                "SSH port",
                &c.network
                    .ssh_port
                    .map(|p| p.to_string())
                    .unwrap_or_default(),
                "A port on 127.0.0.1 forwarded to the guest's SSH; empty for none.",
            )
            .placeholder("none")
            .show_if("network", &["yes"]),
            Field::text(
                "stop_timeout",
                "Shutdown timeout",
                &c.stop_timeout_secs.to_string(),
                "Seconds stop (x) waits for the guest to shut down before it powers the VM off.",
            ),
        ]);
        if c.q35.is_some() {
            fields.push(
                Field::path(
                    "installer",
                    "Installer image",
                    &installer_shown(&vm),
                    "A raw image, or a DMG, converted to a raw image in the VM's directory (Tab completes). Empty to eject.",
                )
                .placeholder("none (ejected)"),
            );
        }
        let extra = c
            .extra_args
            .iter()
            .map(|a| crate::qemu::quote(a))
            .collect::<Vec<_>>()
            .join(" ");
        let env = c
            .extra_env
            .iter()
            .map(|(k, v)| format!("{k}={}", crate::qemu::quote(v)))
            .collect::<Vec<_>>()
            .join(" ");
        fields.extend([
            Field::text(
                "extra_args",
                "Extra QEMU args",
                &extra,
                "More arguments for QEMU, shell-quoted.",
            )
            .placeholder("none"),
            Field::text(
                "extra_env",
                "Extra environment",
                &env,
                "KEY=VALUE pairs for QEMU's environment, shell-quoted.",
            )
            .placeholder("none"),
        ]);
        let note = if runtime::running(&vm) {
            " (applies at its next start)"
        } else {
            ""
        };
        let mut form = Form::new(&format!("Edit {}{note}", c.name), "save", fields);
        display_help(&mut form);
        self.overlay = Some(Overlay::Form(FormKind::Edit(c.name.clone()), form));
    }

    fn submit(&mut self, kind: &FormKind, f: &Form) -> Result<()> {
        match kind {
            FormKind::Create => {
                let spec = self.spec_from(f)?;
                let host = self.host;
                let name = spec.name.clone();
                self.spawn(&name, "creating", move |p| {
                    let vm = create::create(spec, host, p)?;
                    let next = if runtime::install_steps(&vm).is_some() {
                        "; its details say how to install macOS"
                    } else if vm.config.q35.is_some() {
                        "; set Metal's error mode in its macOS if it isn't (README)"
                    } else {
                        ""
                    };
                    Ok(format!(
                        "created {} in {} · s starts it{next}",
                        vm.name(),
                        paths::tilde(&vm.dir)
                    ))
                });
                // Select it once it appears.
                self.pending_select = Some(name);
                Ok(())
            }
            FormKind::Edit(name) => {
                let mut vm = Registry::load()?.open(name)?;
                let installer = installer_change(&vm, f)?;
                if installer.is_some()
                    && let Some(a) = self.activity.get(name)
                {
                    bail!("{name} is busy: {a}");
                }
                apply_edit(&mut vm, f, self.gpus.as_deref())?;
                vm.save()?;
                let note = if runtime::running(&vm) {
                    " (at its next start)"
                } else {
                    ""
                };
                self.say(&format!("{name}: saved{note}"), Level::Ok);
                self.reload()?;
                // After the rest is saved: a DMG takes seconds to convert.
                if let Some(path) = installer {
                    self.spawn(name, "changing the installer", move |p| {
                        set_installer(&mut vm, path, p)
                    });
                }
                Ok(())
            }
        }
    }

    fn spec_from(&self, f: &Form) -> Result<Spec> {
        let mut name = f.text("name");
        if name.is_empty() && f.is_shown("macos") && f.choice("source") == "download" {
            // Its error first: a version that cannot be had names nothing.
            self.download_from(f)?;
            name = self.free_name(&f.choice("macos"));
        }
        crate::vm::check_name(&name)?;
        if self.rows.iter().any(|r| r.name == name) {
            bail!("a VM named {name} already exists");
        }
        let mut s = Spec::new(&name);
        let dir = f.text("dir");
        s.dir = (!dir.is_empty()).then(|| paths::absolute(&dir));
        s.cpus = Some(f.text("cpus").parse().context("CPU cores: a number")?);
        s.memory_mib = Some(parse_size_mib(&f.text("memory")).context("memory")?);
        if f.is_shown("gpu") {
            s.gpu = gpu_from_choice(&f.choice("gpu"), self.gpus.as_deref().unwrap_or(&[]));
        }
        let opt_path = |key: &str| {
            let t = f.text(key);
            (!t.is_empty()).then(|| paths::absolute(&t))
        };
        match self.host.default_profile() {
            Some(Profile::Q35X86_64) => {
                let size = || -> Result<DiskSource> {
                    Ok(DiskSource::New {
                        size_mib: parse_size_mib(&f.text("disk_size")).context("disk size")?,
                    })
                };
                match f.choice("source").as_str() {
                    "download" => {
                        s.macos = Some(self.download_from(f)?);
                        s.disk = Some(size()?);
                    }
                    "installer" => {
                        s.installer =
                            Some(opt_path("installer").context("give the installer image")?);
                        s.disk = Some(size()?);
                    }
                    _ => {
                        s.disk = Some(DiskSource::Existing {
                            path: opt_path("disk_path").context("give the disk image's path")?,
                            mode: ImportMode::parse(&f.choice("disk_mode"))?,
                        });
                    }
                }
                s.opencore = opt_path("opencore");
            }
            Some(Profile::VmappleArm64) => {
                if f.choice("source") == "ipsw" {
                    let i = f.text("ipsw");
                    if i.is_empty() {
                        bail!("give an IPSW, or latest");
                    }
                    s.ipsw = Some(if i == "latest" {
                        i
                    } else {
                        paths::absolute(&i).to_string_lossy().into_owned()
                    });
                    s.disk = Some(DiskSource::New {
                        size_mib: parse_size_mib(&f.text("disk_size")).context("disk size")?,
                    });
                } else {
                    s.bundle = Some(opt_path("bundle").context("give the VM's directory")?);
                    s.bundle_mode = ImportMode::parse(&f.choice("bundle_mode"))?;
                }
            }
            None => bail!("this host runs no VMs"),
        }
        s.display.count = f.text("displays").parse().context("displays: a number")?;
        s.display.resolution = f.text("resolution");
        s.display.hidpi = f.toggle("hidpi");
        s.display.check()?;
        s.logs = f.toggle("logs");
        let ssh = f.text("ssh_port");
        s.ssh_port = if ssh.is_empty() {
            None
        } else {
            Some(ssh.parse().context("SSH port: a number")?)
        };
        Ok(s)
    }

    /// `base`, or `base-2`, `base-3`… where a VM has the name, or its default directory holds
    /// files (a VM deleted with its files kept, or one whose create stopped).
    fn free_name(&self, base: &str) -> String {
        let taken = |n: &str| {
            self.rows.iter().any(|r| r.name == n)
                || self.activity.contains_key(n)
                || std::fs::read_dir(create::default_dir(n)).is_ok_and(|mut d| d.next().is_some())
        };
        (1..)
            .map(|i| {
                if i == 1 {
                    base.to_string()
                } else {
                    format!("{base}-{i}")
                }
            })
            .find(|n| !taken(n))
            .unwrap_or_default()
    }

    /// The download the form asks for: its version, and an offline installer's release.
    fn download_from(&self, f: &Form) -> Result<reims_util::Download> {
        let slug = f.choice("macos");
        let version = match &self.macos_versions {
            Some(Ok(list)) => list.iter().find(|v| v.slug == slug),
            Some(Err(e)) => bail!("{e}"),
            None => None,
        }
        .context("choose the macOS version")?;
        let offline = f.choice("installer_kind") == "offline";
        if !offline {
            return Ok(reims_util::Download {
                version: slug,
                offline,
            });
        }
        if !version.has_offline() {
            bail!(
                "{} has no offline installer (macOS 11 and later do): choose Online",
                version.name
            );
        }
        let release = f.text("release");
        if release.is_empty() {
            return Ok(reims_util::Download {
                version: slug,
                offline,
            });
        }
        if release.split('.').next() != Some(version.number()) {
            bail!(
                "Release: a release of {}, such as {}.1, or empty for the newest",
                version.name,
                version.number()
            );
        }
        Ok(reims_util::Download {
            version: release,
            offline,
        })
    }

    // ------------------------------------------------------------ drawing

    fn draw(&mut self, f: &mut Frame) {
        if let Some(name) = self.pending_select.clone()
            && let Some(i) = self.rows.iter().position(|r| r.name == name)
        {
            self.table.select(Some(i));
            self.pending_select = None;
        }
        let area = f.area();
        let width = area.width as usize;
        // A row each, unless the status or the hints are wider than the terminal.
        let status_rows = match &self.status {
            Some((s, _)) if s.chars().count() + 1 > width => 2,
            _ => 1,
        };
        let hints = key_hints();
        let hint_rows = if hints.width() > width { 2 } else { 1 };
        let [header, body, status, keys] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(status_rows),
            Constraint::Length(hint_rows),
        ])
        .areas(area);
        self.draw_header(f, header);
        if area.width >= 100 {
            let [list, detail] =
                Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)])
                    .areas(body);
            self.draw_list(f, list);
            self.draw_detail(f, detail);
        } else {
            let [list, detail] =
                Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
                    .areas(body);
            self.draw_list(f, list);
            self.draw_detail(f, detail);
        }
        self.draw_status(f, status);
        f.render_widget(Paragraph::new(hints).wrap(Wrap { trim: false }), keys);
        match &self.overlay {
            Some(Overlay::Form(_, form)) => form.render(f, area),
            Some(Overlay::Text {
                title,
                lines,
                scroll,
                ..
            }) => draw_text(f, area, title, lines, *scroll),
            Some(Overlay::Confirm {
                title,
                body,
                options,
                choice,
            }) => draw_confirm(f, area, title, body, options, *choice),
            None => {}
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let profile = self
            .host
            .default_profile()
            .map(|p| p.describe().to_string())
            .unwrap_or_else(|| "no VMs on this host".into());
        let qemu = match &self.qemu {
            Some(q) => Span::styled(q.clone(), Style::new().fg(Color::Green)),
            None => Span::styled("QEMU not installed", Style::new().fg(Color::Red).bold()),
        };
        let line = Line::from(vec![
            Span::styled(
                " reims-ctl ",
                Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
            ),
            Span::raw(format!("  {} · {profile} · ", self.host)),
            qemu,
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    fn draw_list(&mut self, f: &mut Frame, area: Rect) {
        let mut rows: Vec<Row> = self
            .rows
            .iter()
            .map(|r| {
                let (state, style) = match (self.activity.get(&r.name), r.pid, &r.vm) {
                    (Some(a), _, _) => (a.clone(), Style::new().fg(Color::Yellow)),
                    (None, _, Err(_)) => ("broken".into(), Style::new().fg(Color::Red)),
                    (None, Some(_), _) => ("running".into(), Style::new().fg(Color::Green)),
                    (None, None, _) => ("stopped".into(), Style::new().dim()),
                };
                let (cpus, mem) = match &r.vm {
                    Ok(vm) => (vm.config.cpus.to_string(), format_mib(vm.config.memory_mib)),
                    Err(_) => (String::new(), String::new()),
                };
                Row::new(vec![
                    Cell::from(r.name.clone()),
                    Cell::from(state).style(style),
                    Cell::from(cpus),
                    Cell::from(mem),
                ])
            })
            .collect();
        for (name, a) in &self.activity {
            if !self.rows.iter().any(|r| &r.name == name) {
                rows.push(Row::new(vec![
                    Cell::from(name.clone()),
                    Cell::from(a.clone()).style(Style::new().fg(Color::Yellow)),
                    Cell::from(""),
                    Cell::from(""),
                ]));
            }
        }
        let empty = rows.is_empty();
        let table = Table::new(
            rows,
            [
                Constraint::Min(12),
                Constraint::Min(12),
                Constraint::Length(5),
                Constraint::Length(7),
            ],
        )
        .header(Row::new(vec!["NAME", "STATE", "CPUS", "MEMORY"]).style(Style::new().bold().dim()))
        .block(Block::bordered().title(" VMs "))
        .row_highlight_style(
            Style::new()
                .bg(Color::Indexed(237))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");
        f.render_stateful_widget(table, area, &mut self.table);
        if empty {
            let inner = Rect {
                x: area.x + 2,
                y: area.y + 3,
                width: area.width.saturating_sub(4),
                height: 2,
            };
            f.render_widget(
                Paragraph::new("No VMs yet. Press n to create one.").dim(),
                inner,
            );
        }
    }

    fn draw_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::bordered().title(match self.selected() {
            Some(r) => format!(" {} ", r.name),
            None => " details ".into(),
        });
        let lines: Vec<Line> = match self.selected() {
            None => vec![Line::from("")],
            Some(r) => match &r.vm {
                Err(e) => vec![
                    Line::from(Span::styled(e.clone(), Style::new().fg(Color::Red))),
                    Line::from(format!("directory {}", paths::tilde(&r.dir))),
                ],
                Ok(vm) => {
                    let mut lines: Vec<Line> = describe(vm, self.host)
                        .into_iter()
                        .map(|(k, v)| {
                            Line::from(vec![
                                Span::styled(format!("{k:<13} "), Style::new().dim()),
                                Span::raw(v),
                            ])
                        })
                        .collect();
                    if r.pid.is_some() {
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled(
                            "v shows its screen here, in the terminal",
                            Style::new().dim(),
                        )));
                    }
                    if let Some(steps) = runtime::install_steps(vm) {
                        lines.push(Line::from(""));
                        lines.push(Line::from(steps));
                    }
                    // Below its settings, what in them its guest does not work with, then
                    // what starting it warns of on this host.
                    let incompatible = vm.config.incompatibilities();
                    for w in incompatible
                        .into_iter()
                        .chain(runtime::warnings(vm, self.host))
                    {
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled(w, Style::new().fg(Color::Yellow))));
                    }
                    lines
                }
            },
        };
        f.render_widget(
            Paragraph::new(lines)
                .block(block)
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let Some((s, level)) = &self.status else {
            return;
        };
        let style = match level {
            Level::Info => Style::new(),
            Level::Ok => Style::new().fg(Color::Green),
            Level::Warn => Style::new().fg(Color::Yellow),
            Level::Error => Style::new().fg(Color::Red),
        };
        f.render_widget(
            Paragraph::new(Span::styled(format!(" {s}"), style)).wrap(Wrap { trim: true }),
            area,
        );
    }
}

/// The main screen's key hints: the everyday keys, and the help, which lists the rest.
fn key_hints() -> Line<'static> {
    let keys = [
        ("n", "new"),
        ("s", "start"),
        ("x", "stop"),
        ("K", "kill"),
        ("d", "delete"),
        ("u", "update"),
        ("?", "help"),
    ];
    // Non-breaking spaces inside each hint, so a narrow terminal wraps between hints only.
    let mut spans = Vec::new();
    for (k, what) in keys {
        if !spans.is_empty() {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("\u{a0}{k}\u{a0}"),
            Style::new().fg(Color::Black).bg(Color::DarkGray),
        ));
        spans.push(Span::raw(format!("\u{a0}{what}")));
    }
    Line::from(spans)
}

fn gpu_id(g: &crate::vm::Gpu) -> String {
    match (&g.driver_file, g.device_id) {
        (None, _) => "auto".into(),
        (Some(f), Some(id)) => format!("{}#{id}", f.display()),
        (Some(f), None) => f.display().to_string(),
    }
}

fn gpu_options(list: &[GpuInfo]) -> Vec<(String, String)> {
    let mut v = vec![("auto".to_string(), "Automatic (discrete first)".to_string())];
    for g in list {
        v.push((gpu_id(&g.to_config()), g.label()));
    }
    v
}

fn gpu_from_choice(id: &str, list: &[GpuInfo]) -> crate::vm::Gpu {
    if id == "auto" {
        return crate::vm::Gpu::default();
    }
    if let Some(g) = list.iter().find(|g| gpu_id(&g.to_config()) == id) {
        return g.to_config();
    }
    let (file, dev) = match id.split_once('#') {
        Some((f, d)) => (f, d.parse().ok()),
        None => (id, None),
    };
    crate::vm::Gpu {
        driver_file: Some(file.into()),
        device_id: dev,
        name: None,
    }
}

const RESOLUTION_HELP: &str =
    "The guest's first mode: WIDTHxHEIGHT, or host for the host screen's own modes and density.";

const HIDPI_HELP: &str = "Run the mode at twice the density, as a Retina screen does: the desktop \
     looks like half the mode, so a larger mode makes everything smaller (3840x2160 looks like \
     1920x1080, 3200x1800 like 1600x900, 2560x1440 like 1280x720). Needs a mode at least 1920 \
     pixels wide.";

/// The resolution's and HiDPI's help, each ending with what the desktop looks like at the
/// values the form holds now.
fn display_help(f: &mut Form) {
    if !f.is_shown("resolution") || !f.is_shown("hidpi") {
        return;
    }
    let resolution = f.text("resolution");
    let hidpi = f.toggle("hidpi");
    let looks_like = |hidpi| {
        crate::vm::Display {
            resolution: resolution.clone(),
            hidpi,
            ..Default::default()
        }
        .looks_like()
    };
    let now = match (parse_resolution(&resolution), looks_like(true)) {
        _ if resolution == "host" && hidpi => {
            "host takes the host screen's own density: turn HiDPI off.".to_string()
        }
        (Ok(_), Some((w, h))) if hidpi => format!("Now {resolution} looks like {w}x{h}."),
        (Ok(_), Some((w, h))) => format!("At HiDPI, {resolution} would look like {w}x{h}."),
        (Ok((w, _)), None) if hidpi => format!("{w} pixels wide is narrower than 1920."),
        _ => String::new(),
    };
    f.set_help("resolution", &format!("{RESOLUTION_HELP} {now}"));
    f.set_help("hidpi", &format!("{HIDPI_HELP} {now}"));
}

/// The VM's installer as the edit form shows it: its full path, or empty.
fn installer_shown(vm: &Vm) -> String {
    vm.config
        .q35
        .as_ref()
        .and_then(|q| q.installer.as_ref())
        .map(|p| vm.path(p).to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The installer the edit form changes the VM's to, if it changes it; `Some(None)` ejects it.
fn installer_change(vm: &Vm, f: &Form) -> Result<Option<Option<PathBuf>>> {
    if !f.is_shown("installer") {
        return Ok(None);
    }
    let t = f.text("installer");
    if t == installer_shown(vm) {
        return Ok(None);
    }
    let path = (!t.is_empty()).then(|| paths::absolute(&t));
    if let Some(p) = &path
        && !p.is_file()
    {
        bail!("installer image: {} does not exist", p.display());
    }
    Ok(Some(path))
}

fn apply_edit(vm: &mut Vm, f: &Form, gpus: Option<&[GpuInfo]>) -> Result<()> {
    let c = &mut vm.config;
    c.cpus = f.text("cpus").parse().context("CPU cores: a number")?;
    c.memory_mib = parse_size_mib(&f.text("memory")).context("memory")?;
    c.display.count = f.text("displays").parse().context("displays: a number")?;
    c.display.resolution = f.text("resolution");
    c.display.hidpi = f.toggle("hidpi");
    c.display.borderless = f.toggle("borderless");
    let r = f.text("refresh");
    c.display.refresh = if r.is_empty() {
        None
    } else {
        Some(r.parse().context("refresh rate: a number")?)
    };
    if f.is_shown("gpu") {
        let id = f.choice("gpu");
        if id != gpu_id(&c.gpu) {
            c.gpu = gpu_from_choice(&id, gpus.unwrap_or(&[]));
        }
    }
    c.audio.enabled = f.toggle("audio");
    c.logs.enabled = f.toggle("logs");
    if f.is_shown("log_level") && !f.text("log_level").is_empty() {
        c.logs.level = f.text("log_level");
    }
    c.network.enabled = f.toggle("network");
    if f.is_shown("ssh_port") {
        let p = f.text("ssh_port");
        c.network.ssh_port = if p.is_empty() {
            None
        } else {
            Some(p.parse().context("SSH port: a number")?)
        };
    }
    c.stop_timeout_secs = f
        .text("stop_timeout")
        .parse()
        .context("shutdown timeout: seconds")?;
    c.extra_args =
        shlex::split(&f.text("extra_args")).context("extra QEMU args: unbalanced quotes")?;
    c.extra_env.clear();
    for kv in shlex::split(&f.text("extra_env")).context("extra environment: unbalanced quotes")? {
        let (k, v) = kv
            .split_once('=')
            .with_context(|| format!("extra environment takes KEY=VALUE, not {kv:?}"))?;
        c.extra_env.insert(k.into(), v.into());
    }
    c.check()
}

/// Runs less on `files`, each at its end, as system(3) runs a command: Ctrl+C and Ctrl+\ reach
/// the terminal's whole foreground process group, so reims-ctl ignores them until less exits,
/// and less gets them as it would from a shell.
fn run_less(files: &[PathBuf]) -> std::io::Result<std::process::ExitStatus> {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new("less");
    cmd.arg("++G").args(files);
    // SAFETY: only signal, which is async-signal-safe, between fork and exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            Ok(())
        });
    }
    // SAFETY: dispositions set to ignore, then back to what they were.
    let (int, quit) = unsafe {
        (
            libc::signal(libc::SIGINT, libc::SIG_IGN),
            libc::signal(libc::SIGQUIT, libc::SIG_IGN),
        )
    };
    let status = cmd.status();
    unsafe {
        libc::signal(libc::SIGINT, int);
        libc::signal(libc::SIGQUIT, quit);
    }
    status
}

fn copy_to_clipboard(s: &str) -> Option<&'static str> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let tools: &[(&'static str, &[&str])] = &[
        ("pbcopy", &[]),
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["-b", "-i"]),
    ];
    for (tool, args) in tools {
        let Ok(mut child) = Command::new(tool)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(s.as_bytes());
        }
        if child.wait().map(|st| st.success()).unwrap_or(false) {
            return Some(tool);
        }
    }
    None
}

const HELP: &str = "\
reims-ctl runs macOS VMs on QEMU with reims-vgpu: each VM's displays open in windows of their own.

  ↑ ↓ / j k   choose a VM
  n           create a VM. On Linux, choose a macOS version: reims-ctl downloads its
              installer from Apple and makes the VM's OpenCore image and disk, and the VM
              boots into the installer; its details then say how to install macOS
  s / Enter   start it
  x           stop it: the guest is asked to shut down, and powered off if it has not
              when its shutdown timeout ends (edit e to change it)
  K           kill it: power it off at once
  e           edit its settings: CPUs, memory, displays, resolution, HiDPI, borderless,
              GPU, audio, network, logs, extra QEMU arguments (they apply at its next start),
              and its installer image (x86-64)
  c           show the QEMU command it runs, to write as launch.sh (w) or copy (y):
              run it yourself, without reims-ctl
  v           show a running VM's screen here, in the terminal, and use its keyboard and
              mouse from here: for a host with no screen, over SSH too. Ctrl-] comes back
  l           turn its logs on or off
  L           view its QEMU log in less: :n shows the serial console, / searches,
              F follows the log as it grows, q comes back
  b           write a bug report: its logs, settings, command and this host's details,
              zipped into your Downloads folder
  d           delete it
  u           update reims-ctl and QEMU from your reims-vgpu checkout
  r           reload the list
  ?           this help
  q           quit; running VMs keep running

The same from a shell: reims-ctl help.";

fn help() -> Overlay {
    Overlay::Text {
        title: "help  (Esc close)".into(),
        lines: HELP.lines().map(String::from).collect(),
        scroll: 0,
        kind: TextKind::Help,
    }
}

/// The columns and rows of a text view's text: the whole screen below its title row, with no
/// border or margin beside the text, so the terminal's selection copies the text alone.
fn text_inner(area: Rect) -> (usize, usize) {
    (
        area.width.max(1) as usize,
        area.height.saturating_sub(1) as usize,
    )
}

/// `lines` cut into rows of at most `width` characters, so a long line (a command's drive
/// paths, a log's message) shows whole rather than past the screen's edge.
fn wrap_rows(lines: &[String], width: usize) -> Vec<&str> {
    let mut rows = Vec::new();
    for l in lines {
        let mut rest = l.as_str();
        loop {
            match rest.char_indices().nth(width) {
                Some((i, _)) => {
                    rows.push(&rest[..i]);
                    rest = &rest[i..];
                }
                None => {
                    rows.push(rest);
                    break;
                }
            }
        }
    }
    rows
}

fn draw_text(f: &mut Frame, area: Rect, title: &str, lines: &[String], scroll: usize) {
    f.render_widget(Clear, area);
    let [top, body] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
    f.render_widget(
        Paragraph::new(format!(" {title}")).style(Style::new().fg(Color::Black).bg(Color::Cyan)),
        top,
    );
    let (width, height) = text_inner(area);
    let rows = wrap_rows(lines, width);
    let start = scroll.min(rows.len().saturating_sub(height));
    let shown: Vec<Line> = rows
        .into_iter()
        .skip(start)
        .take(height)
        .map(Line::from)
        .collect();
    f.render_widget(Paragraph::new(shown), body);
}

fn draw_confirm(
    f: &mut Frame,
    area: Rect,
    title: &str,
    body: &str,
    options: &[(&str, Action)],
    choice: usize,
) {
    let width = area.width.saturating_sub(4).min(76);
    let body_lines = body
        .lines()
        .map(|l| {
            (l.chars().count() as u16)
                .div_ceil(width.saturating_sub(4).max(10))
                .max(1)
        })
        .sum::<u16>();
    let popup = area.centered(
        Constraint::Length(width),
        Constraint::Length((body_lines + 5).min(area.height)),
    );
    f.render_widget(Clear, popup);
    let block = Block::bordered()
        .title(Line::from(format!(" {title} ")).bold())
        .border_style(Style::new().fg(Color::Yellow));
    let inner = block.inner(popup);
    f.render_widget(block, popup);
    let [b, o] = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), b);
    let mut spans = Vec::new();
    for (i, (label, _)) in options.iter().enumerate() {
        let style = if i == choice {
            Style::new().fg(Color::Black).bg(Color::Yellow).bold()
        } else {
            Style::new()
        };
        spans.push(Span::styled(format!(" {label} "), style));
        spans.push(Span::raw("  "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), o);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_rows_cuts_at_the_width() {
        let lines = vec!["abcdef".to_string(), String::new(), "ééé".to_string()];
        assert_eq!(wrap_rows(&lines, 3), ["abc", "def", "", "ééé"]);
        assert_eq!(wrap_rows(&lines, 2), ["ab", "cd", "ef", "", "éé", "é"]);
    }

    #[test]
    fn key_hints_fit_an_80_column_terminal() {
        assert!(key_hints().width() <= 80);
    }

    #[test]
    fn display_help_says_what_the_desktop_looks_like() {
        let now = |resolution: &str, hidpi: bool| {
            let mut f = Form::new(
                "t",
                "save",
                vec![
                    Field::text("resolution", "Resolution", resolution, RESOLUTION_HELP),
                    Field::toggle("hidpi", "HiDPI", hidpi, HIDPI_HELP),
                ],
            );
            display_help(&mut f);
            assert_eq!(
                f.help("resolution").strip_prefix(RESOLUTION_HELP),
                f.help("hidpi").strip_prefix(HIDPI_HELP)
            );
            f.help("hidpi")[HIDPI_HELP.len()..].trim().to_string()
        };
        assert_eq!(now("3200x1800", true), "Now 3200x1800 looks like 1600x900.");
        assert_eq!(
            now("3840x2160", false),
            "At HiDPI, 3840x2160 would look like 1920x1080."
        );
        assert_eq!(
            now("1280x800", true),
            "1280 pixels wide is narrower than 1920."
        );
        assert_eq!(now("1280x800", false), "");
        assert_eq!(
            now("host", true),
            "host takes the host screen's own density: turn HiDPI off."
        );
        assert_eq!(now("host", false), "");
        assert_eq!(now("19x", true), "");
    }
}

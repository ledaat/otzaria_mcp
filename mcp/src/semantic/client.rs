//! The stdio MCP's side of the semantic tools: they are answered by the `--http` daemon, so
//! Otzaria's vectors and the model are loaded once however many clients there are (Claude, the
//! Obsidian plugin, the Otzaria plugin). Whichever starts first starts the daemon; a daemon of
//! another version (or the old Python server) on the port is asked to shut down and replaced.

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use super::service::VERSION;

pub struct DaemonClient {
    url: String,
    /// The arguments a started daemon gets after `--http <addr>` (the data paths).
    spawn_args: Vec<OsString>,
    log_file: Option<PathBuf>,
    child: Mutex<Option<std::process::Child>>,
    agent: ureq::Agent,
}

fn log_line(path: &Option<PathBuf>, line: &str) {
    if let Some(p) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = writeln!(
                f,
                "[{}] {line}",
                super::feedback::utc_iso(std::time::SystemTime::now())
            );
        }
    }
}

impl DaemonClient {
    pub fn new(url: &str, spawn_args: Vec<OsString>, log_file: Option<PathBuf>) -> DaemonClient {
        DaemonClient {
            url: url.trim_end_matches('/').to_string(),
            spawn_args,
            log_file,
            child: Mutex::new(None),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(3))
                .build(),
        }
    }

    fn addr(&self) -> String {
        self.url
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .to_string()
    }

    pub fn health(&self) -> Option<Value> {
        self.agent
            .get(&format!("{}/health", self.url))
            .timeout(Duration::from_secs(3))
            .call()
            .ok()?
            .into_json()
            .ok()
    }

    fn ours(h: &Value) -> bool {
        h.get("server").and_then(Value::as_str) == Some("otzaria-mcp")
            && h.get("version").and_then(Value::as_str) == Some(VERSION)
    }

    /// The daemon's /health, starting or replacing it when needed.
    pub fn ensure(&self) -> Result<Value> {
        let mut child = self.child.lock().unwrap();
        if let Some(h) = self.health() {
            if Self::ours(&h) {
                return Ok(h);
            }
            let theirs = h
                .get("version")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string();
            log_line(
                &self.log_file,
                &format!(
                    "otzaria-mcp {VERSION} (stdio) is replacing the server {theirs} on {}",
                    self.url
                ),
            );
            self.replace()?;
            if let Some(mut c) = child.take() {
                let _ = c.wait();
            }
        }
        let running = child
            .as_mut()
            .is_some_and(|c| matches!(c.try_wait(), Ok(None)));
        if !running {
            *child = Some(self.spawn()?);
        }
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(180) {
            std::thread::sleep(Duration::from_millis(500));
            if let Some(h) = self.health() {
                return Ok(h);
            }
            if let Some(c) = child.as_mut() {
                if let Ok(Some(status)) = c.try_wait() {
                    *child = None;
                    bail!(
                        "the semantic server exited ({status}) while starting{}",
                        self.log_tail()
                    );
                }
            }
        }
        bail!(
            "the semantic server did not start within 3 minutes{}",
            self.log_tail()
        )
    }

    fn log_tail(&self) -> String {
        let Some(p) = &self.log_file else {
            return String::new();
        };
        let Ok(text) = std::fs::read(p) else {
            return String::new();
        };
        let text = String::from_utf8_lossy(&text);
        let tail: Vec<&str> = text
            .lines()
            .rev()
            .take(12)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!(". Last lines of {}:\n{}", p.display(), tail.join("\n"))
    }

    /// Ask the server on the port to stop; if it does not (a server without /shutdown),
    /// end the process listening there (Windows).
    fn replace(&self) -> Result<()> {
        let _ = self
            .agent
            .post(&format!("{}/shutdown", self.url))
            .timeout(Duration::from_secs(3))
            .send_string("{}");
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(300));
            if self.health().is_none() {
                return Ok(());
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let port = self.addr().rsplit(':').next().unwrap_or("").to_string();
            if let Ok(out) = std::process::Command::new("netstat")
                .args(["-ano", "-p", "TCP"])
                .creation_flags(0x0800_0000)
                .output()
            {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    let f: Vec<&str> = line.split_whitespace().collect();
                    if f.len() >= 5
                        && f[0] == "TCP"
                        && f[3] == "LISTENING"
                        && f[1].ends_with(&format!(":{port}"))
                        && f[4] != "0"
                    {
                        let _ = std::process::Command::new("taskkill")
                            .args(["/PID", f[4], "/F"])
                            .creation_flags(0x0800_0000)
                            .output();
                    }
                }
            }
            for _ in 0..10 {
                std::thread::sleep(Duration::from_millis(300));
                if self.health().is_none() {
                    return Ok(());
                }
            }
        }
        bail!("another server on {} does not stop", self.url)
    }

    fn spawn(&self) -> Result<std::process::Child> {
        let exe = std::env::current_exe()?;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--http").arg(self.addr()).args(&self.spawn_args);
        cmd.stdin(std::process::Stdio::null());
        match &self.log_file {
            Some(p) => {
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)?;
                cmd.stdout(f.try_clone()?).stderr(f);
            }
            None => {
                cmd.stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        log_line(
            &self.log_file,
            &format!(
                "=== started by Claude (otzaria-mcp {VERSION}, pid {}): {}",
                std::process::id(),
                exe.display()
            ),
        );
        cmd.spawn()
            .map_err(|e| anyhow!("cannot start the semantic server: {e}"))
    }

    pub fn call(&self, path: &str, body: &Value) -> Result<Value> {
        self.ensure()?;
        let resp = self
            .agent
            .post(&format!("{}{path}", self.url))
            .timeout(Duration::from_secs(600))
            .send_json(body.clone());
        match resp {
            Ok(r) => Ok(r.into_json()?),
            Err(ureq::Error::Status(_, r)) => {
                let v: Value = r.into_json().unwrap_or(Value::Null);
                bail!(
                    "{}",
                    v.get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("the semantic server failed")
                )
            }
            Err(e) => bail!("the semantic server is unreachable: {e}"),
        }
    }
}

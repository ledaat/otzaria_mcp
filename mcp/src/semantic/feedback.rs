//! Relevance feedback to the Meivin model author's feedback service, through a local outbox.
//!
//! Protocol per the author's guide: `POST {base}/call/v2/save_feedback {"data": {...}}` returns
//! an `event_id`; `GET {base}/call/save_feedback/{event_id}` streams server-sent events, and the
//! event is saved only on `event: complete` with `success: true`. Events wait in the outbox (one
//! JSON file each) until confirmed, are retried every few minutes under the same id, and expire
//! after 60 days. Only the HTTP mode starts an outbox: the stdio MCP never reaches the network.

use std::io::BufRead;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

pub const FEEDBACK_BASE: &str = "https://arielll123-meivin-api-data.hf.space/gradio_api";
const MAX_AGE: Duration = Duration::from_secs(60 * 86_400);
const RETRY: Duration = Duration::from_secs(180);

struct Shared {
    dir: PathBuf,
    base: String,
    last_error: Mutex<Option<String>>,
    wake: (Mutex<bool>, Condvar),
}

pub struct Outbox {
    shared: Arc<Shared>,
}

impl Outbox {
    pub fn start(dir: PathBuf) -> Outbox {
        let shared = Arc::new(Shared {
            dir,
            base: FEEDBACK_BASE.to_string(),
            last_error: Mutex::new(None),
            wake: (Mutex::new(false), Condvar::new()),
        });
        let s = shared.clone();
        std::thread::Builder::new()
            .name("feedback-outbox".into())
            .spawn(move || loop {
                if let Err(e) = flush(&s) {
                    *s.last_error.lock().unwrap() = Some(format!("{e:#}"));
                }
                let (lock, cv) = &s.wake;
                let woken = lock.lock().unwrap();
                let (mut woken, _) = cv.wait_timeout_while(woken, RETRY, |w| !*w).unwrap();
                *woken = false;
            })
            .ok();
        Outbox { shared }
    }

    pub fn pending_count(&self) -> usize {
        pending(&self.shared).len()
    }

    pub fn last_error(&self) -> Option<String> {
        self.shared.last_error.lock().unwrap().clone()
    }

    pub fn enqueue(&self, event: &Value) -> Result<()> {
        let id = event["id"].as_str().ok_or_else(|| anyhow!("the event has no id"))?;
        std::fs::create_dir_all(&self.shared.dir)?;
        let tmp = self.shared.dir.join(format!("{id}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec(event)?)?;
        std::fs::rename(&tmp, self.shared.dir.join(format!("{id}.json")))?;
        let (lock, cv) = &self.shared.wake;
        *lock.lock().unwrap() = true;
        cv.notify_all();
        Ok(())
    }
}

fn pending(s: &Shared) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(&s.dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn flush(s: &Shared) -> Result<()> {
    for path in pending(s) {
        let event: Value = match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()) {
            Some(v) => v,
            None => {
                let _ = std::fs::remove_file(&path);
                continue;
            }
        };
        let old = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|m| m.elapsed().ok())
            .is_some_and(|age| age > MAX_AGE);
        if old {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        // offline, the service asleep, ...: retry later, under the same id
        send(&s.base, &event)?;
        let _ = std::fs::remove_file(&path);
        *s.last_error.lock().unwrap() = None;
    }
    Ok(())
}

fn send(base: &str, event: &Value) -> Result<()> {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(90)).build();
    let resp: Value = agent
        .post(&format!("{base}/call/v2/save_feedback"))
        .timeout(Duration::from_secs(30))
        .send_json(serde_json::json!({"data": event}))?
        .into_json()?;
    let event_id = resp["event_id"].as_str().ok_or_else(|| anyhow!("the feedback service returned no event_id"))?;
    let stream = agent.get(&format!("{base}/call/save_feedback/{event_id}")).call()?.into_reader();
    let (mut name, mut data): (Option<String>, Option<String>) = (None, None);
    for line in std::io::BufReader::new(stream).lines() {
        let line = line?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            match name.as_deref() {
                Some("error") => bail!("the feedback service reported an error"),
                Some("complete") => {
                    let result: Value = serde_json::from_str(data.as_deref().unwrap_or("null")).unwrap_or(Value::Null);
                    let ok = result.as_array().and_then(|a| a.first()).and_then(|r| r["success"].as_bool()) == Some(true);
                    if !ok {
                        bail!("the feedback service did not confirm saving");
                    }
                    return Ok(());
                }
                _ => {}
            }
            name = None;
            data = None;
            continue;
        }
        if let Some(n) = line.strip_prefix("event:") {
            name = Some(n.trim().to_string());
        } else if let Some(d) = line.strip_prefix("data:") {
            data = Some(d.trim().to_string());
        }
    }
    bail!("the stream ended without a save confirmation")
}

/// `2026-11-03T10:24:00.123Z`.
pub fn utc_iso(t: SystemTime) -> String {
    let d = t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil from days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(utc_iso(SystemTime::UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        let t = SystemTime::UNIX_EPOCH + Duration::from_millis(1_793_697_295_502);
        assert_eq!(utc_iso(t), "2026-11-03T09:14:55.502Z"); // Python: time.gmtime
    }
}

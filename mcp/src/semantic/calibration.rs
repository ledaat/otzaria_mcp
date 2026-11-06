//! Calibrated defaults for the served model: `defaults-<slug>.json`, written by
//! otzaria-sources' `calibrate_semantic.py` (`run` + `report`).
//!
//! The file is looked up by the vector set's model family, slugged as the suite slugs it
//! (`otzaria/Meivin-Embed-QA491-native256` → `defaults-otzaria-Meivin-Embed-QA491-native256.json`),
//! in the calibration folder: `--calibration-dir` (or `OTZARIA_CALIBRATION_DIR`), else
//! `calibration/` beside the builder (`--builder`) or the plugin's data.json, else beside the
//! personal index. It is read again when it changes, and dropped when the model changes and
//! no file matches the new one.
//!
//! What it sets, each only when the request leaves it out:
//! - `min_score` by query length (the suite's buckets: up to 8 words, 9–30, 31 and more);
//!   `/parallels` takes `parallels.min_score`.
//! - the long-line window penalty of every scan ([`window_penalty`]).
//! - `confidence` on each hit: P(a true answer) at the hit's score, from the suite's isotonic
//!   table for the query's bucket.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde_json::{json, Value};

/// The scans' penalty on a later window of a long line (Otzaria's default until a calibration
/// file sets another).
static WINDOW_PENALTY_BITS: AtomicU32 = AtomicU32::new(0x3D23_D70A); // 0.04f32

pub const DEFAULT_WINDOW_PENALTY: f32 = 0.04;

pub fn window_penalty() -> f32 {
    f32::from_bits(WINDOW_PENALTY_BITS.load(Ordering::Relaxed))
}

fn set_window_penalty(p: f32) {
    WINDOW_PENALTY_BITS.store(p.to_bits(), Ordering::Relaxed);
}

/// The suite's query-length buckets, in words (`calibrate_semantic.BUCKETS`).
pub fn bucket_of(words: usize) -> &'static str {
    match words {
        0..=8 => "short",
        9..=30 => "medium",
        _ => "long",
    }
}

/// `calibrate_semantic.slugify`.
pub fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    let t: String = out.trim_matches('-').chars().take(80).collect();
    if t.is_empty() {
        "model".into()
    } else {
        t
    }
}

pub fn file_name(model_family: &str) -> String {
    format!("defaults-{}.json", slugify(model_family))
}

/// One bucket's calibrated values.
#[derive(Debug, Clone, Default)]
pub struct Bucket {
    pub min_score: Option<f32>,
    /// (score, P(true answer)), by score.
    pub confidence: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Calibration {
    pub file: PathBuf,
    pub mtime: Option<SystemTime>,
    pub model: String,
    pub short: Bucket,
    pub medium: Bucket,
    pub long: Bucket,
    pub parallels_min_score: Option<f32>,
    pub window_penalty: Option<f32>,
    pub library_version: Option<Value>,
    pub run_created: Option<String>,
}

fn num(v: Option<&Value>) -> Option<f32> {
    v.and_then(Value::as_f64)
        .map(|x| x as f32)
        .filter(|x| x.is_finite())
}

impl Calibration {
    pub fn load(file: &Path) -> Result<Calibration> {
        let text =
            std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let v: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", file.display()))?;
        Ok(Self::from_value(file, &v))
    }

    pub fn from_value(file: &Path, v: &Value) -> Calibration {
        let bucket = |name: &str| Bucket {
            min_score: num(v.pointer(&format!("/min_score/{name}"))),
            confidence: v
                .pointer(&format!("/confidence/{name}"))
                .and_then(Value::as_array)
                .map(|rows| {
                    let mut t: Vec<(f32, f32)> = rows
                        .iter()
                        .filter_map(|r| Some((num(r.get(0))?, num(r.get(1))?)))
                        .collect();
                    t.sort_by(|a, b| a.0.total_cmp(&b.0));
                    t
                })
                .unwrap_or_default(),
        };
        Calibration {
            file: file.to_path_buf(),
            mtime: std::fs::metadata(file).and_then(|m| m.modified()).ok(),
            model: v
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            short: bucket("short"),
            medium: bucket("medium"),
            long: bucket("long"),
            parallels_min_score: num(v.pointer("/parallels/min_score")),
            window_penalty: num(v.get("window_penalty")).filter(|p| (0.0..=0.2).contains(p)),
            library_version: v.get("library_version").cloned().filter(|x| !x.is_null()),
            run_created: v
                .get("run_created")
                .and_then(Value::as_str)
                .map(String::from),
        }
    }

    pub fn bucket(&self, name: &str) -> &Bucket {
        match name {
            "short" => &self.short,
            "medium" => &self.medium,
            _ => &self.long,
        }
    }

    /// The bucket's min_score; a bucket without one (too few samples) takes its neighbour's.
    pub fn min_score(&self, bucket: &str) -> Option<f32> {
        let order: &[&str] = match bucket {
            "short" => &["short", "medium", "long"],
            "medium" => &["medium", "long", "short"],
            _ => &["long", "medium", "short"],
        };
        order.iter().find_map(|b| self.bucket(b).min_score)
    }

    /// P(a true answer) at `score`, interpolated in the bucket's table.
    pub fn confidence(&self, bucket: &str, score: f32) -> Option<f32> {
        let t = &self.bucket(bucket).confidence;
        let (first, last) = (t.first()?, t.last()?);
        if score <= first.0 {
            return Some(first.1);
        }
        if score >= last.0 {
            return Some(last.1);
        }
        let i = t.partition_point(|p| p.0 <= score);
        let (a, b) = (t[i - 1], t[i]);
        let f = if b.0 > a.0 {
            (score - a.0) / (b.0 - a.0)
        } else {
            1.0
        };
        Some(a.1 + f * (b.1 - a.1))
    }

    /// Make the file's window penalty the scans' (Otzaria's default without one).
    pub fn apply(this: Option<&Calibration>) {
        set_window_penalty(
            this.and_then(|c| c.window_penalty)
                .unwrap_or(DEFAULT_WINDOW_PENALTY),
        );
    }

    /// For /health.
    pub fn summary(this: Option<&Calibration>, looked_for: &[PathBuf]) -> Value {
        match this {
            None => json!({"loaded": false, "looked_for": looked_for,
                           "window_penalty": window_penalty()}),
            Some(c) => json!({
                "loaded": true,
                "file": c.file,
                "model": c.model,
                "library_version": c.library_version,
                "run_created": c.run_created,
                "min_score": {"short": c.short.min_score, "medium": c.medium.min_score,
                              "long": c.long.min_score},
                "parallels_min_score": c.parallels_min_score,
                "window_penalty": window_penalty(),
                "confidence": !(c.short.confidence.is_empty() && c.medium.confidence.is_empty()
                                && c.long.confidence.is_empty()),
            }),
        }
    }
}

/// Where `defaults-<slug>.json` may be, first match wins.
pub fn candidates(dirs: &[PathBuf], model_family: &str) -> Vec<PathBuf> {
    let name = file_name(model_family);
    dirs.iter().map(|d| d.join(&name)).collect()
}

/// Words in a query, as the suite counts them (`str.split()`).
pub fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Calibration {
        Calibration::from_value(
            Path::new("defaults-x.json"),
            &json!({
                "model": "otzaria/Meivin-Embed-QA491-native256",
                "min_score": {"short": 0.61, "medium": 0.55, "long": null},
                "parallels": {"min_score": 0.52},
                "window_penalty": 0.05,
                "confidence": {"short": [[0.5, 0.1], [0.6, 0.5], [0.7, 0.9]], "medium": [], "long": []},
            }),
        )
    }

    #[test]
    fn slug_matches_the_suite() {
        assert_eq!(
            slugify("otzaria/Meivin-Embed-QA491-native256"),
            "otzaria-Meivin-Embed-QA491-native256"
        );
        assert_eq!(
            slugify("pardes-student@695cbe2/pardes-clean-1"),
            "pardes-student-695cbe2-pardes-clean-1"
        );
        assert_eq!(slugify("//"), "model");
        assert_eq!(slugify("a  b"), "a-b");
    }

    #[test]
    fn buckets_match_the_suite() {
        assert_eq!(bucket_of(0), "short");
        assert_eq!(bucket_of(8), "short");
        assert_eq!(bucket_of(9), "medium");
        assert_eq!(bucket_of(30), "medium");
        assert_eq!(bucket_of(31), "long");
    }

    #[test]
    fn min_score_falls_back_to_a_neighbour() {
        let c = sample();
        assert_eq!(c.min_score("short"), Some(0.61));
        assert_eq!(c.min_score("long"), Some(0.55));
        assert_eq!(c.parallels_min_score, Some(0.52));
    }

    #[test]
    fn confidence_interpolates_and_clamps() {
        let c = sample();
        assert_eq!(c.confidence("short", 0.4), Some(0.1));
        assert!((c.confidence("short", 0.65).unwrap() - 0.7).abs() < 1e-5);
        assert_eq!(c.confidence("short", 0.9), Some(0.9));
        assert_eq!(c.confidence("medium", 0.6), None);
    }

    #[test]
    fn window_penalty_is_read_and_bounded() {
        assert_eq!(sample().window_penalty, Some(0.05));
        let wild = Calibration::from_value(Path::new("d.json"), &json!({"window_penalty": 0.9}));
        assert_eq!(wild.window_penalty, None);
        assert_eq!(f32::from_bits(0x3D23_D70A), DEFAULT_WINDOW_PENALTY);
    }

    /// A file calibrate_semantic.py wrote (metrics left out), read from disk as the server does.
    #[test]
    fn reads_a_suite_file() {
        let dir = tempfile::tempdir().unwrap();
        let family = "otzaria/Meivin-Embed-QA491-native256";
        let path = candidates(&[dir.path().to_path_buf()], family).remove(0);
        assert!(path.ends_with("defaults-otzaria-Meivin-Embed-QA491-native256.json"));
        std::fs::write(&path, REAL).unwrap();
        let c = Calibration::load(&path).unwrap();
        assert_eq!(c.model, family);
        assert_eq!(c.min_score("short"), Some(0.745));
        assert_eq!(c.min_score("medium"), Some(0.717));
        assert_eq!(c.min_score("long"), Some(0.675));
        assert_eq!(c.parallels_min_score, Some(0.675));
        assert_eq!(c.window_penalty, Some(0.04));
        assert_eq!(c.library_version, Some(json!("36")));
        assert_eq!(c.confidence("long", 0.5), Some(0.0));
        assert!((c.confidence("long", 0.69).unwrap() - 0.665).abs() < 1e-3);
        assert_eq!(c.confidence("short", 0.9), Some(1.0));
        let h = Calibration::summary(Some(&c), &[]);
        assert_eq!(h["loaded"], true);
        assert_eq!(h["confidence"], true);
    }

    const REAL: &str = r#"{"model": "otzaria/Meivin-Embed-QA491-native256", "backend": "server", "slug": "otzaria-Meivin-Embed-QA491-native256", "library_version": "36", "evalset_created": "2026-11-06T10:20:09", "run_created": "2026-11-06T10:24:26", "min_score": {"short": 0.745, "medium": 0.717, "long": 0.675}, "obsidian": {"minScore": 0.675, "searchMinScore": {"short": 0.745, "medium": 0.717}}, "parallels": {"min_score": 0.675}, "window_penalty": 0.04, "confidence": {"short": [[0.48, 0.0], [0.5, 0.0], [0.52, 0.07], [0.58, 0.07], [0.6, 0.178], [0.64, 0.178], [0.66, 0.474], [0.68, 0.612], [0.7, 0.655], [0.72, 0.742], [0.78, 0.742], [0.8, 1.0], [0.88, 1.0]], "medium": [[0.52, 0.053], [0.56, 0.053], [0.58, 0.066], [0.6, 0.066], [0.62, 0.13], [0.64, 0.13], [0.66, 0.472], [0.68, 0.604], [0.7, 0.604], [0.72, 0.698], [0.74, 0.729], [0.76, 1.0], [1.0, 1.0]], "long": [[0.52, 0.0], [0.62, 0.0], [0.64, 0.05], [0.66, 0.05], [0.68, 0.536], [0.7, 0.794], [0.72, 1.0], [1.0, 1.0]]}, "metrics": {}}"#;
}

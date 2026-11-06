//! The personal index built by otzaria-sources' `build_personal_index.py`: vectors of the
//! personal `.txt` books and of the attached book databases, in the model of Otzaria's own
//! vectors. A set is flat files (`set-NNNNNN/`): `vectors.i8` [slots × dim] with one scale
//! per vector (`scales.f32`, Otzaria's i8-sym-vec codec), `slot_book.i32`, `slot_line.i32`,
//! `slot_window.u8` and `books.json`; `index.json` names the live set and its model.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use rusqlite::types::Value;
use serde::Deserialize;

use crate::text::LineCodec;

use super::calibration::window_penalty;
use super::catalog::{connect_ro, has_table};
use super::textutil::SourceKey;

#[derive(Debug, Clone, Deserialize)]
pub struct PersonalBook {
    #[serde(default = "default_kind")]
    pub kind: String,
    pub book_id: i64,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub category: Vec<String>,
    /// personal .txt books
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub mtime_ns: Option<i128>,
    /// attached databases
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub lib: Option<String>,
    #[serde(default)]
    pub db: Option<String>,
    #[serde(default)]
    pub hash: Option<String>,
}

fn default_kind() -> String {
    "user".into()
}

impl PersonalBook {
    pub fn attached(&self) -> bool {
        self.kind == "attached"
    }

    pub fn source(&self) -> SourceKey {
        if self.attached() {
            SourceKey::Attached(self.slug.clone().unwrap_or_default())
        } else {
            SourceKey::User
        }
    }

    pub fn url(&self, line: i64) -> String {
        self.source().url(self.book_id, line)
    }

    /// The category path shown with a hit: the database's name first for an attached book.
    pub fn category_path(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.attached() {
            if let Some(l) = &self.lib {
                out.push(l.clone());
            }
        }
        out.extend(self.category.iter().cloned());
        out
    }
}

/// `os.path.normcase(os.path.abspath(p))`: case-folded with backslashes on Windows.
pub fn norm_path(p: &str) -> String {
    if cfg!(windows) || p.contains('\\') || p.get(1..2) == Some(":") {
        p.replace('/', "\\").to_lowercase()
    } else {
        p.to_string()
    }
}

struct Cached {
    sig: (u64, Option<SystemTime>),
    lines: std::sync::Arc<Vec<String>>,
    stale: bool,
}

pub struct PersonalSet {
    pub dir: PathBuf,
    pub mtime: Option<SystemTime>,
    pub info: serde_json::Value,
    pub n: usize,
    pub dim: usize,
    vectors: Option<Mmap>,
    scales: Option<Mmap>,
    slot_book: Option<Mmap>,
    slot_line: Option<Mmap>,
    slot_window: Option<Mmap>,
    pub books: Vec<PersonalBook>,
    pub by_key: HashMap<(SourceKey, i64), usize>,
    cache: Mutex<Vec<(usize, Cached)>>,
}

fn map(path: &Path) -> Result<Mmap> {
    let f = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    // SAFETY: the builder writes a new set directory and switches index.json to it; a set's
    // files are never modified in place while mapped.
    unsafe { Mmap::map(&f) }.with_context(|| format!("cannot map {}", path.display()))
}

impl PersonalSet {
    pub fn index_json(dir: &Path) -> PathBuf {
        dir.join("index.json")
    }

    pub fn open(dir: &Path, user_books_db: Option<&Path>) -> Result<PersonalSet> {
        let ij = Self::index_json(dir);
        let mtime = std::fs::metadata(&ij).and_then(|m| m.modified()).ok();
        let info: serde_json::Value = serde_json::from_slice(&std::fs::read(&ij)?)?;
        let set = dir.join(
            info["set"]
                .as_str()
                .ok_or_else(|| anyhow!("index.json names no set"))?,
        );
        let n = info["slots"].as_u64().unwrap_or(0) as usize;
        let dim = info["dim"].as_u64().unwrap_or(256) as usize;
        let (vectors, scales, slot_book, slot_line, slot_window) = if n > 0 {
            let sw = set.join("slot_window.u8");
            (
                Some(map(&set.join("vectors.i8"))?),
                Some(map(&set.join("scales.f32"))?),
                Some(map(&set.join("slot_book.i32"))?),
                Some(map(&set.join("slot_line.i32"))?),
                if sw.exists() { Some(map(&sw)?) } else { None },
            )
        } else {
            (None, None, None, None, None)
        };
        if let Some(v) = &vectors {
            if v.len() < n * dim {
                return Err(anyhow!(
                    "{}: vectors.i8 is shorter than {n} × {dim}",
                    set.display()
                ));
            }
        }
        let mut books: Vec<PersonalBook> =
            serde_json::from_slice(&std::fs::read(set.join("books.json"))?)?;
        // Otzaria may renumber personal books; links follow the file
        if let Some(db) = user_books_db.filter(|p| p.exists()) {
            if let Ok(ids) = user_book_ids(db) {
                for b in books.iter_mut().filter(|b| !b.attached()) {
                    if let Some(p) = &b.path {
                        if let Some(id) = ids.get(&norm_path(p)) {
                            b.book_id = *id;
                        }
                    }
                }
            }
        }
        let by_key = books
            .iter()
            .enumerate()
            .map(|(i, b)| ((b.source(), b.book_id), i))
            .collect();
        Ok(PersonalSet {
            dir: dir.to_path_buf(),
            mtime,
            info,
            n,
            dim,
            vectors,
            scales,
            slot_book,
            slot_line,
            slot_window,
            books,
            by_key,
            cache: Mutex::new(Vec::new()),
        })
    }

    pub fn model(&self) -> Option<&str> {
        self.info["model"].as_str()
    }

    fn i32_at(m: &Option<Mmap>, slot: usize) -> i32 {
        let b = &m.as_ref().unwrap()[slot * 4..slot * 4 + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// (book index, line index) of a slot.
    pub fn record(&self, slot: usize) -> (usize, i64) {
        (
            Self::i32_at(&self.slot_book, slot) as usize,
            Self::i32_at(&self.slot_line, slot) as i64,
        )
    }

    pub fn slot_window(&self, slot: usize) -> u8 {
        self.slot_window.as_ref().map_or(0, |w| w[slot])
    }

    fn scale(&self, slot: usize) -> f32 {
        let b = &self.scales.as_ref().unwrap()[slot * 4..slot * 4 + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn codes(&self, slot: usize) -> &[i8] {
        let v = self.vectors.as_ref().unwrap();
        let b = &v[slot * self.dim..(slot + 1) * self.dim];
        // SAFETY: i8 and u8 have the same layout.
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) }
    }

    /// Which books a category scope admits (`None`: all), by the titles of their category
    /// path and their database's name, as the Python server matched them.
    pub fn book_mask(&self, include: &[String], exclude: &[String]) -> Option<Vec<bool>> {
        if include.is_empty() && exclude.is_empty() {
            return None;
        }
        let inc: HashSet<&str> = include.iter().map(|s| s.trim()).collect();
        let exc: HashSet<&str> = exclude.iter().map(|s| s.trim()).collect();
        Some(
            self.books
                .iter()
                .map(|b| {
                    let mut t: HashSet<&str> = b.category.iter().map(|s| s.trim()).collect();
                    if let Some(l) = &b.lib {
                        t.insert(l.trim());
                    }
                    (inc.is_empty() || t.iter().any(|x| inc.contains(x)))
                        && !t.iter().any(|x| exc.contains(x))
                })
                .collect(),
        )
    }

    /// The `kc` best slots for each query, exact over every admitted slot: an int8 pass
    /// (each query quantized to int8, as Otzaria's scanner does) keeps `kc` per query, which
    /// are then rescored exactly; a slot holding a later window of a long line scores
    /// [`window_penalty`] lower. Scores are cosine similarities.
    pub fn scan(
        &self,
        queries: &[Vec<f32>],
        kc: usize,
        books: Option<&[bool]>,
    ) -> Vec<Vec<(f32, usize)>> {
        let m = queries.len();
        if self.n == 0 || m == 0 || kc == 0 {
            return vec![Vec::new(); m];
        }
        let dim = self.dim;
        let quant: Vec<(Vec<i8>, f32)> = queries
            .iter()
            .map(|q| {
                let mut qs = q.iter().fold(0f32, |a, &x| a.max(x.abs())) / 127.0;
                if qs == 0.0 {
                    qs = 1.0;
                }
                (
                    q.iter()
                        .map(|&x| (x / qs).round().clamp(-127.0, 127.0) as i8)
                        .collect(),
                    qs,
                )
            })
            .collect();
        let chunk = 1 << 16;
        let n_chunks = self.n.div_ceil(chunk);
        let partial: Vec<Vec<Vec<(f32, usize)>>> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let a = c * chunk;
                let b = (a + chunk).min(self.n);
                let mut heaps: Vec<TopK> = (0..m).map(|_| TopK::new(kc)).collect();
                for slot in a..b {
                    if let Some(mask) = books {
                        let (bi, _) = self.record(slot);
                        if !mask.get(bi).copied().unwrap_or(false) {
                            continue;
                        }
                    }
                    let codes = self.codes(slot);
                    let s = self.scale(slot);
                    let pen = if self.slot_window(slot) > 0 {
                        window_penalty()
                    } else {
                        0.0
                    };
                    for (qi, (qc, qs)) in quant.iter().enumerate() {
                        let dot: i32 = codes
                            .iter()
                            .zip(qc)
                            .map(|(&x, &y)| x as i32 * y as i32)
                            .sum();
                        heaps[qi].push(dot as f32 * s * qs - pen, slot);
                    }
                }
                heaps.into_iter().map(|h| h.into_vec()).collect()
            })
            .collect();
        (0..m)
            .map(|qi| {
                let mut all: Vec<(f32, usize)> =
                    partial.iter().flat_map(|p| p[qi].iter().copied()).collect();
                all.sort_by(|x, y| y.0.total_cmp(&x.0));
                all.truncate(kc);
                // exact rescore of what the int8 pass kept
                let q = &queries[qi];
                for e in all.iter_mut() {
                    let codes = self.codes(e.1);
                    let dot: f32 = codes
                        .iter()
                        .zip(q)
                        .map(|(&x, &y)| x as f32 * y)
                        .sum::<f32>();
                    let pen = if self.slot_window(e.1) > 0 {
                        window_penalty()
                    } else {
                        0.0
                    };
                    e.0 = dot * self.scale(e.1) - pen;
                }
                all.sort_by(|x, y| y.0.total_cmp(&x.0));
                let _ = dim;
                all
            })
            .collect()
    }

    /// The exact score of one slot against `q` (cosine, the long-window penalty taken off),
    /// as [`Self::scan`] rescores what it keeps.
    pub fn score(&self, q: &[f32], slot: usize) -> f32 {
        if slot >= self.n {
            return -1.0;
        }
        let dot: f32 = self
            .codes(slot)
            .iter()
            .zip(q)
            .map(|(&x, &y)| x as f32 * y)
            .sum();
        let pen = if self.slot_window(slot) > 0 {
            window_penalty()
        } else {
            0.0
        };
        dot * self.scale(slot) - pen
    }

    /// The raw lines of a book (a small cache), and whether its text changed since indexing.
    pub fn lines_of(&self, bi: usize) -> (Option<std::sync::Arc<Vec<String>>>, bool) {
        let Some(b) = self.books.get(bi) else {
            return (None, true);
        };
        let sig = match signature(b) {
            Some(s) => s,
            None => return (None, true),
        };
        {
            let cache = self.cache.lock().unwrap();
            if let Some((_, c)) = cache.iter().find(|(i, c)| *i == bi && c.sig == sig) {
                return (Some(c.lines.clone()), c.stale);
            }
        }
        let lines = match read_book(b) {
            Ok(l) => std::sync::Arc::new(l),
            Err(_) => return (None, true),
        };
        let stale = match &b.hash {
            Some(h) => text_hash(&lines).0 != *h,
            None => Some(sig.0) != b.size,
        };
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|(i, _)| *i != bi);
        cache.push((
            bi,
            Cached {
                sig,
                lines: lines.clone(),
                stale,
            },
        ));
        if cache.len() > 24 {
            cache.remove(0);
        }
        (Some(lines), stale)
    }
}

/// Top-k by score, keeping the first of equal scores.
struct TopK {
    k: usize,
    items: Vec<(f32, usize)>,
    min: f32,
}

impl TopK {
    fn new(k: usize) -> TopK {
        TopK {
            k,
            items: Vec::with_capacity(k * 2),
            min: f32::NEG_INFINITY,
        }
    }

    fn push(&mut self, s: f32, slot: usize) {
        if self.items.len() >= self.k && s <= self.min {
            return;
        }
        self.items.push((s, slot));
        if self.items.len() >= self.k * 2 {
            self.compact();
        }
    }

    fn compact(&mut self) {
        self.items.sort_by(|x, y| y.0.total_cmp(&x.0));
        self.items.truncate(self.k);
        self.min = self.items.last().map_or(f32::NEG_INFINITY, |x| x.0);
    }

    fn into_vec(mut self) -> Vec<(f32, usize)> {
        self.compact();
        self.items
    }
}

fn signature(b: &PersonalBook) -> Option<(u64, Option<SystemTime>)> {
    let p = if b.attached() {
        b.db.as_deref()?
    } else {
        b.path.as_deref()?
    };
    let md = std::fs::metadata(p).ok()?;
    Some((md.len(), md.modified().ok()))
}

/// `decode_book`: UTF-8 when valid (BOM dropped), UTF-16 by BOM, else Windows-1255.
pub fn decode_book(data: &[u8]) -> String {
    let data = data.strip_prefix(b"\xef\xbb\xbf").unwrap_or(data);
    if let Ok(s) = std::str::from_utf8(data) {
        return s.to_string();
    }
    if data.starts_with(b"\xff\xfe") || data.starts_with(b"\xfe\xff") {
        let (s, _, _) = encoding_rs::UTF_16LE.decode(data);
        return s.into_owned();
    }
    let (s, _, _) = encoding_rs::WINDOWS_1255.decode(data);
    s.into_owned()
}

pub fn read_book(b: &PersonalBook) -> Result<Vec<String>> {
    if b.attached() {
        let db = b.db.as_deref().ok_or_else(|| anyhow!("no database path"))?;
        return attached_book_lines(Path::new(db), b.book_id);
    }
    let path = b.path.as_deref().ok_or_else(|| anyhow!("no file path"))?;
    let data = std::fs::read(path)?;
    Ok(decode_book(&data).split('\n').map(String::from).collect())
}

/// Every line of a book of an Otzaria-format database, by line index (gaps empty).
pub fn attached_book_lines(db: &Path, book: i64) -> Result<Vec<String>> {
    let conn = match connect_ro(db, false) {
        Ok(c)
            if c.query_row("SELECT count(*) FROM book", [], |_| Ok(()))
                .is_ok() =>
        {
            c
        }
        _ => connect_ro(db, true)?,
    };
    let compressed = has_table(&conn, "line_content")? && {
        let mut stmt = conn.prepare("PRAGMA table_info(line)")?;
        let cols: HashSet<String> = stmt
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<rusqlite::Result<_>>()?;
        !cols.contains("content")
    };
    let mut codec = LineCodec::load(&conn)?;
    let sql = if compressed {
        "SELECT l.lineIndex, c.content FROM line l JOIN line_content c ON c.id = l.id WHERE l.bookId = ?1 ORDER BY l.lineIndex"
    } else {
        "SELECT l.lineIndex, l.content FROM line l WHERE l.bookId = ?1 ORDER BY l.lineIndex"
    };
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([book])?;
    let mut out: Vec<String> = Vec::new();
    while let Some(r) = rows.next()? {
        let li: i64 = r.get(0)?;
        let v: Value = r.get(1)?;
        let text = codec.text_lossy(&v);
        let li = li.max(0) as usize;
        if li == out.len() {
            out.push(text);
        } else if li > out.len() {
            out.resize(li, String::new());
            out.push(text);
        } else {
            out[li] = text;
        }
    }
    Ok(out)
}

/// `text_hash`: sha1 of the book's text, CR and trailing empty lines ignored.
pub fn text_hash(lines: &[String]) -> (String, usize) {
    let mut ls: Vec<&str> = lines.iter().map(|x| x.trim_end_matches('\r')).collect();
    while ls.last().is_some_and(|l| l.trim().is_empty()) {
        ls.pop();
    }
    let data = ls.join("\n");
    let mut h = sha1_smol::Sha1::new();
    h.update(data.as_bytes());
    (h.digest().to_string(), data.len())
}

fn user_book_ids(db: &Path) -> Result<HashMap<String, i64>> {
    let conn = match connect_ro(db, false) {
        Ok(c)
            if c.query_row("SELECT count(*) FROM book", [], |_| Ok(()))
                .is_ok() =>
        {
            c
        }
        _ => connect_ro(db, true)?,
    };
    let mut stmt = conn.prepare("SELECT id, filePath FROM book WHERE filePath IS NOT NULL")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
    let mut out = HashMap::new();
    for row in rows {
        let (id, fp) = row?;
        out.insert(norm_path(&fp), id);
    }
    Ok(out)
}

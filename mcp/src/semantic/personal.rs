//! The personal index built by otzaria-sources' `build_personal_index.py`: vectors of the
//! personal `.txt` books and of the attached book databases, in the model of Otzaria's own
//! vectors. A set is flat files (`set-NNNNNN/`): `vectors.i8` [slots × dim] with one scale
//! per vector (`scales.f32`, Otzaria's i8-sym-vec codec), `slot_book.i32`, `slot_line.i32`,
//! `slot_window.u8` and `books.json`; `index.json` names the live set and its model.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use rusqlite::types::Value;
use serde::Deserialize;

use crate::text::LineCodec;

use super::calibration::window_penalty;
use super::catalog::{connect_ro, has_table};
use super::textutil::{HeadingIndex, SourceKey};

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

    /// The book's name in Otzaria's search index and vector set (its `filePath` there).
    pub fn set_name(&self) -> String {
        if self.attached() {
            format!("db:{}:{}", self.slug.as_deref().unwrap_or(""), self.book_id)
        } else {
            format!("uid:{}", self.book_id)
        }
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

/// A personal or database book's raw lines, with its headings indexed on first use.
pub struct BookText {
    pub lines: Vec<String>,
    heads: OnceLock<HeadingIndex>,
    bytes: usize,
}

impl BookText {
    fn new(lines: Vec<String>) -> BookText {
        let bytes = lines.iter().map(|l| l.len() + 24).sum::<usize>() + 64;
        BookText {
            lines,
            heads: OnceLock::new(),
            bytes,
        }
    }

    /// The headings in force at line `upto`, outermost first ([`heading_path`] of the lines,
    /// without rescanning the book).
    ///
    /// [`heading_path`]: super::textutil::heading_path
    pub fn heading_path(&self, upto: usize) -> Vec<String> {
        self.heads
            .get_or_init(|| HeadingIndex::build(&self.lines))
            .path(upto)
    }
}

impl std::ops::Deref for BookText {
    type Target = Vec<String>;
    fn deref(&self) -> &Vec<String> {
        &self.lines
    }
}

/// The book text cache's budget: the books most recently read, up to this many bytes of text
/// (and at least the one just read). A search that touches many large database books (a
/// quotation checked at each of its places) no longer reads the same book again and again.
const CACHE_BYTES: usize = 384 << 20;
const CACHE_BOOKS: usize = 256;

struct Cached {
    sig: (u64, Option<SystemTime>),
    text: Arc<BookText>,
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
    /// Each book by the name Otzaria's search index and vector set know it by (`uid:<id>`,
    /// `db:<slug>:<id>`), with the id it had when the index was built: the id
    /// `oxv_export.py` wrote into Otzaria's set from this index.
    pub by_set_name: HashMap<String, usize>,
    cache: Mutex<Vec<(usize, Cached)>>,
    /// Open handles of attached databases, by file: see [`DbPool`].
    dbs: Mutex<HashMap<String, DbPool>>,
    /// Headings and staleness of attached-database books, by book: see [`BookMeta`].
    metas: Mutex<HashMap<usize, (FileSig, Arc<BookMeta>)>>,
}

/// A file's size and mtime: a book or database read is valid while they stay.
type FileSig = (u64, Option<SystemTime>);

/// What showing a line of an attached-database book needs beyond its own lines: the
/// headings in force at each line, and whether the book changed since it was indexed. Built
/// from one read of the whole book and kept (it is small); the lines themselves are read a
/// few at a time ([`PersonalSet::rows`]). Database books can be tens of megabytes of text each,
/// and a search touches dozens: kept whole they did not fit any cache.
pub struct BookMeta {
    heads: HeadingIndex,
    pub stale: bool,
}

/// An open attached database: its connection and its line decompressor (every zstd
/// dictionary loaded), kept between books. Opening one and loading its dictionaries for each
/// book read was most of what a first read of a short database book cost.
struct DbHandle {
    conn: rusqlite::Connection,
    codec: LineCodec,
    compressed: bool,
}

/// The idle handles of one database file, valid while the file keeps its size and mtime.
/// A handle is taken out for one read and put back, so books of one database can be read in
/// parallel ([`PersonalSet::prefetch`]).
struct DbPool {
    sig: (u64, Option<SystemTime>),
    idle: Vec<DbHandle>,
}

/// The most idle handles kept per database.
const DB_IDLE: usize = 16;

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

    /// `map_vectors: false` opens the set for its books only (their titles, files and
    /// text): nothing of its vectors is mapped, and [`Self::scan`] finds nothing.
    pub fn open(
        dir: &Path,
        user_books_db: Option<&Path>,
        map_vectors: bool,
    ) -> Result<PersonalSet> {
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
        let (vectors, scales, slot_book, slot_line, slot_window) = if n > 0 && map_vectors {
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
        let by_set_name = books
            .iter()
            .enumerate()
            .map(|(i, b)| (b.set_name(), i))
            .collect();
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
            by_set_name,
            cache: Mutex::new(Vec::new()),
            dbs: Mutex::new(HashMap::new()),
            metas: Mutex::new(HashMap::new()),
        })
    }

    pub fn model(&self) -> Option<&str> {
        self.info["model"].as_str()
    }

    /// Whether the set's vectors are mapped (see [`Self::open`]).
    pub fn has_vectors(&self) -> bool {
        self.vectors.is_some()
    }

    /// The names in Otzaria's vector set of the books a scope admits (`mask` as
    /// [`Self::book_mask`] gives it; `None`: every book).
    pub fn set_names<'a>(&'a self, mask: Option<&'a [bool]>) -> impl Iterator<Item = &'a str> {
        self.by_set_name
            .iter()
            .filter(move |(_, &i)| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false)))
            .map(|(name, _)| name.as_str())
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
        if self.n == 0 || m == 0 || kc == 0 || self.vectors.is_none() {
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
        if slot >= self.n || self.vectors.is_none() {
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

    /// The raw lines of a book, and whether its text changed since indexing. Kept in a cache
    /// of the books most recently used (see [`CACHE_BYTES`]).
    pub fn lines_of(&self, bi: usize) -> (Option<Arc<BookText>>, bool) {
        let Some(b) = self.books.get(bi) else {
            return (None, true);
        };
        let sig = match signature(b) {
            Some(s) => s,
            None => return (None, true),
        };
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(pos) = cache.iter().position(|(i, c)| *i == bi && c.sig == sig) {
                // most recently used last
                let e = cache.remove(pos);
                let out = (Some(e.1.text.clone()), e.1.stale);
                cache.push(e);
                return out;
            }
        }
        let lines = match self.read_book(b) {
            Ok(l) => l,
            Err(_) => return (None, true),
        };
        let stale = match &b.hash {
            Some(h) => text_hash(&lines).0 != *h,
            None => Some(sig.0) != b.size,
        };
        let text = Arc::new(BookText::new(lines));
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|(i, _)| *i != bi);
        cache.push((
            bi,
            Cached {
                sig,
                text: text.clone(),
                stale,
            },
        ));
        let mut total: usize = cache.iter().map(|(_, c)| c.text.bytes).sum();
        while cache.len() > 1 && (total > CACHE_BYTES || cache.len() > CACHE_BOOKS) {
            total -= cache.remove(0).1.text.bytes;
        }
        (Some(text), stale)
    }

    /// Ready what showing lines of the books `bis` names will need, in parallel, so a request
    /// that shows lines of many books reads them all at once instead of one after another:
    /// a personal file's text, and with `headings`, an attached book's [`BookMeta`] (its lines
    /// are read a few at a time anyway).
    pub fn prefetch(&self, bis: impl IntoIterator<Item = usize>, headings: bool) {
        let mut want: Vec<usize> = bis
            .into_iter()
            .filter(|&bi| {
                self.books
                    .get(bi)
                    .is_some_and(|b| headings || !b.attached())
            })
            .collect();
        want.sort_unstable();
        want.dedup();
        {
            let cache = self.cache.lock().unwrap();
            let metas = self.metas.lock().unwrap();
            want.retain(|bi| {
                if self.books[*bi].attached() {
                    !metas.contains_key(bi)
                } else {
                    !cache.iter().any(|(i, _)| i == bi)
                }
            });
        }
        let ready = |bi: usize| {
            if self.books[bi].attached() {
                self.meta(bi);
            } else {
                self.lines_of(bi);
            }
        };
        match want.len() {
            0 => {}
            1 => ready(want[0]),
            _ => want.par_iter().for_each(|&bi| ready(bi)),
        }
    }

    /// Lines `lo..=hi` of a book that exist, with their indexes: an attached book's from its
    /// database (only those lines), a personal file's from its cached text.
    pub fn rows(&self, bi: usize, lo: usize, hi: usize) -> Option<Vec<(usize, String)>> {
        let b = self.books.get(bi)?;
        if !b.attached() {
            let (text, _) = self.lines_of(bi);
            let text = text?;
            let hi = hi.min(text.len().saturating_sub(1));
            if text.is_empty() || lo > hi {
                return Some(Vec::new());
            }
            return Some((lo..=hi).map(|li| (li, text[li].clone())).collect());
        }
        self.with_db(b, |h| read_attached_range(h, b.book_id, lo, hi))
            .ok()
    }

    /// The headings in force at line `li` of a book, outermost first.
    pub fn heading_path(&self, bi: usize, li: usize) -> Vec<String> {
        match self.books.get(bi) {
            Some(b) if b.attached() => self.meta(bi).map_or_else(Vec::new, |m| m.heads.path(li)),
            Some(_) => self
                .lines_of(bi)
                .0
                .map_or_else(Vec::new, |t| t.heading_path(li)),
            None => Vec::new(),
        }
    }

    /// Whether a book's text changed since it was indexed (true when it cannot be read).
    pub fn stale(&self, bi: usize) -> bool {
        match self.books.get(bi) {
            Some(b) if b.attached() => self.meta(bi).is_none_or(|m| m.stale),
            Some(_) => self.lines_of(bi).1,
            None => true,
        }
    }

    /// An attached book's [`BookMeta`], built from one read of the whole book.
    fn meta(&self, bi: usize) -> Option<Arc<BookMeta>> {
        let b = self.books.get(bi)?;
        let sig = signature(b)?;
        if let Some((s, m)) = self.metas.lock().unwrap().get(&bi) {
            if *s == sig {
                return Some(m.clone());
            }
        }
        let lines = self.read_book(b).ok()?;
        let stale = match &b.hash {
            Some(h) => text_hash(&lines).0 != *h,
            None => Some(sig.0) != b.size,
        };
        let m = Arc::new(BookMeta {
            heads: HeadingIndex::build(&lines),
            stale,
        });
        self.metas.lock().unwrap().insert(bi, (sig, m.clone()));
        Some(m)
    }

    /// A book's raw lines: a personal file, or an attached database's book through a pooled
    /// handle of that database.
    fn read_book(&self, b: &PersonalBook) -> Result<Vec<String>> {
        if !b.attached() {
            return read_book(b);
        }
        self.with_db(b, |h| read_attached(h, b.book_id))
    }

    /// Run `f` with a pooled handle of an attached book's database.
    fn with_db<T>(
        &self,
        b: &PersonalBook,
        f: impl FnOnce(&mut DbHandle) -> Result<T>,
    ) -> Result<T> {
        let db = b.db.as_deref().ok_or_else(|| anyhow!("no database path"))?;
        let sig = {
            let md = std::fs::metadata(db)?;
            (md.len(), md.modified().ok())
        };
        let taken = {
            let mut dbs = self.dbs.lock().unwrap();
            let pool = dbs.entry(db.to_string()).or_insert_with(|| DbPool {
                sig,
                idle: Vec::new(),
            });
            if pool.sig != sig {
                // the database changed: its handles may hold a stale schema or dictionaries
                pool.sig = sig;
                pool.idle.clear();
            }
            pool.idle.pop()
        };
        let mut h = match taken {
            Some(h) => h,
            None => open_attached(Path::new(db))?,
        };
        let out = f(&mut h);
        if out.is_ok() {
            let mut dbs = self.dbs.lock().unwrap();
            if let Some(pool) = dbs.get_mut(db) {
                if pool.sig == sig && pool.idle.len() < DB_IDLE {
                    pool.idle.push(h);
                }
            }
        }
        out
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
    read_attached(&mut open_attached(db)?, book)
}

fn open_attached(db: &Path) -> Result<DbHandle> {
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
    let codec = LineCodec::load(&conn)?;
    Ok(DbHandle {
        conn,
        codec,
        compressed,
    })
}

/// Lines `lo..=hi` of an attached book that exist, with their indexes.
fn read_attached_range(
    h: &mut DbHandle,
    book: i64,
    lo: usize,
    hi: usize,
) -> Result<Vec<(usize, String)>> {
    if lo > hi {
        return Ok(Vec::new());
    }
    let sql = if h.compressed {
        "SELECT l.lineIndex, c.content FROM line l JOIN line_content c ON c.id = l.id WHERE l.bookId = ?1 AND l.lineIndex BETWEEN ?2 AND ?3 ORDER BY l.lineIndex"
    } else {
        "SELECT l.lineIndex, l.content FROM line l WHERE l.bookId = ?1 AND l.lineIndex BETWEEN ?2 AND ?3 ORDER BY l.lineIndex"
    };
    let mut stmt = h.conn.prepare_cached(sql)?;
    let mut rows = stmt.query(rusqlite::params![book, lo as i64, hi as i64])?;
    let mut out: Vec<(usize, String)> = Vec::new();
    while let Some(r) = rows.next()? {
        let li = r.get::<_, i64>(0)?.max(0) as usize;
        let v: Value = r.get(1)?;
        let text = h.codec.text_lossy(&v);
        // a line index given twice: the later row, as a read of the whole book keeps it
        match out.last_mut() {
            Some(last) if last.0 == li => last.1 = text,
            _ => out.push((li, text)),
        }
    }
    Ok(out)
}

fn read_attached(h: &mut DbHandle, book: i64) -> Result<Vec<String>> {
    let sql = if h.compressed {
        "SELECT l.lineIndex, c.content FROM line l JOIN line_content c ON c.id = l.id WHERE l.bookId = ?1 ORDER BY l.lineIndex"
    } else {
        "SELECT l.lineIndex, l.content FROM line l WHERE l.bookId = ?1 ORDER BY l.lineIndex"
    };
    let mut stmt = h.conn.prepare_cached(sql)?;
    let mut rows = stmt.query([book])?;
    let mut out: Vec<String> = Vec::new();
    while let Some(r) = rows.next()? {
        let li: i64 = r.get(0)?;
        let v: Value = r.get(1)?;
        let text = h.codec.text_lossy(&v);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::textutil::heading_path;

    /// Books of an attached database are read through pooled handles, in parallel by
    /// `prefetch`, and give the same lines and headings as a read of their own.
    #[test]
    fn attached_books_are_read_through_the_pool_and_prefetched() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("lib.db");
        {
            let c = rusqlite::Connection::open(&db).unwrap();
            c.execute_batch(
                "CREATE TABLE book(id INTEGER PRIMARY KEY, title TEXT);
                 CREATE TABLE line(id INTEGER PRIMARY KEY, bookId INTEGER, lineIndex INTEGER, content TEXT);",
            )
            .unwrap();
            for (bid, n) in [(7i64, 40usize), (9, 25)] {
                c.execute("INSERT INTO book VALUES (?1, 'b')", [bid])
                    .unwrap();
                for li in 0..n {
                    let content = if li % 10 == 0 {
                        format!("<h2>פרק {}</h2>", li / 10)
                    } else {
                        format!("שורה {bid}-{li}")
                    };
                    c.execute(
                        "INSERT INTO line(bookId, lineIndex, content) VALUES (?1, ?2, ?3)",
                        rusqlite::params![bid, li as i64, content],
                    )
                    .unwrap();
                }
            }
        }
        std::fs::create_dir(dir.path().join("set1")).unwrap();
        let books: Vec<serde_json::Value> = [7, 9]
            .iter()
            .map(|id| {
                serde_json::json!({"kind": "attached", "book_id": id, "title": format!("b{id}"),
                                   "slug": "lib", "db": db.to_string_lossy()})
            })
            .collect();
        std::fs::write(
            dir.path().join("set1/books.json"),
            serde_json::to_vec(&books).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("index.json"),
            br#"{"set": "set1", "slots": 0}"#,
        )
        .unwrap();
        let ps = PersonalSet::open(dir.path(), None, false).unwrap();
        // without headings an attached book needs nothing ahead: its lines are read a few
        // at a time
        ps.prefetch([0, 1], false);
        assert!(ps.metas.lock().unwrap().is_empty());
        ps.prefetch([0, 1, 1, 0, 5], true);
        assert_eq!(ps.metas.lock().unwrap().len(), 2);
        // whole attached books are not kept: only their headings and staleness
        assert!(ps.cache.lock().unwrap().is_empty());
        // one handle back in the pool per parallel read at most, never more than DB_IDLE
        let idle = ps.dbs.lock().unwrap()[&*db.to_string_lossy()].idle.len();
        assert!((1..=2).contains(&idle), "{idle} idle handles");
        for (bi, bid, n) in [(0usize, 7i64, 40usize), (1, 9, 25)] {
            let whole = attached_book_lines(&db, bid).unwrap();
            assert_eq!(whole.len(), n);
            let rows = ps.rows(bi, 18, 22).unwrap();
            assert_eq!(
                rows,
                (18..=22).map(|i| (i, whole[i].clone())).collect::<Vec<_>>()
            );
            // past the end: the lines that exist
            assert_eq!(ps.rows(bi, n - 2, n + 5).unwrap().len(), 2);
            assert!(ps.rows(bi, n + 1, n + 5).unwrap().is_empty());
            assert_eq!(ps.rows(bi, 3, 3).unwrap()[0].1, format!("שורה {bid}-3"));
            for li in 0..n + 2 {
                assert_eq!(
                    ps.heading_path(bi, li),
                    heading_path(&whole, li),
                    "line {li}"
                );
            }
            // no hash in books.json: stale when the database's size differs from `size` (none)
            assert!(ps.stale(bi));
        }
    }
}

//! The semantic service: what otzaria-sources' Python server did, over Otzaria's own vectors
//! (the sidecar) and the personal index. Requests and responses are the JSON the Obsidian
//! plugin and the MCP tools already speak (see `http.rs` for the routes).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, bail, Result};
use otzaria_semantic_search::semantic::resolve::{BookSet, RecordRef, SlotRef, VectorHit};
use serde_json::{json, Map, Value};

use super::align::{align_words, Op};
use super::calibration::{bucket_of, words, Calibration};
use super::catalog::{era_label, Catalog, SourceRank, ERA_UNKNOWN};
use super::feedback::Outbox;
use super::official::{Official, OfficialConfig};
use super::personal::PersonalSet;
use super::textutil::{
    char_len, char_prefix, char_slice, clean_markdown, heading_path, line_text, norm_word,
    note_windows, parse_book_url, round_to, strip_data_uris, strip_html, strip_nikud, window_span,
    EchoReference, SourceKey,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A request error the caller made (HTTP 400), as opposed to a failure of the service.
#[derive(Debug)]
pub struct BadRequest(pub String);

impl std::fmt::Display for BadRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BadRequest {}

fn bad(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(BadRequest(msg.into()))
}

pub struct SemanticConfig {
    /// Otzaria's data folder (%APPDATA%\otzaria): vectors, databases, settings, logs.
    pub otzaria_dir: PathBuf,
    pub books_dir: PathBuf,
    pub db: PathBuf,
    pub snapshot: bool,
    pub vectors_dir: PathBuf,
    pub model_dir: Option<PathBuf>,
    pub onnx_runtime: Option<PathBuf>,
    pub scan_threads: Option<usize>,
    pub personal_dir: PathBuf,
    pub user_books_db: PathBuf,
    /// build_personal_index.py, for /personal/status (`--check --json`).
    pub builder: Option<PathBuf>,
    pub python: String,
    /// The feedback outbox (HTTP mode only: the stdio MCP never reaches the network).
    pub feedback_dir: Option<PathBuf>,
    /// Where `defaults-<model>.json` may be (see [`super::calibration`]), first match wins.
    pub calibration_dirs: Vec<PathBuf>,
    /// Where the vectors of personal and database books are read from. `false` (the default):
    /// Otzaria's own vector set, where `oxv_export.py` publishes them — one scan finds
    /// library, personal and database books, and the personal index serves only as the list
    /// of those books (titles, files, text). `true` (`--scan-personal-index`): the personal
    /// index's own vectors are scanned beside Otzaria's set, and whatever the set holds of
    /// books outside the library is ignored.
    pub scan_personal_index: bool,
}

// ── options ─────────────────────────────────────────────────────────────────

pub struct Opts<'a>(pub &'a Map<String, Value>);

impl<'a> Opts<'a> {
    pub fn get(&self, k: &str) -> Option<&Value> {
        self.0.get(k).filter(|v| !v.is_null())
    }
    pub fn int(&self, k: &str, d: i64) -> i64 {
        match self.get(k) {
            Some(Value::Number(n)) => n
                .as_i64()
                .or_else(|| n.as_f64().map(|f| f as i64))
                .unwrap_or(d),
            Some(Value::String(s)) => s.trim().parse().unwrap_or(d),
            Some(Value::Bool(b)) => *b as i64,
            _ => d,
        }
    }
    pub fn float(&self, k: &str, d: f64) -> f64 {
        match self.get(k) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(d),
            Some(Value::String(s)) => s.trim().parse().unwrap_or(d),
            _ => d,
        }
    }
    pub fn str(&self, k: &str) -> String {
        match self.get(k) {
            Some(Value::String(s)) => s.clone(),
            Some(v @ Value::Number(_)) => v.to_string(),
            _ => String::new(),
        }
    }
    pub fn bool(&self, k: &str, d: bool) -> bool {
        match self.get(k) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
            Some(Value::String(s)) => !s.is_empty(),
            _ => d,
        }
    }
    pub fn list(&self, k: &str) -> Vec<String> {
        match self.get(k) {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
                .collect(),
            Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
            _ => Vec::new(),
        }
    }
}

// ── candidates ──────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Cand {
    Official(Arc<VectorHit>),
    Personal(usize),
}

impl Cand {
    fn key(&self) -> (u8, u64) {
        match self {
            Cand::Official(h) => (0, ((h.seg as u64) << 32) | h.slot as u64),
            Cand::Personal(s) => (1, *s as u64),
        }
    }
}

/// One place a candidate's text is at: a line of a library book, or of a personal or
/// database book (`book`: its index in the personal set).
#[derive(Debug, Clone, PartialEq)]
struct Place {
    src: SourceKey,
    bid: i64,
    li: i64,
    /// The window of the line the vector was embedded from (255: a short line that shares
    /// its neighbour's vector).
    win: u8,
    book: Option<usize>,
}

fn library_book_id(name: &str) -> Option<i64> {
    name.strip_prefix("id:")?.parse().ok()
}

/// The places of a vector's records, library books first and each kind in the records'
/// order (the scan lists the books its filter admitted first). `scope`: the books the search
/// admitted, when it was filtered — a record of another book is not a place of this search.
/// `personal`: a book outside the library (`uid:…`, `db:…`) to its index in the personal
/// set; a record no book answers for is left out.
fn record_places(
    records: &[RecordRef],
    scope: Option<&BookSet>,
    library: impl Fn(i64) -> bool,
    personal: impl Fn(&str) -> Option<(SourceKey, i64, usize)>,
) -> Vec<Place> {
    let mut lib = Vec::new();
    let mut other = Vec::new();
    for r in records {
        if scope.is_some_and(|s| !s.contains(&r.book)) {
            continue;
        }
        let (li, win) = (r.hint as i64, r.window);
        if r.book.starts_with("id:") {
            if let Some(bid) = library_book_id(&r.book).filter(|b| library(*b)) {
                lib.push(Place {
                    src: SourceKey::Official,
                    bid,
                    li,
                    win,
                    book: None,
                });
            }
        } else if let Some((src, bid, bi)) = personal(&r.book) {
            other.push(Place {
                src,
                bid,
                li,
                win,
                book: Some(bi),
            });
        }
    }
    lib.extend(other);
    lib
}

/// Lines a result already stands for besides its own: the other books its text is in
/// (`also_in`). A later candidate at or beside one of them is the same passage again.
#[derive(Default)]
struct Covered(HashMap<(SourceKey, i64), Vec<(i64, usize)>>);

impl Covered {
    fn add(&mut self, src: &SourceKey, bid: i64, li: i64, hit: usize) {
        self.0
            .entry((src.clone(), bid))
            .or_default()
            .push((li, hit));
    }

    /// The result that covers a line within `merge_lines` of `li` in this book.
    fn find(&self, key: &(SourceKey, i64), li: i64, merge_lines: i64) -> Option<usize> {
        self.0
            .get(key)?
            .iter()
            .find(|(l, _)| (l - li).abs() <= merge_lines)
            .map(|(_, hit)| *hit)
    }
}

struct Run {
    per_query: Vec<Vec<(f32, Cand)>>,
    personal: Option<Arc<PersonalSet>>,
    /// The books the scan of Otzaria's set admitted, when it was filtered.
    scope: Option<Arc<BookSet>>,
    embed_ms: u128,
    scan_ms: u128,
}

// ── the service ─────────────────────────────────────────────────────────────

struct StatusState {
    value: Option<(Value, SystemTime)>,
    busy: bool,
}

/// (categories, excluded categories, personal books admitted, library only) of a search scope.
type ScopeKey = (Vec<String>, Vec<String>, bool, bool);
/// Per attached database: its mtime and the era of each book.
type EraCache = HashMap<String, (Option<SystemTime>, HashMap<i64, u8>)>;

pub struct SemanticService {
    cfg: SemanticConfig,
    pub catalog: Catalog,
    refs: Option<Arc<crate::db::Db>>,
    official: RwLock<Official>,
    personal: RwLock<Option<Arc<PersonalSet>>>,
    personal_note: Mutex<Option<String>>,
    book_sets: Mutex<HashMap<ScopeKey, Option<Arc<BookSet>>>>,
    status: Arc<Mutex<StatusState>>,
    outbox: Option<Outbox>,
    last_refresh: Mutex<Instant>,
    started: Instant,
    /// The served model's calibrated defaults, when a file for it exists.
    calibration: RwLock<Option<Arc<Calibration>>>,
}

fn official_config(cfg: &SemanticConfig) -> OfficialConfig {
    OfficialConfig {
        vectors_dir: cfg.vectors_dir.clone(),
        books_dir: cfg.books_dir.clone(),
        model_dir: cfg.model_dir.clone(),
        onnx_runtime: cfg.onnx_runtime.clone(),
        scan_threads: cfg.scan_threads,
    }
}

impl SemanticService {
    pub fn open(cfg: SemanticConfig, refs: Option<Arc<crate::db::Db>>) -> Result<SemanticService> {
        let t0 = Instant::now();
        let official = Official::open(&official_config(&cfg))?;
        let catalog = Catalog::open(&cfg.db, cfg.snapshot)?;
        let outbox = cfg.feedback_dir.as_ref().map(|d| Outbox::start(d.clone()));
        let svc = SemanticService {
            catalog,
            refs,
            official: RwLock::new(official),
            personal: RwLock::new(None),
            personal_note: Mutex::new(None),
            book_sets: Mutex::new(HashMap::new()),
            status: Arc::new(Mutex::new(StatusState {
                value: None,
                busy: false,
            })),
            outbox,
            last_refresh: Mutex::new(Instant::now()),
            started: Instant::now(),
            calibration: RwLock::new(None),
            cfg,
        };
        svc.load_personal();
        svc.load_calibration();
        {
            let o = svc.official.read().unwrap();
            eprintln!(
                "otzaria-mcp: semantic ready in {:.1}s: {} vectors, library v{} ({}), model {} ({}/{})",
                t0.elapsed().as_secs_f64(),
                o.slots(),
                o.declared.library_version,
                o.declared.release_tag,
                o.family(),
                o.model_dir.file_name().unwrap_or_default().to_string_lossy(),
                o.query_graph.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        Ok(svc)
    }

    fn load_personal(&self) {
        let ij = PersonalSet::index_json(&self.cfg.personal_dir);
        if !ij.exists() {
            *self.personal.write().unwrap() = None;
            return;
        }
        match PersonalSet::open(
            &self.cfg.personal_dir,
            Some(&self.cfg.user_books_db),
            self.cfg.scan_personal_index,
        ) {
            Ok(ps) => {
                if self.cfg.scan_personal_index {
                    eprintln!(
                        "otzaria-mcp: personal index: {} passages from {} books",
                        ps.n,
                        ps.books.len()
                    );
                } else {
                    eprintln!(
                        "otzaria-mcp: personal index: {} books (their vectors are read from \
                         Otzaria's set; --scan-personal-index scans the index's own)",
                        ps.books.len()
                    );
                }
                *self.personal.write().unwrap() = Some(Arc::new(ps));
                self.check_personal_model();
            }
            Err(e) => {
                eprintln!("otzaria-mcp: personal index not loaded: {e:#}");
                *self.personal.write().unwrap() = None;
            }
        }
        self.book_sets.lock().unwrap().clear();
    }

    /// What keeps personal and database books out of searches, if anything. Scanning the
    /// personal index: its vectors must come from the vector set's model. Reading them from
    /// Otzaria's set (the default): the set must hold vectors of those books at all.
    fn check_personal_model(&self) {
        let ps = self.personal.read().unwrap().clone();
        let note = if self.cfg.scan_personal_index {
            let want = self.official.read().unwrap().family().to_string();
            match ps.as_ref().and_then(|p| p.model().map(String::from)) {
                Some(have) if have != want => Some(format!(
                    "personal index was built with {have}; Otzaria's vectors now use {want}. \
                     Personal and database books are left out of searches until the personal \
                     index is rebuilt."
                )),
                _ => None,
            }
        } else {
            match &ps {
                Some(p) if !self.set_holds_books_of(p) => Some(
                    "Otzaria's vector set holds no vectors of the personal and database books: \
                     they are left out of searches until they are published into it \
                     (oxv_export.py, or build_personal_index.py --publish-oxv), or the server is \
                     started with --scan-personal-index."
                        .to_string(),
                ),
                _ => None,
            }
        };
        let mut cur = self.personal_note.lock().unwrap();
        if note.is_some() && *cur != note {
            eprintln!("otzaria-mcp: {}", note.as_deref().unwrap());
        }
        *cur = note;
    }

    /// Whether Otzaria's set holds a vector of any book of the personal set: one scan that
    /// admits only those books. A scan that fails says nothing, and is not held against them.
    fn set_holds_books_of(&self, ps: &PersonalSet) -> bool {
        let names: BookSet = ps.set_names(None).collect();
        if names.is_empty() {
            return false;
        }
        let o = self.official.read().unwrap();
        let mut probe = vec![0f32; (o.declared.model.embedding_dim as usize).max(1)];
        probe[0] = 1.0;
        o.scan(&probe, 1, Some(&names))
            .map(|hits| !hits.is_empty())
            .unwrap_or(true)
    }

    /// The personal set a search reads personal and database books through: its vectors when
    /// the personal index is scanned, else its books (the vectors are in Otzaria's set).
    fn personal_searchable(&self) -> Option<Arc<PersonalSet>> {
        if self.personal_note.lock().unwrap().is_some() {
            return None;
        }
        let ps = self.personal.read().unwrap().clone();
        if self.cfg.scan_personal_index {
            ps.filter(|p| p.n > 0)
        } else {
            ps
        }
    }

    fn personal_any(&self) -> Option<Arc<PersonalSet>> {
        self.personal.read().unwrap().clone()
    }

    /// Pick up a new vector generation, a rebuilt personal index or a replaced seforim.db.
    /// Called before each request and every 30 s, so a stale mapping is released soon and
    /// Windows lets Otzaria delete the files of a generation it replaced.
    pub fn refresh(&self) {
        *self.last_refresh.lock().unwrap() = Instant::now();
        // personal index
        let ij = PersonalSet::index_json(&self.cfg.personal_dir);
        let m = std::fs::metadata(&ij).and_then(|m| m.modified()).ok();
        let changed = match &*self.personal.read().unwrap() {
            Some(p) => p.mtime != m,
            None => m.is_some(),
        };
        if changed {
            self.load_personal();
        }
        // official vectors
        let before = self.official.read().unwrap().declared.generation;
        let now = super::official::read_set(&self.cfg.vectors_dir)
            .map(|d| d.generation)
            .ok();
        if now.is_some() && now != Some(before) {
            let mut o = self.official.write().unwrap();
            match o.reload() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("otzaria-mcp: reload of the vector set failed ({e}); reopening it");
                    match Official::open(&official_config(&self.cfg)) {
                        Ok(fresh) => *o = fresh,
                        Err(e2) => {
                            eprintln!("otzaria-mcp: reopening the vector set failed: {e2:#}")
                        }
                    }
                }
            }
            eprintln!(
                "otzaria-mcp: vector set v{} (generation {}), model {}",
                o.declared.library_version,
                o.declared.generation,
                o.family()
            );
            drop(o);
            self.book_sets.lock().unwrap().clear();
            self.check_personal_model();
        }
        if self.catalog.maybe_reopen() {
            self.book_sets.lock().unwrap().clear();
        }
        self.load_calibration();
    }

    /// The files `defaults-<model>.json` is looked for in, for the served model.
    fn calibration_candidates(&self) -> Vec<PathBuf> {
        let family = self.official.read().unwrap().family().to_string();
        super::calibration::candidates(&self.cfg.calibration_dirs, &family)
    }

    /// (Re)load the served model's calibration when its file appeared, changed or went away.
    fn load_calibration(&self) {
        let found = self
            .calibration_candidates()
            .into_iter()
            .find(|p| p.is_file());
        let mtime = found
            .as_ref()
            .and_then(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
        {
            let cur = self.calibration.read().unwrap();
            let same = match (&*cur, &found) {
                (None, None) => true,
                (Some(c), Some(f)) => c.file == *f && c.mtime == mtime,
                _ => false,
            };
            if same {
                return;
            }
        }
        let next = match found {
            None => {
                eprintln!(
                    "otzaria-mcp: no calibration file for the served model; built-in defaults"
                );
                None
            }
            Some(f) => match Calibration::load(&f) {
                Ok(c) => {
                    eprintln!(
                        "otzaria-mcp: calibration {}: min_score {:?}/{:?}/{:?} (short/medium/long), \
                         window penalty {:?}",
                        f.display(),
                        c.short.min_score,
                        c.medium.min_score,
                        c.long.min_score,
                        c.window_penalty
                    );
                    Some(Arc::new(c))
                }
                Err(e) => {
                    // keep what was loaded: a file being rewritten is read again next time
                    eprintln!("otzaria-mcp: calibration not loaded: {e:#}");
                    return;
                }
            },
        };
        Calibration::apply(next.as_deref());
        *self.calibration.write().unwrap() = next;
    }

    fn calibration(&self) -> Option<Arc<Calibration>> {
        self.calibration.read().unwrap().clone()
    }

    /// Run `refresh` every 30 s in the background.
    pub fn watch(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        std::thread::Builder::new()
            .name("semantic-refresh".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_secs(30));
                match weak.upgrade() {
                    Some(s) => {
                        if s.last_refresh.lock().unwrap().elapsed() >= Duration::from_secs(25) {
                            s.refresh();
                        }
                    }
                    None => break,
                }
            })
            .ok();
    }

    // ── scopes ──────────────────────────────────────────────────────────────

    /// The books a scan of Otzaria's set admits (`None`: every book). `personal`: the personal
    /// set whose books the scan may return (their vectors are read from Otzaria's set), by
    /// the same category scope. `library_only`: a search without personal books of a set that
    /// holds them.
    fn official_scope(
        &self,
        include: &[String],
        exclude: &[String],
        personal: Option<&PersonalSet>,
        library_only: bool,
    ) -> Option<Arc<BookSet>> {
        if include.is_empty() && exclude.is_empty() && !library_only {
            return None;
        }
        let key = (
            include.to_vec(),
            exclude.to_vec(),
            personal.is_some(),
            library_only,
        );
        if let Some(v) = self.book_sets.lock().unwrap().get(&key) {
            return v.clone();
        }
        let mut admitted: HashSet<i64> = if include.is_empty() {
            self.catalog.all_book_ids()
        } else {
            self.catalog.books_in_categories(include)
        };
        if !exclude.is_empty() {
            for b in self.catalog.books_in_categories(exclude) {
                admitted.remove(&b);
            }
        }
        let mut names: HashSet<Box<str>> = admitted
            .into_iter()
            .map(|b| format!("id:{b}").into_boxed_str())
            .collect();
        if let Some(ps) = personal {
            let mask = ps.book_mask(include, exclude);
            names.extend(ps.set_names(mask.as_deref()).map(Box::from));
        }
        let set = Arc::new(BookSet(names));
        self.book_sets
            .lock()
            .unwrap()
            .insert(key, Some(set.clone()));
        Some(set)
    }

    /// The personal set of a request (`include_personal`), and the scope of its scan of
    /// Otzaria's set.
    fn request_scope(
        &self,
        opts: &Opts,
        include: &[String],
        exclude: &[String],
    ) -> (Option<Arc<PersonalSet>>, Option<Arc<BookSet>>) {
        let searchable = self.personal_searchable();
        let personal = if opts.bool("include_personal", true) {
            searchable.clone()
        } else {
            None
        };
        let scope = if self.cfg.scan_personal_index {
            self.official_scope(include, exclude, None, false)
        } else {
            // the set holds the personal books' vectors: a search without them names the library
            self.official_scope(
                include,
                exclude,
                personal.as_deref(),
                personal.is_none() && searchable.is_some(),
            )
        };
        (personal, scope)
    }

    // ── scanning ────────────────────────────────────────────────────────────

    fn run_queries(&self, texts: &[String], extra: Vec<Vec<f32>>, opts: &Opts) -> Result<Run> {
        self.refresh();
        let k = opts.int("k", 20).max(1) as usize;
        let kc = 64.max(k * opts.int("oversample", 8).max(1) as usize);
        let include = opts.list("categories");
        let exclude = opts.list("exclude_categories");
        let official = self.official.read().unwrap();
        let t0 = Instant::now();
        let mut vectors = Vec::with_capacity(texts.len() + extra.len());
        for t in texts {
            vectors.push(official.embed_query(t)?);
        }
        vectors.extend(extra);
        let embed_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let (personal, scope) = self.request_scope(opts, &include, &exclude);
        let mut per_query: Vec<Vec<(f32, Cand)>> = Vec::with_capacity(vectors.len());
        for v in &vectors {
            let hits = official.scan(v, kc, scope.as_deref())?;
            per_query.push(
                hits.into_iter()
                    .map(|h| (h.score, Cand::Official(Arc::new(h))))
                    .collect(),
            );
        }
        drop(official);
        if let Some(ps) = personal.as_ref().filter(|_| self.cfg.scan_personal_index) {
            let mask = ps.book_mask(&include, &exclude);
            let res = ps.scan(&vectors, kc, mask.as_deref());
            for (qi, hits) in res.into_iter().enumerate() {
                per_query[qi].extend(hits.into_iter().map(|(s, slot)| (s, Cand::Personal(slot))));
            }
        }
        for q in per_query.iter_mut() {
            q.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        Ok(Run {
            per_query,
            personal,
            scope,
            embed_ms,
            scan_ms: t1.elapsed().as_millis(),
        })
    }

    pub fn search(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        let q = opts.str("query").trim().to_string();
        if q.is_empty() {
            return Err(bad("empty query"));
        }
        let run = self.run_queries(std::slice::from_ref(&q), Vec::new(), &opts)?;
        let t = Instant::now();
        let cands: Vec<(f32, Cand, Vec<usize>)> = run.per_query[0]
            .iter()
            .map(|(s, c)| (*s, c.clone(), vec![0]))
            .collect();
        let hits = self.resolve(
            &cands,
            &opts,
            None,
            run.personal.as_deref(),
            run.scope.as_deref(),
        )?;
        Ok(json!({"query": q, "hits": hits,
                  "timing_ms": {"embed": run.embed_ms, "scan": run.scan_ms, "resolve": t.elapsed().as_millis()}}))
    }

    fn count_tokens(&self) -> Result<impl Fn(&str) -> usize> {
        type Cached = Option<(PathBuf, Arc<tokenizers::Tokenizer>)>;
        static CACHE: std::sync::LazyLock<Mutex<Cached>> =
            std::sync::LazyLock::new(Default::default);
        let dir = self.official.read().unwrap().model_dir.clone();
        let mut cache = CACHE.lock().unwrap();
        let tok = match &*cache {
            Some((d, t)) if *d == dir => t.clone(),
            _ => {
                let t = Arc::new(
                    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                        .map_err(|e| anyhow!("cannot read the model's tokenizer: {e}"))?,
                );
                *cache = Some((dir, t.clone()));
                t
            }
        };
        Ok(move |s: &str| tok.encode(s, false).map(|e| e.get_ids().len()).unwrap_or(0))
    }

    /// Token counts by the query model's tokenizer, and the query cap, for a client that
    /// cuts its own query windows (the Otzaria plugin).
    pub fn tokens(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        let texts = opts.list("texts");
        if texts.len() > 5000 {
            return Err(bad("at most 5000 texts"));
        }
        let count = self.count_tokens()?;
        let counts: Vec<usize> = texts.iter().map(|t| count(t)).collect();
        Ok(json!({"counts": counts, "query_max_tokens": self.query_cap()}))
    }

    /// The Otzaria plugin "מקבילות": its query windows embedded and scanned in one request
    /// (the library, and the personal index when it matches the library's model), the hits
    /// pooled with each vector's best window, and — `rescore`, for several sources — each
    /// pooled vector scored exactly against every window. The plugin filters, looks the books
    /// up and reads the text through Otzaria's own API, so this returns only where each
    /// vector is: a library book and line, or a personal book's Otzaria uid.
    pub fn parallels(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        let windows: Vec<String> = opts
            .list("windows")
            .into_iter()
            .map(|w| w.trim().to_string())
            .filter(|w| !w.is_empty())
            .collect();
        if windows.is_empty() {
            return Err(bad("no query windows"));
        }
        if windows.len() > 32 {
            return Err(bad("at most 32 query windows"));
        }
        self.refresh();
        let k = opts.int("k", 120).clamp(1, 2000) as usize;
        let calibration = self.calibration();
        let min_score = match opts.get("min_score") {
            Some(_) => opts.float("min_score", 0.0),
            None => calibration
                .clone()
                .and_then(|c| c.parallels_min_score.or_else(|| c.min_score("long")))
                .map_or_else(|| opts.float("min_score_fallback", 0.0), f64::from),
        } as f32;
        let max_cands = opts.int("max_candidates", 2 * k as i64).clamp(1, 5000) as usize;
        let rescore = opts.bool("rescore", false) && windows.len() > 1;
        let (personal, scope) = self.request_scope(&opts, &[], &[]);
        let official = self.official.read().unwrap();
        let t0 = Instant::now();
        let mut vectors = Vec::with_capacity(windows.len());
        for w in &windows {
            vectors.push(official.embed_query(w)?);
        }
        let embed_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        // the best window of each vector
        let mut best: HashMap<(u8, u64), (f32, usize, Cand)> = HashMap::new();
        let mut keep = |score: f32, wi: usize, c: Cand| {
            if score < min_score {
                return;
            }
            let e = best.entry(c.key()).or_insert((score, wi, c.clone()));
            if score > e.0 {
                *e = (score, wi, c);
            }
        };
        for (wi, v) in vectors.iter().enumerate() {
            for h in official.scan(v, k, scope.as_deref())? {
                keep(h.score, wi, Cand::Official(Arc::new(h)));
            }
        }
        if let Some(ps) = personal.as_ref().filter(|_| self.cfg.scan_personal_index) {
            for (wi, hits) in ps.scan(&vectors, k, None).into_iter().enumerate() {
                for (score, slot) in hits {
                    keep(score, wi, Cand::Personal(slot));
                }
            }
        }
        let mut pool: Vec<(f32, usize, Cand)> = best.into_values().collect();
        pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.2.key().cmp(&b.2.key())));
        pool.truncate(max_cands);
        let scan_ms = t1.elapsed().as_millis();
        let t2 = Instant::now();
        let mut matrix: HashMap<(u8, u64), Vec<f32>> = HashMap::new();
        if rescore {
            matrix = pool
                .iter()
                .map(|(_, _, c)| (c.key(), vec![-1.0; vectors.len()]))
                .collect();
            let refs: Vec<SlotRef> = pool
                .iter()
                .filter_map(|(_, _, c)| match c {
                    Cand::Official(h) => Some(SlotRef {
                        seg: h.seg,
                        slot: h.slot,
                        key: h.key,
                    }),
                    Cand::Personal(_) => None,
                })
                .collect();
            for (wi, v) in vectors.iter().enumerate() {
                for h in official.score_slots(v, &refs)? {
                    if let Some(row) = matrix.get_mut(&(0, ((h.seg as u64) << 32) | h.slot as u64))
                    {
                        row[wi] = h.score;
                    }
                }
                if let Some(ps) = &personal {
                    for (_, _, c) in &pool {
                        if let Cand::Personal(slot) = c {
                            if let Some(row) = matrix.get_mut(&c.key()) {
                                row[wi] = ps.score(v, *slot);
                            }
                        }
                    }
                }
            }
        }
        let rescore_ms = t2.elapsed().as_millis();
        let r5 = |x: f32| round_to(x as f64, 5);
        let mut hits = Vec::with_capacity(pool.len());
        for (score, wi, c) in &pool {
            let mut o = match c {
                Cand::Official(h) => {
                    let places = self.hit_places(personal.as_deref(), scope.as_deref(), h);
                    let Some(at) = places.first() else {
                        continue;
                    };
                    // 255: a short line sharing its neighbour's vector, at its own line
                    let win = if at.win == 255 { 0 } else { at.win };
                    let key = format!("o:{}:{}", h.seg, h.slot);
                    match at.book.and_then(|bi| personal.as_ref()?.books.get(bi)) {
                        // a personal or database book whose vector is in Otzaria's set
                        Some(b) => personal_parallel(b, key, at.li, win),
                        None if at.book.is_some() => continue,
                        None => {
                            json!({"store": "lib", "key": key, "source": "library",
                                   "id": at.bid, "line": at.li, "window": win})
                        }
                    }
                }
                Cand::Personal(slot) => {
                    let Some(ps) = &personal else { continue };
                    let (bi, li) = ps.record(*slot);
                    let Some(b) = ps.books.get(bi) else { continue };
                    personal_parallel(b, format!("p:{slot}"), li, ps.slot_window(*slot))
                }
            };
            let m = o.as_object_mut().unwrap();
            m.insert("score".into(), json!(r5(*score)));
            if let Some(p) = calibration
                .as_ref()
                .and_then(|c| c.confidence(bucket_of(words(&windows[*wi])), *score))
            {
                m.insert("confidence".into(), json!(round_to(p as f64, 3)));
            }
            m.insert("win".into(), json!(wi));
            if let Some(row) = matrix.get(&c.key()) {
                m.insert(
                    "scores".into(),
                    json!(row.iter().map(|x| r5(*x)).collect::<Vec<_>>()),
                );
            }
            hits.push(o);
        }
        let any = self.personal_any();
        Ok(json!({
            "hits": hits, "windows": windows.len(),
            "library": {"release_tag": official.declared.release_tag, "library_version": official.declared.library_version,
                        "generation": official.declared.generation, "segments": official.segments(),
                        "model_family": official.family(),
                        "query_graph": official.query_graph.file_name().unwrap_or_default().to_string_lossy()},
            "personal": {"searched": personal.is_some(), "books": any.as_ref().map_or(0, |p| p.books.len()),
                         "slots": any.as_ref().map_or(0, |p| p.n), "vectors": self.personal_vectors(),
                         "note": self.personal_note.lock().unwrap().clone()},
            "timing_ms": {"embed": embed_ms, "scan": scan_ms, "rescore": rescore_ms}
        }))
    }

    fn query_cap(&self) -> usize {
        let o = self.official.read().unwrap();
        o.declared
            .model
            .query_max_tokens
            .unwrap_or(o.declared.model.max_tokens)
    }

    pub fn related(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        let count = self.count_tokens()?;
        let wt = (opts.int("window_tokens", 160).max(8) as usize)
            .min(self.query_cap().saturating_sub(8).max(8));
        let windows = note_windows(
            &count,
            &opts.str("text"),
            &opts.str("title"),
            wt,
            opts.int("max_windows", 12).max(1) as usize,
            opts.int("min_hebrew", 12).max(0) as usize,
        );
        if windows.is_empty() {
            return Ok(
                json!({"windows": [], "hits": [], "note": "no Hebrew/Aramaic text long enough to search"}),
            );
        }
        let run = self.run_queries(&windows, Vec::new(), &opts)?;
        // merge: a passage keeps its best score and every window that found it, strongest first
        let mut best: HashMap<(u8, u64), (f32, Cand, Vec<usize>)> = HashMap::new();
        let mut order: Vec<(u8, u64)> = Vec::new();
        for (c, hits) in run.per_query.iter().enumerate() {
            for (s, cand) in hits {
                let key = cand.key();
                match best.get_mut(&key) {
                    Some(cur) => {
                        cur.2.push(c);
                        if *s > cur.0 {
                            cur.0 = *s;
                            let last = cur.2.pop().unwrap();
                            cur.2.insert(0, last);
                        }
                    }
                    None => {
                        best.insert(key, (*s, cand.clone(), vec![c]));
                        order.push(key);
                    }
                }
            }
        }
        let bonus = opts.float("multi_window_bonus", 0.01) as f32;
        let mut cands: Vec<(f32, Cand, Vec<usize>)> = order
            .into_iter()
            .map(|k| {
                let (s, c, w) = best.remove(&k).unwrap();
                (s + bonus * (w.len() as f32 - 1.0), c, w)
            })
            .collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        let t = Instant::now();
        let mut o = req.clone();
        if !o.contains_key("echo_text") {
            o.insert(
                "echo_text".into(),
                Value::String(format!(
                    "{}\n{}",
                    clean_markdown(&opts.str("title")),
                    clean_markdown(&opts.str("text"))
                )),
            );
        }
        let hits = self.resolve(
            &cands,
            &Opts(&o),
            Some(&windows),
            run.personal.as_deref(),
            run.scope.as_deref(),
        )?;
        Ok(json!({"windows": windows, "hits": hits,
                  "timing_ms": {"embed": run.embed_ms, "scan": run.scan_ms, "resolve": t.elapsed().as_millis()}}))
    }

    // ── resolution ──────────────────────────────────────────────────────────

    /// Where a vector of Otzaria's set is: see [`record_places`]. Books outside the library
    /// are places only when their vectors are read from the set (not `--scan-personal-index`)
    /// and the personal set names them.
    fn hit_places(
        &self,
        ps: Option<&PersonalSet>,
        scope: Option<&BookSet>,
        hit: &VectorHit,
    ) -> Vec<Place> {
        let ps = ps.filter(|_| !self.cfg.scan_personal_index);
        record_places(
            &hit.records,
            scope,
            |bid| self.catalog.has_book(bid),
            |name| {
                let ps = ps?;
                let bi = *ps.by_set_name.get(name)?;
                let b = &ps.books[bi];
                Some((b.source(), b.book_id, bi))
            },
        )
    }

    /// Every place of a candidate, the one it is shown at first.
    fn cand_places(
        &self,
        ps: Option<&PersonalSet>,
        scope: Option<&BookSet>,
        c: &Cand,
    ) -> Vec<Place> {
        match c {
            Cand::Official(h) => self.hit_places(ps, scope, h),
            Cand::Personal(slot) => {
                let Some(ps) = ps else { return Vec::new() };
                let (bi, li) = ps.record(*slot);
                ps.books
                    .get(bi)
                    .map(|b| Place {
                        src: b.source(),
                        bid: b.book_id,
                        li,
                        win: ps.slot_window(*slot),
                        book: Some(bi),
                    })
                    .into_iter()
                    .collect()
            }
        }
    }

    /// Where the personal and database books' vectors are read from (`/health`).
    fn personal_vectors(&self) -> &'static str {
        if self.cfg.scan_personal_index {
            "personal-index"
        } else {
            "otzaria-set"
        }
    }

    /// A candidate as a result, at the first of its places, with the others as `also_in`.
    fn base_at(
        &self,
        conn: &rusqlite::Connection,
        ps: Option<&PersonalSet>,
        places: &[Place],
    ) -> Result<Option<Map<String, Value>>> {
        let Some(at) = places.first() else {
            return Ok(None);
        };
        let base = match at.book {
            None => self.base_official(conn, at.bid, at.li, at.win)?,
            Some(bi) => ps.and_then(|p| self.base_personal(p, bi, at.li, at.win)),
        };
        let Some(mut m) = base else { return Ok(None) };
        // the same text elsewhere: not an alias (a short neighbouring line that shares this
        // line's vector, window 255) nor the line's own neighbours
        let mut also = Vec::new();
        let others = places[1..].iter().filter(|p| {
            p.win != 255 && !(p.src == at.src && p.bid == at.bid && (p.li - at.li).abs() <= 2)
        });
        for p in others.take(4) {
            let title = match p.book {
                None => self.catalog.book(p.bid).map(|b| b.title),
                Some(bi) => ps.and_then(|s| s.books.get(bi)).map(|b| b.title.clone()),
            };
            if let Some(title) = title {
                also.push(
                    json!({"book_id": p.bid, "book_title": title, "line_index": p.li,
                                 "url": p.src.url(p.bid, p.li)}),
                );
            }
        }
        m.insert("also_in".into(), Value::Array(also));
        Ok(Some(m))
    }

    fn base_official(
        &self,
        conn: &rusqlite::Connection,
        bid: i64,
        li: i64,
        win: u8,
    ) -> Result<Option<Map<String, Value>>> {
        let Some(book) = self.catalog.book(bid) else {
            return Ok(None);
        };
        let rows = self.catalog.lines(conn, bid, li, li + 3)?;
        if rows.first().map(|r| r.index) != Some(li) {
            return Ok(None);
        }
        let text = strip_html(&rows[0].content);
        let mut snippet = text.clone();
        if win != 0 && win != 255 {
            // the vector is a later window of a long line: show that part of the line, cut from
            // the line as the index stores it (no nikud), as the window was (a line whose
            // footnotes the indexer put back can still be shifted a little)
            let t = line_text(&rows[0].content);
            let (a, b) = window_span(&t, win as usize);
            snippet = format!(
                "… {}{}",
                char_slice(&t, a, b),
                if b < char_len(&t) { " …" } else { "" }
            );
        }
        for r in &rows[1..] {
            if char_len(&snippet) >= 40 {
                break;
            }
            snippet = format!("{} {}", snippet, strip_html(&r.content))
                .trim()
                .to_string();
        }
        let mut toc = match rows[0].toc {
            Some(id) => self.catalog.toc_path(conn, id, 6)?,
            None => Vec::new(),
        };
        let nq = |s: &str| s.replace('"', "");
        if let Some(first) = toc.first() {
            if nq(first) == nq(&book.title) || nq(first) == nq(book.he_ref.as_deref().unwrap_or(""))
            {
                toc.remove(0);
            }
        }
        // a heRef that is only "<title> <line number>" says nothing; build one from the TOC
        let heref = rows[0].he_ref.clone().unwrap_or_default();
        let exact = !heref.is_empty() && !ends_with_space_number(&heref);
        let he_ref = if exact {
            heref
        } else if !toc.is_empty() {
            std::iter::once(book.title.clone())
                .chain(toc.iter().cloned())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            book.title.clone()
        };
        let mut m = Map::new();
        m.insert("key".into(), json!(["o", bid]));
        m.insert("book_id".into(), json!(bid));
        m.insert("book_title".into(), json!(book.title));
        m.insert(
            "category".into(),
            json!(self.catalog.cat_path(book.category)),
        );
        m.insert("line_index".into(), json!(li));
        m.insert("he_ref".into(), json!(he_ref));
        m.insert("has_exact_ref".into(), json!(exact));
        m.insert("toc".into(), json!(toc));
        m.insert("text".into(), json!(text));
        m.insert("snippet".into(), json!(snippet));
        m.insert("url".into(), json!(SourceKey::Official.url(bid, li)));
        m.insert("also_in".into(), json!([]));
        m.insert("window".into(), json!(win));
        m.insert("book_source".into(), json!("official"));
        m.insert("personal".into(), json!(false));
        m.insert("stale".into(), json!(false));
        Ok(Some(m))
    }

    fn base_personal(
        &self,
        ps: &PersonalSet,
        bi: usize,
        li: i64,
        win: u8,
    ) -> Option<Map<String, Value>> {
        let b = ps.books.get(bi)?;
        let (lines, stale) = ps.lines_of(bi);
        let lines = lines?;
        let raw = lines.get(li as usize)?;
        let text = strip_html(raw);
        if text.is_empty() {
            return None;
        }
        let mut snippet = text.clone();
        if win > 0 && win != 255 {
            // a later window of a long line: the windows were cut from the line as Otzaria's
            // index stores it
            let t = line_text(&strip_data_uris(raw));
            let (a, e) = window_span(&t, win as usize);
            snippet = format!(
                "… {}{}",
                char_slice(&t, a, e),
                if e < char_len(&t) { " …" } else { "" }
            );
        }
        let mut j = li as usize + 1;
        while char_len(&snippet) < 40 && j < lines.len() && j <= li as usize + 3 {
            snippet = format!("{} {}", snippet, strip_html(&lines[j]))
                .trim()
                .to_string();
            j += 1;
        }
        let mut hp = heading_path(&lines, li as usize);
        if hp
            .first()
            .is_some_and(|h| h.replace('"', "") == b.title.replace('"', ""))
        {
            hp.remove(0);
        }
        let he_ref = std::iter::once(b.title.clone())
            .chain(hp.iter().cloned())
            .collect::<Vec<_>>()
            .join(", ");
        let attached = b.attached();
        let mut m = Map::new();
        m.insert("key".into(), json!([b.source().code(), b.book_id]));
        m.insert("book_id".into(), json!(b.book_id));
        m.insert("book_title".into(), json!(b.title));
        m.insert("category".into(), json!(b.category_path()));
        m.insert("line_index".into(), json!(li));
        m.insert("he_ref".into(), json!(he_ref));
        m.insert("has_exact_ref".into(), json!(false));
        m.insert("toc".into(), json!(hp));
        m.insert("text".into(), json!(text));
        m.insert("snippet".into(), json!(snippet));
        m.insert("url".into(), json!(b.url(li)));
        m.insert("also_in".into(), json!([]));
        m.insert("window".into(), json!(win));
        m.insert(
            "book_source".into(),
            json!(if attached { "attached" } else { "user" }),
        );
        m.insert(
            "library".into(),
            if attached { json!(b.lib) } else { Value::Null },
        );
        m.insert(
            "library_id".into(),
            if attached { json!(b.slug) } else { Value::Null },
        );
        m.insert("personal".into(), json!(true));
        m.insert("stale".into(), json!(stale));
        Some(m)
    }

    fn resolve(
        &self,
        cands: &[(f32, Cand, Vec<usize>)],
        opts: &Opts,
        windows: Option<&[String]>,
        ps: Option<&PersonalSet>,
        scope: Option<&BookSet>,
    ) -> Result<Vec<Value>> {
        let k = opts.int("k", 20).max(1) as usize;
        // the query's length bucket: the query, or a note's windows on average
        let n_words = match windows {
            Some(w) if !w.is_empty() => w.iter().map(|t| words(t)).sum::<usize>() / w.len(),
            _ => words(&opts.str("query")),
        };
        let bucket = bucket_of(n_words);
        let calibration = self.calibration();
        let min_score = match opts.get("min_score") {
            Some(_) => opts.float("min_score", 0.0),
            None => calibration
                .as_ref()
                .and_then(|c| c.min_score(bucket))
                .map_or_else(|| opts.float("min_score_fallback", 0.0), f64::from),
        } as f32;
        let max_per_book = opts.int("max_per_book", 3).max(1) as usize;
        let merge_lines = opts.int("merge_lines", 2).max(0);
        let mut cited_urls: HashSet<(SourceKey, i64, i64)> = HashSet::new();
        let mut cited_refs: HashSet<String> = HashSet::new();
        for c in opts.list("cited") {
            match parse_book_url(&c) {
                Some(u) => {
                    cited_urls.insert(u);
                }
                None => {
                    cited_refs.insert(c.trim().to_string());
                }
            }
        }
        let n_win = windows.map_or(1, |w| w.len().max(1));
        let max_per_window = match opts.int("max_per_window", 0) {
            0 => 2.max((2 * k).div_ceil(n_win)),
            v => v.max(1) as usize,
        };
        let echo_ref = EchoReference::new(
            &opts.str("echo_text"),
            opts.int("echo_n", 3).max(0) as usize,
        );
        let echo_containment = opts.float("echo_containment", 0.5);
        let echo_run = opts.int("echo_run", 8).max(0) as usize;
        let max_chars = opts.int("max_chars", 700).max(1) as usize;
        let conn = self.catalog.connect()?;
        let mut kept: HashMap<(SourceKey, i64), Vec<usize>> = HashMap::new();
        let mut per_book: HashMap<(SourceKey, i64), usize> = HashMap::new();
        let mut per_window: HashMap<usize, usize> = HashMap::new();
        let mut seen_text: HashMap<String, usize> = HashMap::new();
        let mut covered = Covered::default();
        let mut out: Vec<Map<String, Value>> = Vec::new();
        let mut counted = 0; // flagged hits (cited / echoing the note) are shown but not counted
        for (score, cand, wins) in cands {
            if *score < min_score || counted >= k || out.len() >= 3 * k {
                break;
            }
            let places = self.cand_places(ps, scope, cand);
            let Some((src, bid, li)) = places.first().map(|p| (p.src.clone(), p.bid, p.li)) else {
                continue;
            };
            let key = (src.clone(), bid);
            // any of its places beside a line already shown — a result's own, or one of its
            // `also_in` places: the same passage again, from either side
            let near = places.iter().find_map(|p| {
                let pk = (p.src.clone(), p.bid);
                kept.get(&pk)
                    .and_then(|v| {
                        v.iter().copied().find(|&i| {
                            (out[i]["line_index"].as_i64().unwrap_or(-99) - p.li).abs()
                                <= merge_lines
                        })
                    })
                    .or_else(|| covered.find(&pk, p.li, merge_lines))
            });
            if let Some(near) = near {
                let h = &mut out[near];
                let n = h
                    .get("also_matched_lines")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                h.insert("also_matched_lines".into(), json!(n + 1));
                continue;
            }
            let mut cited = cited_urls.contains(&(src.clone(), bid, li));
            if !cited && per_book.get(&key).copied().unwrap_or(0) >= max_per_book {
                continue;
            }
            let Some(base) = self.base_at(&conn, ps, &places)? else {
                continue;
            };
            let text = base["text"].as_str().unwrap_or("").to_string();
            let norm: String = strip_nikud(&text)
                .chars()
                .filter(|c| ('\u{05D0}'..='\u{05EA}').contains(c))
                .take(80)
                .collect();
            if char_len(&norm) >= 30 {
                if let Some(&twin) = seen_text.get(&norm) {
                    let also = out[twin]
                        .get_mut("also_in")
                        .and_then(Value::as_array_mut)
                        .unwrap();
                    if also.len() < 6 {
                        also.push(
                            json!({"book_id": base["book_id"], "book_title": base["book_title"],
                                         "line_index": li, "url": base["url"]}),
                        );
                    }
                    for p in &places {
                        covered.add(&p.src, p.bid, p.li, twin);
                    }
                    continue;
                }
            }
            cited = cited || cited_refs.contains(base["he_ref"].as_str().unwrap_or(""));
            let echo = echo_ref.measure(&text);
            // short lines (a heading, a rule's name) only count via the run test, never by share alone
            let is_echo = echo.as_ref().is_some_and(|e| {
                e.run >= echo_run || (e.words >= echo_run && e.containment >= echo_containment)
            });
            let flagged = cited || is_echo;
            if !flagged
                && windows.is_some()
                && !wins.is_empty()
                && per_window.get(&wins[0]).copied().unwrap_or(0) >= max_per_window
            {
                continue;
            }
            let mut hit = base.clone();
            hit.remove("key");
            hit.insert("score".into(), json!(round_to(*score as f64, 4)));
            if let Some(p) = calibration
                .as_ref()
                .and_then(|c| c.confidence(bucket, *score))
            {
                hit.insert("confidence".into(), json!(round_to(p as f64, 3)));
            }
            let snip = base["snippet"].as_str().unwrap_or("");
            hit.insert("snippet".into(), json!(char_prefix(snip, max_chars)));
            hit.insert("cited".into(), json!(cited));
            hit.insert("echo".into(), json!(is_echo));
            hit.insert("echo_stats".into(), serde_json::to_value(&echo)?);
            if windows.is_some() {
                hit.insert("windows".into(), json!(wins));
            }
            self.add_context(&conn, &mut hit, &src, bid, opts, ps);
            let idx = out.len();
            out.push(hit);
            kept.entry(key.clone()).or_default().push(idx);
            for p in &places[1..] {
                covered.add(&p.src, p.bid, p.li, idx);
            }
            if char_len(&norm) >= 30 {
                seen_text.insert(norm, idx);
            }
            if flagged {
                continue;
            }
            counted += 1;
            *per_book.entry(key).or_default() += 1;
            if windows.is_some() && !wins.is_empty() {
                *per_window.entry(wins[0]).or_default() += 1;
            }
        }
        Ok(out.into_iter().map(Value::Object).collect())
    }

    /// `context_before` / `context_after`: up to `context` non-empty lines on each side, at
    /// most `context_chars` characters a side (nearest lines kept), for hits shorter than
    /// `context_if_shorter` characters (0 = every hit).
    fn add_context(
        &self,
        conn: &rusqlite::Connection,
        hit: &mut Map<String, Value>,
        src: &SourceKey,
        bid: i64,
        opts: &Opts,
        ps: Option<&PersonalSet>,
    ) {
        let n = opts.int("context", 0).clamp(0, 5);
        let thr = opts.int("context_if_shorter", 0).max(0) as usize;
        if n == 0 || (thr > 0 && char_len(hit["text"].as_str().unwrap_or("")) >= thr) {
            return;
        }
        let cap = 40.max(opts.int("context_chars", 400)) as usize;
        let li = hit["line_index"].as_i64().unwrap_or(0);
        let Ok(rows) = self.book_rows(conn, src, bid, (li - 3 * n).max(0), li + 3 * n, false, ps)
        else {
            return;
        };
        let texts: HashMap<i64, String> = rows
            .rows
            .iter()
            .map(|r| (r.0, strip_html(&r.3).trim().to_string()))
            .collect();
        let url = |l: i64| src.url(bid, l);
        let side = |order: Vec<i64>| -> Vec<Value> {
            let (mut out, mut used) = (Vec::new(), 0usize);
            for lix in order {
                let t = texts.get(&lix).cloned().unwrap_or_default();
                if t.is_empty() {
                    continue;
                }
                let mut t = t;
                if used + char_len(&t) > cap {
                    t = char_prefix(&t, cap.saturating_sub(used))
                        .trim_end()
                        .to_string();
                    if t.is_empty() {
                        break;
                    }
                    t.push('…');
                }
                used += char_len(&t);
                out.push(json!({"line_index": lix, "text": t, "url": url(lix)}));
                if out.len() >= n as usize || used >= cap {
                    break;
                }
            }
            out
        };
        let mut before = side(((li - 3 * n).max(0)..li).rev().collect());
        before.reverse();
        if let Some(first) = before.first_mut() {
            // cut at the far end, not the near one
            let ft = first["text"].as_str().unwrap_or("").to_string();
            if ft.ends_with('…') {
                let lix = first["line_index"].as_i64().unwrap_or(0);
                let full = &texts[&lix];
                let keep = char_len(&ft) - 1;
                let tail: String = full
                    .chars()
                    .skip(char_len(full) - keep.min(char_len(full)))
                    .collect();
                first["text"] = json!(format!("…{tail}"));
            }
        }
        hit.insert("context_before".into(), Value::Array(before));
        hit.insert(
            "context_after".into(),
            Value::Array(side((li + 1..=li + 3 * n).collect())),
        );
    }

    // ── text by location ────────────────────────────────────────────────────

    /// (source, book id, line index, note) from a url, ids, or a Hebrew reference.
    fn locate(&self, opts: &Opts) -> Result<(SourceKey, i64, i64, Option<String>)> {
        if let Some(u) = parse_book_url(&opts.str("url")) {
            return Ok((u.0, u.1, u.2, None));
        }
        if opts.get("book_id").is_some() {
            let src = SourceKey::parse(&opts.str("source")).ok_or_else(|| bad("unknown source"))?;
            return Ok((
                src,
                opts.int("book_id", 0),
                opts.int("line_index", 0).max(0),
                None,
            ));
        }
        let r = opts.str("ref").trim().to_string();
        if r.is_empty() {
            return Err(bad("give url, book_id + line_index, or ref"));
        }
        let db = self
            .refs
            .as_ref()
            .ok_or_else(|| bad("references cannot be resolved here; pass url or book_id"))?;
        let cands = db.resolve_ref(&r, 5)?;
        let first = cands
            .first()
            .ok_or_else(|| bad(format!("no Otzaria book or location matches {r:?}")))?;
        let note = (cands.len() > 1).then(|| {
            format!(
                "{} candidates; took {} ({})",
                cands.len(),
                first.reference,
                first.match_kind
            )
        });
        Ok((SourceKey::Official, first.book_id, first.line_index, note))
    }

    /// (title, category, rows (line index, heRef, TOC path, raw text), url) for any source.
    #[allow(clippy::too_many_arguments)]
    fn book_rows(
        &self,
        conn: &rusqlite::Connection,
        src: &SourceKey,
        bid: i64,
        lo: i64,
        hi: i64,
        with_toc: bool,
        ps: Option<&PersonalSet>,
    ) -> Result<BookRows> {
        if *src == SourceKey::Official {
            let book = self
                .catalog
                .book(bid)
                .ok_or_else(|| bad(format!("no book {bid} in the library")))?;
            let raw = self.catalog.lines(conn, bid, lo, hi)?;
            let mut tocs: HashMap<Option<i64>, Vec<String>> = HashMap::new();
            let mut rows = Vec::with_capacity(raw.len());
            for r in raw {
                let toc = if with_toc {
                    if let std::collections::hash_map::Entry::Vacant(e) = tocs.entry(r.toc) {
                        let p = match r.toc {
                            Some(id) => self.catalog.toc_path(conn, id, 6)?,
                            None => Vec::new(),
                        };
                        e.insert(p);
                    }
                    tocs[&r.toc].clone()
                } else {
                    Vec::new()
                };
                rows.push((r.index, r.he_ref, toc, r.content));
            }
            return Ok(BookRows {
                title: book.title,
                category: self.catalog.cat_path(book.category),
                rows,
            });
        }
        let ps = ps.ok_or_else(|| bad("the personal index is not loaded"))?;
        let bi = *ps.by_key.get(&(src.clone(), bid)).ok_or_else(|| {
            bad(format!(
                "book {bid} ({}) is not in the personal index",
                match src {
                    SourceKey::User => "personal".to_string(),
                    SourceKey::Attached(s) => format!("database {s}"),
                    SourceKey::Official => String::new(),
                }
            ))
        })?;
        let b = &ps.books[bi];
        let (lines, _stale) = ps.lines_of(bi);
        let lines = lines.ok_or_else(|| anyhow!("cannot read {} right now", b.title))?;
        let mut rows = Vec::new();
        let (lo, hi) = (
            lo.max(0) as usize,
            (hi.max(0) as usize).min(lines.len().saturating_sub(1)),
        );
        if !lines.is_empty() {
            for li in lo..=hi {
                if li >= lines.len() {
                    break;
                }
                rows.push((li as i64, None, heading_path(&lines, li), lines[li].clone()));
            }
        }
        Ok(BookRows {
            title: b.title.clone(),
            category: b.category_path(),
            rows,
        })
    }

    pub fn text(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        self.refresh();
        let (src, bid, li, note) = self.locate(&opts)?;
        let before = opts.int("before", 2).clamp(0, 40);
        let after = opts.int("after", 6).clamp(0, 80);
        let conn = self.catalog.connect()?;
        let ps = self.personal_any();
        let br = self.book_rows(
            &conn,
            &src,
            bid,
            (li - before).max(0),
            li + after,
            true,
            ps.as_deref(),
        )?;
        let toc_first = br.rows.first().map(|r| r.2.clone()).unwrap_or_default();
        let lines: Vec<Value> = br
            .rows
            .iter()
            .map(|(lix, heref, _toc, content)| {
                json!({"line_index": lix, "he_ref": heref, "url": src.url(bid, *lix),
                       "heading": is_heading_line(content), "text": strip_html(content)})
            })
            .collect();
        Ok(
            json!({"book_id": bid, "book_title": br.title, "category": br.category, "source": src.code(),
                  "url_pattern": src.url(bid, 0).replace("index=0", "index=<line>"),
                  "center": li, "toc": toc_first, "note": note, "lines": lines, "timing_ms": {}}),
        )
    }

    // ── does this quotation exist, and where, and with which variants? ──────

    fn source_rank(&self, src: &SourceKey, bid: i64, ps: Option<&PersonalSet>) -> SourceRank {
        match src {
            SourceKey::Official => {
                self.catalog
                    .rank(bid)
                    .unwrap_or((ERA_UNKNOWN, 0, 3, 1, 2, vec![(1_000_000, 0)]))
            }
            SourceKey::Attached(_) => {
                let era = ps
                    .and_then(|p| p.by_key.get(&(src.clone(), bid)).map(|&i| &p.books[i]))
                    .and_then(|b| b.db.as_deref())
                    .map(|db| self.attached_era(db, bid))
                    .unwrap_or(ERA_UNKNOWN);
                (era, 1, 3, 1, 2, vec![(1_000_000, bid)])
            }
            SourceKey::User => (ERA_UNKNOWN, 2, 3, 1, 2, vec![(1_000_000, bid)]),
        }
    }

    fn attached_era(&self, db: &str, bid: i64) -> u8 {
        static CACHE: std::sync::LazyLock<Mutex<EraCache>> =
            std::sync::LazyLock::new(Default::default);
        let m = std::fs::metadata(db).and_then(|m| m.modified()).ok();
        let mut cache = CACHE.lock().unwrap();
        if cache.get(db).map(|e| e.0) != Some(m) {
            let eras = super::catalog::connect_ro(std::path::Path::new(db), false)
                .and_then(|c| super::catalog::book_eras(&c))
                .unwrap_or_default();
            cache.insert(db.to_string(), (m, eras));
        }
        cache[db].1.get(&bid).copied().unwrap_or(ERA_UNKNOWN)
    }

    pub fn check_quote(&self, req: &Map<String, Value>) -> Result<Value> {
        let opts = Opts(req);
        let quote = opts.str("quote").trim().to_string();
        if char_len(&quote) < 6 {
            return Err(bad("quote too short"));
        }
        if !opts.str("ref").is_empty()
            || !opts.str("url").is_empty()
            || opts.get("book_id").is_some()
        {
            return self.check_quote_at(&quote, &opts);
        }
        let mut o = req.clone();
        o.entry("k").or_insert(json!(40));
        let passage = self.official.read().unwrap().embed_passage(&quote)?;
        let run = self.run_queries(std::slice::from_ref(&quote), vec![passage], &Opts(&o))?;
        let ps = run.personal.as_deref();
        let mut best: HashMap<(u8, u64), (f32, Cand)> = HashMap::new();
        for hits in &run.per_query {
            for (s, c) in hits {
                let e = best.entry(c.key()).or_insert((*s, c.clone()));
                if *s > e.0 {
                    e.0 = *s;
                }
            }
        }
        let mut cands: Vec<(f32, Cand)> = best.into_values().collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        // A widely quoted line has many near-identical vectors (every work that quotes it), and
        // the base text's own line is often longer and scores lower than a bare quotation: the
        // pool has to be wide enough for the earliest source to be in it at all.
        cands.truncate(opts.int("candidates", 80).max(1) as usize);
        let (qw_orig, qw) = words_of(&quote);
        let t = Instant::now();
        let conn = self.catalog.connect()?;
        let mut seen: HashSet<(SourceKey, i64, i64)> = HashSet::new();
        let mut results: Vec<(Map<String, Value>, SourceRank)> = Vec::new();
        // every place of a candidate: the same text in several books is one vector of the set
        let places = cands.iter().flat_map(|(score, cand)| {
            self.cand_places(ps, run.scope.as_deref(), cand)
                .into_iter()
                .take(8)
                .map(move |p| (score, p))
        });
        for (score, place) in places {
            let (src, bid, li) = (place.src, place.bid, place.li);
            if src == SourceKey::Official && !self.catalog.has_book(bid) {
                continue;
            }
            if !seen.insert((src.clone(), bid, li)) {
                continue;
            }
            let Ok(br) = self.book_rows(&conn, &src, bid, (li - 1).max(0), li + 3, false, ps)
            else {
                continue;
            };
            let (mut words_orig, mut owner) = (Vec::new(), Vec::new());
            for (lix, _r, _t, content) in &br.rows {
                for w in strip_html(content).split_whitespace() {
                    words_orig.push(w.to_string());
                    owner.push(*lix);
                }
            }
            let mut cw = Vec::new();
            let mut wo = Vec::new();
            let mut ow = Vec::new();
            for (i, w) in words_orig.iter().enumerate() {
                let n = norm_word(w);
                if !n.is_empty() {
                    cw.push(n);
                    wo.push(w.clone());
                    ow.push(owner[i]);
                }
            }
            let Some(al) = align_words(&qw, &cw, &qw_orig, &wo) else {
                continue;
            };
            let start_line = ow.get(al.start).copied().unwrap_or(li);
            if start_line != li && seen.contains(&(src.clone(), bid, start_line)) {
                continue; // the same stretch, already found from the line it starts on
            }
            seen.insert((src.clone(), bid, start_line));
            let heref = br
                .rows
                .iter()
                .find(|r| r.0 == start_line)
                .and_then(|r| r.1.clone());
            let rank = self.source_rank(&src, bid, ps);
            let mut m = Map::new();
            m.insert("ratio".into(), json!(round_to(al.ratio, 3)));
            m.insert("score".into(), json!(round_to(*score as f64, 4)));
            m.insert("book_id".into(), json!(bid));
            m.insert("book_title".into(), json!(br.title));
            m.insert("category".into(), json!(br.category));
            m.insert("source".into(), json!(src.code()));
            m.insert("era".into(), json!(era_label(rank.0)));
            m.insert("line_index".into(), json!(start_line));
            m.insert(
                "he_ref".into(),
                json!(heref.unwrap_or_else(|| br.title.clone())),
            );
            m.insert("url".into(), json!(src.url(bid, start_line)));
            m.insert("library_text".into(), json!(wo[al.start..al.end].join(" ")));
            m.insert(
                "diffs".into(),
                diffs_json(&al.ops, &qw_orig, &wo[al.start..]),
            );
            results.push((m, rank));
        }
        let timing =
            json!({"embed": run.embed_ms, "scan": run.scan_ms, "align": t.elapsed().as_millis()});
        if results.is_empty() {
            return Ok(
                json!({"quote": quote, "verdict": "not found", "best": null, "source": null, "closest": null,
                             "alternatives": [], "timing_ms": timing}),
            );
        }
        let ratio = |m: &Map<String, Value>| m["ratio"].as_f64().unwrap_or(0.0);
        let score = |m: &Map<String, Value>| m["score"].as_f64().unwrap_or(0.0);
        let top_ratio = results
            .iter()
            .map(|(m, _)| ratio(m))
            .fold(f64::MIN, f64::max);
        // Match quality first, in tiers: within `source_tolerance` of the closest wording counts
        // as matching equally well, and inside a tier the earliest work leads. The default
        // allows one word of the quote (or 5%, for long quotes). Ratios are rounded to 3 decimals.
        let tol = match opts.get("source_tolerance") {
            Some(_) => opts.float("source_tolerance", 0.05),
            None => 0.05f64.max(1.0 / qw.len().max(1) as f64) + 0.002,
        };
        let tier = |x: f64| -> i32 {
            if x >= top_ratio - tol {
                4
            } else if x >= 0.97 {
                3
            } else if x >= 0.85 {
                2
            } else if x >= 0.6 {
                1
            } else {
                0
            }
        };
        let band = |x: f64| -> i32 {
            if x >= 0.97 {
                3
            } else if x >= 0.85 {
                2
            } else if x >= 0.6 {
                1
            } else {
                0
            }
        };
        results.sort_by(|(a, ra), (b, rb)| {
            tier(ratio(b))
                .cmp(&tier(ratio(a)))
                .then_with(|| ra.cmp(rb))
                .then_with(|| ratio(b).total_cmp(&ratio(a)))
                .then_with(|| score(b).total_cmp(&score(a)))
        });
        // The other places: exact matches before partial ones, earliest first in each.
        results[1..].sort_by(|(a, ra), (b, rb)| {
            band(ratio(b))
                .cmp(&band(ratio(a)))
                .then_with(|| ra.cmp(rb))
                .then_with(|| ratio(b).total_cmp(&ratio(a)))
                .then_with(|| score(b).total_cmp(&score(a)))
        });
        let src_hit = results[0].0.clone();
        // A later work is reported beside the source only when its wording matches better: the
        // earliest of those with the best wording.
        let closest_idx = if ratio(&src_hit) < top_ratio {
            results
                .iter()
                .skip(1)
                .position(|(m, _)| ratio(m) >= top_ratio)
                .map(|i| i + 1)
        } else {
            None
        };
        let verdict = verdict_of(top_ratio, "not found");
        let n_alt = opts.int("alternatives", 6).max(0) as usize;
        let alts: Vec<Value> = results
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(i, _)| Some(*i) != closest_idx)
            .take(n_alt)
            .map(|(_, (m, _))| Value::Object(m.clone()))
            .collect();
        Ok(json!({"quote": quote, "verdict": verdict,
                  "best": src_hit, "source": src_hit,
                  "closest": closest_idx.map(|i| Value::Object(results[i].0.clone())),
                  "alternatives": alts, "timing_ms": timing}))
    }

    /// Align a quotation against the text at a given location (the author's citation): no
    /// vector search, so it also finds words deep inside long lines.
    fn check_quote_at(&self, quote: &str, opts: &Opts) -> Result<Value> {
        self.refresh();
        let (src, bid, li, note) = self.locate(opts)?;
        let span = opts.int("span", 80).clamp(1, 400);
        let (qw_orig, qw) = words_of(quote);
        let conn = self.catalog.connect()?;
        let ps = self.personal_any();
        let br = self.book_rows(
            &conn,
            &src,
            bid,
            (li - 2).max(0),
            li + span,
            false,
            ps.as_deref(),
        )?;
        let line_words: Vec<(i64, Option<String>, Vec<String>)> = br
            .rows
            .iter()
            .map(|(lix, heref, _t, content)| {
                let ws = strip_html(content)
                    .split_whitespace()
                    .filter(|w| !norm_word(w).is_empty())
                    .map(String::from)
                    .collect();
                (*lix, heref.clone(), ws)
            })
            .collect();
        let rank = self.source_rank(&src, bid, ps.as_deref());
        let mut results: Vec<Map<String, Value>> = Vec::new();
        for n in 0..line_words.len() {
            let (mut words_orig, mut owner) = (Vec::new(), Vec::new());
            for (lix, _h, ws) in line_words.iter().skip(n).take(3) {
                words_orig.extend(ws.iter().cloned());
                owner.extend(std::iter::repeat_n(*lix, ws.len()));
            }
            let cw: Vec<String> = words_orig.iter().map(|w| norm_word(w)).collect();
            let Some(al) = align_words(&qw, &cw, &qw_orig, &words_orig) else {
                continue;
            };
            let start_line = owner.get(al.start).copied().unwrap_or(line_words[n].0);
            if start_line != line_words[n].0 {
                continue; // that window is found again from the line it starts on
            }
            let heref = line_words[n].1.clone();
            let mut m = Map::new();
            m.insert("ratio".into(), json!(round_to(al.ratio, 3)));
            m.insert("score".into(), Value::Null);
            m.insert("book_id".into(), json!(bid));
            m.insert("book_title".into(), json!(br.title));
            m.insert("category".into(), json!(br.category));
            m.insert("source".into(), json!(src.code()));
            m.insert("era".into(), json!(era_label(rank.0)));
            m.insert("line_index".into(), json!(start_line));
            m.insert(
                "he_ref".into(),
                json!(heref.unwrap_or_else(|| br.title.clone())),
            );
            m.insert("url".into(), json!(src.url(bid, start_line)));
            m.insert(
                "library_text".into(),
                json!(words_orig[al.start..al.end].join(" ")),
            );
            m.insert(
                "diffs".into(),
                diffs_json(&al.ops, &qw_orig, &words_orig[al.start..]),
            );
            results.push(m);
        }
        // a stable sort: equal ratios keep the order of their lines, as Python's sort does
        results.sort_by(|a, b| {
            b["ratio"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a["ratio"].as_f64().unwrap_or(0.0))
        });
        let top = results.first().cloned();
        let r = top
            .as_ref()
            .and_then(|t| t["ratio"].as_f64())
            .unwrap_or(0.0);
        let alts: Vec<Value> = results
            .iter()
            .skip(1)
            .take(3)
            .filter(|x| x["ratio"].as_f64().unwrap_or(0.0) >= 0.5)
            .cloned()
            .map(Value::Object)
            .collect();
        Ok(
            json!({"quote": quote, "verdict": verdict_of(r, "not found at this location"),
                  "best": top, "source": top, "closest": null, "located_by": "reference",
                  "note": note, "alternatives": alts, "timing_ms": {}}),
        )
    }

    // ── service status ──────────────────────────────────────────────────────

    pub fn health(&self) -> Value {
        let calibration = Calibration::summary(
            self.calibration().as_deref(),
            &self.calibration_candidates(),
        );
        let o = self.official.read().unwrap();
        let ps = self.personal_any();
        let (pending, last_error) = match &self.outbox {
            Some(ob) => (ob.pending_count(), ob.last_error()),
            None => (0, None),
        };
        json!({"ok": true, "version": VERSION, "server": "otzaria-mcp",
               "slots": o.slots(), "personal_slots": ps.as_ref().map_or(0, |p| p.n),
               "feedback_pending": pending, "feedback_error": last_error, "feedback": self.outbox.is_some(),
               "segments": o.segments(), "library_version": o.declared.library_version,
               "release_tag": o.declared.release_tag,
               "model": o.query_graph.file_name().unwrap_or_default().to_string_lossy(),
               "model_family": o.family(),
               "query_max_tokens": o.declared.model.query_max_tokens.unwrap_or(o.declared.model.max_tokens),
               "generation": o.declared.generation,
               "personal_searched": self.personal_searchable().is_some(),
               "personal_vectors": self.personal_vectors(),
               "personal_note": self.personal_note.lock().unwrap().clone(),
               "personal_model": ps.as_ref().and_then(|p| p.model().map(String::from)),
               "calibration": calibration,
               "uptime_s": self.started.elapsed().as_secs()})
    }

    pub fn categories(&self) -> Value {
        Value::Array(self.catalog.categories())
    }

    /// What changed in the personal books since the index was built, from
    /// `build_personal_index.py --check --json` run in the background (at most every 20 s).
    pub fn personal_status(&self) -> Value {
        let (snapshot, start) = {
            let mut st = self.status.lock().unwrap();
            let fresh = st
                .value
                .as_ref()
                .is_some_and(|(_, at)| at.elapsed().is_ok_and(|e| e < Duration::from_secs(20)));
            let start = !st.busy && !fresh && self.cfg.builder.is_some();
            if start {
                st.busy = true;
            }
            (st.value.clone(), start)
        };
        if start {
            let status = self.status.clone();
            let python = self.cfg.python.clone();
            let builder = self.cfg.builder.clone().unwrap();
            let otzaria = self.cfg.otzaria_dir.clone();
            let index = self.cfg.personal_dir.clone();
            std::thread::spawn(move || {
                let changes = run_check(&python, &builder, &otzaria, &index);
                let mut st = status.lock().unwrap();
                st.value = Some((changes, SystemTime::now()));
                st.busy = false;
            });
        }
        let busy = self.status.lock().unwrap().busy;
        let ps = self.personal_any();
        json!({"indexed": ps.as_ref().map(|p| p.info.clone()),
               "changes": snapshot.as_ref().map(|s| s.0.clone()).unwrap_or(json!({})),
               "checking": busy,
               "checked_at": snapshot.as_ref().and_then(|s| s.1.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()),
               "index_dir": self.cfg.personal_dir.to_string_lossy(),
               "builder": self.cfg.builder.as_ref().map(|b| b.to_string_lossy().into_owned()),
               "searched": self.personal_searchable().is_some(),
               "vectors": self.personal_vectors(),
               "note": self.personal_note.lock().unwrap().clone()})
    }

    pub fn feedback(&self, req: &Map<String, Value>) -> Result<Value> {
        let ob = self
            .outbox
            .as_ref()
            .ok_or_else(|| bad("feedback is not enabled in this server"))?;
        let opts = Opts(req);
        if let Some(Value::Object(pre)) = opts.get("event") {
            // An event built by another client (the Otzaria plugin "מקבילות", which cannot reach
            // the feedback service itself): relayed as is, under its own id.
            let fid = pre.get("id").and_then(Value::as_str).unwrap_or("");
            if !(8..=64).contains(&fid.len())
                || !fid.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
            {
                return Err(bad("event.id is missing or malformed"));
            }
            if !matches!(
                pre.get("feedbackType").and_then(Value::as_str),
                Some("like" | "dislike" | "entry")
            ) {
                return Err(bad("event.feedbackType must be like, dislike or entry"));
            }
            if !pre.get("result").is_some_and(Value::is_object) {
                return Err(bad("event.result is required"));
            }
            ob.enqueue(&Value::Object(pre.clone()))?;
            return Ok(
                json!({"queued": true, "id": fid, "pending": ob.pending_count(), "last_error": ob.last_error(), "timing_ms": {}}),
            );
        }
        let ftype = match opts.str("vote") {
            v if !v.is_empty() => v,
            _ => opts.str("type"),
        };
        if !matches!(ftype.as_str(), "like" | "dislike" | "entry") {
            return Err(bad("type must be like, dislike or entry"));
        }
        let hit = match opts.get("hit") {
            Some(Value::Object(h)) if !h.is_empty() => h.clone(),
            _ => return Err(bad("hit is required")),
        };
        let event = self.feedback_event(&opts, &ftype, &hit);
        let fid = event["id"].as_str().unwrap_or("").to_string();
        ob.enqueue(&event)?;
        Ok(
            json!({"queued": true, "id": fid, "pending": ob.pending_count(), "last_error": ob.last_error(), "timing_ms": {}}),
        )
    }

    fn feedback_event(&self, opts: &Opts, ftype: &str, hit: &Map<String, Value>) -> Value {
        use super::textutil::trunc16;
        let o = self.official.read().unwrap();
        let fid = uuid::Uuid::new_v4().to_string();
        let kind = opts.str("query_kind");
        let searched = opts.get("searched_at").and_then(Value::as_f64);
        let personal = matches!(
            hit.get("book_source").and_then(Value::as_str),
            Some("user" | "attached")
        ) || hit
            .get("personal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let source = hit
            .get("book_source")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| {
                if personal {
                    "user".into()
                } else {
                    "official".into()
                }
            });
        let section_prefix = match source.as_str() {
            "user" => "user:".to_string(),
            "attached" => format!(
                "db:{}:",
                hit.get("library_id").and_then(Value::as_str).unwrap_or("")
            ),
            _ => String::new(),
        };
        let paragraph = hit
            .get("text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| hit.get("snippet").and_then(Value::as_str))
            .unwrap_or("");
        json!({
            "id": fid, "eventId": fid, "schemaVersion": 1, "occurredAt": super::feedback::utc_iso(SystemTime::now()),
            "feedbackType": ftype,
            "software": "Otzaria Sources (Obsidian plugin)", "softwareVersion": VERSION,
            "host": {"otzariaVersion": self.otzaria_version(), "client": "obsidian"},
            "search": {
                "searchId": opts.get("session_id").map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).unwrap_or_else(|| fid.clone()),
                "query": trunc16(opts.str("query").trim(), 2000),
                "mode": if kind == "note_window" { "semantic-note-paragraph" } else { "semantic-query" },
                "bookIds": [],
                "settings": {
                    "categories": opts.list("categories"),
                    "excludeCategories": opts.list("exclude_categories"),
                    "resultCount": opts.get("k"), "minScore": opts.get("min_score"),
                    "maxPerBook": opts.get("max_per_book"), "mergeNearbyLines": opts.get("merge_lines"),
                    "includesPersonalBooks": opts.bool("include_personal", true),
                },
                "searchedAt": searched.map(|ms| super::feedback::utc_iso(SystemTime::UNIX_EPOCH + Duration::from_secs_f64(ms / 1000.0))),
                "latencyMs": opts.get("latency_ms"),
                "shownCount": match opts.get("shown") { Some(Value::Array(a)) if !a.is_empty() => json!(a.len()), _ => Value::Null },
            },
            "modelVersion": format!("{} (query graph: {})", o.family(), o.query_graph.file_name().unwrap_or_default().to_string_lossy()),
            "vectors": {"releaseTag": o.declared.release_tag, "libraryVersion": o.declared.library_version, "personalBook": personal},
            "result": {
                "bookId": hit.get("book_id"), "bookSource": source,
                "bookTitle": hit.get("book_title"),
                "sectionId": format!("{section_prefix}{}:{}", val_str(hit.get("book_id")), val_str(hit.get("line_index"))),
                "reference": hit.get("he_ref"),
                "paragraph": trunc16(paragraph, 20000),
                "matchedParts": [],
                "position": opts.get("rank"),
                "score": hit.get("score"),
                "url": hit.get("url"),
            },
        })
    }

    fn otzaria_version(&self) -> Option<String> {
        let text = std::fs::read(self.cfg.otzaria_dir.join("logs").join("errors.txt")).ok()?;
        let text = String::from_utf8_lossy(&text);
        text.lines()
            .filter_map(|l| l.strip_prefix("Version:"))
            .map(|v| v.split_whitespace().next().unwrap_or("").to_string())
            .rfind(|v| !v.is_empty())
    }
}

pub struct BookRows {
    pub title: String,
    pub category: Vec<String>,
    pub rows: Vec<(i64, Option<String>, Vec<String>, String)>,
}

/// A personal or database book's passage in `/parallels`.
fn personal_parallel(
    b: &super::personal::PersonalBook,
    key: String,
    line: i64,
    window: u8,
) -> Value {
    let kind = if b.attached() { "attached" } else { "user" };
    json!({"store": "personal", "key": key, "source": kind, "kind": kind,
           "id": b.book_id, "bookUid": b.set_name(), "title": b.title, "line": line,
           "window": window})
}

fn val_str(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => "None".into(),
        Some(v) => v.to_string(),
    }
}

fn is_heading_line(content: &str) -> bool {
    let t = content.trim_start();
    t.len() > 2
        && t.as_bytes()[0] == b'<'
        && (t.as_bytes()[1] == b'h' || t.as_bytes()[1] == b'H')
        && t.as_bytes()[2].is_ascii_digit()
}

/// `re.fullmatch(r".*\s\d+", heref)`.
fn ends_with_space_number(s: &str) -> bool {
    let t: Vec<char> = s.chars().collect();
    let mut i = t.len();
    while i > 0 && t[i - 1].is_numeric() && t[i - 1].is_ascii_digit() {
        i -= 1;
    }
    i < t.len() && i > 0 && t[i - 1].is_whitespace() && !s.contains('\n')
}

fn words_of(quote: &str) -> (Vec<String>, Vec<String>) {
    let orig: Vec<String> = quote
        .split_whitespace()
        .filter(|w| !norm_word(w).is_empty())
        .map(String::from)
        .collect();
    let norm = orig.iter().map(|w| norm_word(w)).collect();
    (orig, norm)
}

fn diffs_json(ops: &[(Op, usize, usize, usize, usize)], qo: &[String], window: &[String]) -> Value {
    Value::Array(
        ops.iter()
            .map(|(op, i1, i2, j1, j2)| {
                json!({"op": op, "quote": qo[*i1..*i2].join(" "), "library": window[*j1..*j2].join(" ")})
            })
            .collect(),
    )
}

fn verdict_of(r: f64, none: &str) -> String {
    if r >= 0.97 {
        "exact".into()
    } else if r >= 0.85 {
        "close, with variants".into()
    } else if r >= 0.6 {
        "partial or paraphrased".into()
    } else {
        none.into()
    }
}

fn run_check(
    python: &str,
    builder: &std::path::Path,
    otzaria: &std::path::Path,
    index: &std::path::Path,
) -> Value {
    let mut cmd = std::process::Command::new(python);
    cmd.arg(builder)
        .arg("--check")
        .arg("--json")
        .arg("--otzaria-dir")
        .arg(otzaria)
        .arg("--index-dir")
        .arg(index);
    cmd.stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.env("PYTHONIOENCODING", "utf-8");
    if let Some(dir) = builder.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let last = text
                .lines()
                .rev()
                .find(|l| l.trim_start().starts_with('{'))
                .unwrap_or("");
            let mut v: Value = serde_json::from_str(last)
                .unwrap_or_else(|_| json!({"error": "the builder printed no JSON"}));
            if let Value::Object(m) = &mut v {
                m.entry("legacy").or_insert(json!(false));
            }
            v
        }
        Ok(out) => {
            json!({"error": format!("the builder's check failed (exit {:?})", out.status.code())})
        }
        Err(e) => json!({"error": format!("cannot run {python}: {e}")}),
    }
}

#[allow(dead_code)]
fn ensure(cond: bool, msg: &str) -> Result<()> {
    if !cond {
        bail!("{msg}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(book: &str, hint: u32, window: u8) -> RecordRef {
        RecordRef {
            book: book.into(),
            hint,
            window,
            anchor: None,
        }
    }

    fn personal(name: &str) -> Option<(SourceKey, i64, usize)> {
        match name {
            "uid:7" => Some((SourceKey::User, 70, 0)), // renumbered by Otzaria since the index
            "db:x:3" => Some((SourceKey::Attached("x".into()), 3, 1)),
            _ => None,
        }
    }

    #[test]
    fn places_put_the_library_first_and_keep_the_records_order() {
        let records = [
            rec("db:x:3", 11, 0),
            rec("id:5", 40, 0),
            rec("uid:7", 2, 1),
            rec("id:9", 8, 255),
            rec("uid:99", 1, 0), // no such book
            rec("id:404", 1, 0), // not in the library
        ];
        let places = record_places(&records, None, |b| b != 404, personal);
        let seen: Vec<(String, i64, i64, u8, Option<usize>)> = places
            .iter()
            .map(|p| (p.src.code(), p.bid, p.li, p.win, p.book))
            .collect();
        assert_eq!(
            seen,
            vec![
                ("o".into(), 5, 40, 0, None),
                ("o".into(), 9, 8, 255, None),
                ("d:x".into(), 3, 11, 0, Some(1)),
                ("u".into(), 70, 2, 1, Some(0)),
            ]
        );
    }

    #[test]
    fn places_outside_the_scope_or_without_a_personal_set_are_left_out() {
        let records = [rec("db:x:3", 11, 0), rec("id:5", 40, 0), rec("uid:7", 2, 0)];
        let scope: BookSet = ["uid:7", "id:5"].into_iter().collect();
        let places = record_places(&records, Some(&scope), |_| true, personal);
        assert_eq!(
            places.iter().map(|p| p.bid).collect::<Vec<_>>(),
            vec![5, 70]
        );
        // --scan-personal-index: the set's records of books outside the library are ignored
        let places = record_places(&records, None, |_| true, |_| None);
        assert_eq!(places.len(), 1);
        assert_eq!(places[0].src, SourceKey::Official);
    }

    #[test]
    fn a_line_beside_an_also_in_place_is_covered() {
        let mut covered = Covered::default();
        covered.add(&SourceKey::User, 2492, 11, 0);
        covered.add(&SourceKey::User, 2492, 40, 3);
        let book = (SourceKey::User, 2492);
        assert_eq!(covered.find(&book, 13, 2), Some(0));
        assert_eq!(covered.find(&book, 14, 2), None);
        assert_eq!(covered.find(&book, 39, 2), Some(3));
        assert_eq!(covered.find(&book, 13, 0), None);
        assert_eq!(covered.find(&(SourceKey::Official, 2492), 11, 2), None);
    }
}

//! The library as the semantic tools see it: books, categories, the order in which
//! `check_quote` lists sources, and line text read the way the Python server read it
//! (`line_content` as stored, with heRef and TOC path). No connection is held between
//! requests, so Otzaria can replace `seforim.db` at any time; the catalogue is reloaded when
//! the file's modification time changes.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use anyhow::{Context, Result};
use rusqlite::types::Value;
use rusqlite::{Connection, OpenFlags, OptionalExtension};

use crate::text::LineCodec;

use super::textutil::strip_html;

/// A book's place among the sources of a quotation: era, source (library, database,
/// personal), corpus layer inside חז"ל, not-a-base-book, dependence, catalogue position.
/// Smaller is earlier.
pub type SourceRank = (u8, u8, u8, u8, u8, Vec<(i64, i64)>);

pub const ERA_UNKNOWN: u8 = 5;

pub fn era_label(rank: u8) -> Option<&'static str> {
    match rank {
        0 => Some("תורה שבכתב"),
        1 => Some("חז\"ל"),
        2 => Some("ראשונים"),
        3 => Some("אחרונים"),
        4 => Some("מחברי זמננו"),
        _ => None,
    }
}

fn strip_gershayim(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '"' | '\'' | '\u{05F3}' | '\u{05F4}'))
        .collect::<String>()
        .trim()
        .to_string()
}

/// `era_rank`: a `generation.name` to its order.
pub fn era_rank(name: &str) -> u8 {
    match strip_gershayim(name).as_str() {
        "תורה שבכתב" => 0,
        "חזל" => 1,
        "ראשונים" => 2,
        "אחרונים" => 3,
        "מחברי זמננו" => 4,
        _ => ERA_UNKNOWN,
    }
}

/// bookId -> era (earliest generation), for any Otzaria-format book database.
pub fn book_eras(conn: &Connection) -> Result<HashMap<i64, u8>> {
    let mut eras = HashMap::new();
    if !(has_table(conn, "book_generation")? && has_table(conn, "generation")?) {
        return Ok(eras);
    }
    let mut stmt = conn.prepare(
        "SELECT bg.bookId, g.name FROM book_generation bg JOIN generation g ON g.id = bg.generationId",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (bid, name) = row?;
        let r = era_rank(&name);
        let e = eras.entry(bid).or_insert(ERA_UNKNOWN + 1);
        if r < *e {
            *e = r;
        }
    }
    eras.retain(|_, v| *v <= ERA_UNKNOWN);
    Ok(eras)
}

/// Inside חז"ל the catalogue's top level is not chronological (תלמוד בבלי sorts before משנה),
/// so the corpus layers come first: tannaitic texts, the Yerushalmi, the Bavli, then the rest.
fn chazal_layer(titles: &[String]) -> u8 {
    let t: Vec<String> = titles.iter().map(|x| strip_gershayim(x)).collect();
    match (t.first().map(String::as_str), t.get(1).map(String::as_str)) {
        (Some("מדרש"), Some("הלכה")) => 0,
        (Some("תלמוד בבלי"), Some("מסכתות קטנות")) => 3,
        (Some("משנה" | "תוספתא" | "בית שני"), _) => 0,
        (Some("תלמוד ירושלמי"), _) => 1,
        (Some("תלמוד בבלי"), _) => 2,
        _ => 3,
    }
}

fn dep_rank(dep: Option<&str>) -> u8 {
    match dep.map(|d| d.trim().to_lowercase()).as_deref() {
        None | Some("") => 0,
        Some("targum" | "midrash") => 1,
        _ => 2,
    }
}

pub fn has_table(conn: &Connection, name: &str) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
            [name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn columns(conn: &Connection, table: &str) -> Result<HashSet<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let cols = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<rusqlite::Result<HashSet<_>>>()?;
    Ok(cols)
}

#[derive(Debug, Clone)]
pub struct Book {
    pub title: String,
    pub he_ref: Option<String>,
    pub category: i64,
}

/// One line as `lines()` returns it: index, heRef, TOC entry id, raw text.
pub struct RawLine {
    pub index: i64,
    pub he_ref: Option<String>,
    pub toc: Option<i64>,
    pub content: String,
}

struct Loaded {
    mtime: Option<SystemTime>,
    books: HashMap<i64, Book>,
    cats: HashMap<i64, (Option<i64>, String)>,
    closure: HashMap<i64, HashSet<i64>>,
    ranks: HashMap<i64, SourceRank>,
    line_sql: String,
    db_version: Option<String>,
}

pub struct Catalog {
    path: PathBuf,
    immutable: bool,
    state: std::sync::RwLock<std::sync::Arc<Loaded>>,
    codec: Mutex<Option<(Option<SystemTime>, LineCodec)>>,
}

fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl Catalog {
    pub fn open(path: &Path, immutable: bool) -> Result<Catalog> {
        let loaded = Self::load(path, immutable)?;
        Ok(Catalog {
            path: path.to_path_buf(),
            immutable,
            state: std::sync::RwLock::new(std::sync::Arc::new(loaded)),
            codec: Mutex::new(None),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A read-only connection; `immutable=1` in snapshot mode (WAL databases on read-only media).
    pub fn connect(&self) -> Result<Connection> {
        connect_ro(&self.path, self.immutable)
    }

    fn load(path: &Path, immutable: bool) -> Result<Loaded> {
        let m = mtime(path);
        let conn = connect_ro(path, immutable)?;
        let mut books = HashMap::new();
        {
            let mut stmt = conn.prepare("SELECT id, title, heRef, categoryId FROM book")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    Book {
                        title: r.get(1)?,
                        he_ref: r.get(2)?,
                        category: r.get(3)?,
                    },
                ))
            })?;
            for row in rows {
                let (id, b) = row?;
                books.insert(id, b);
            }
        }
        let ccols = columns(&conn, "category")?;
        let mut cats = HashMap::new();
        let mut corder = HashMap::new();
        {
            let sel = if ccols.contains("orderIndex") {
                "orderIndex"
            } else {
                "999"
            };
            let mut stmt =
                conn.prepare(&format!("SELECT id, parentId, title, {sel} FROM category"))?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<i64>>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                ))
            })?;
            for row in rows {
                let (id, parent, title, order) = row?;
                cats.insert(id, (parent, title));
                corder.insert(id, order.unwrap_or(999));
            }
        }
        let mut closure: HashMap<i64, HashSet<i64>> = HashMap::new();
        if has_table(&conn, "category_closure")? {
            let mut stmt = conn.prepare("SELECT ancestorId, descendantId FROM category_closure")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
            for row in rows {
                let (a, d) = row?;
                closure.entry(a).or_default().insert(d);
            }
        }
        let bcols = columns(&conn, "book")?;
        let col = |c: &str| {
            if bcols.contains(c) {
                c.to_string()
            } else {
                "NULL".to_string()
            }
        };
        let eras = book_eras(&conn)?;
        let mut ranks = HashMap::new();
        {
            let sql = format!(
                "SELECT id, categoryId, {}, {}, {} FROM book",
                col("isBaseBook"),
                col("dependenceType"),
                col("orderIndex")
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                ))
            })?;
            for row in rows {
                let (bid, cid, is_base, dep, order) = row?;
                let ids = id_path(&cats, cid);
                let mut catalogue: Vec<(i64, i64)> = ids
                    .iter()
                    .map(|c| (*corder.get(c).unwrap_or(&999), *c))
                    .collect();
                catalogue.push((order.unwrap_or(999), bid));
                let era = *eras.get(&bid).unwrap_or(&ERA_UNKNOWN);
                let layer = if era == 1 {
                    let titles: Vec<String> = ids
                        .iter()
                        .filter_map(|c| cats.get(c).map(|x| x.1.clone()))
                        .collect();
                    chazal_layer(&titles)
                } else {
                    0
                };
                let not_base = if is_base.unwrap_or(0) != 0 { 0 } else { 1 };
                ranks.insert(
                    bid,
                    (era, 0, layer, not_base, dep_rank(dep.as_deref()), catalogue),
                );
            }
        }
        // the line query, by what this schema has
        let lcols = columns(&conn, "line")?;
        let has_content_table = has_table(&conn, "line_content")?;
        let has_line_toc = has_table(&conn, "line_toc")?;
        let toc = match (lcols.contains("tocEntryId"), has_line_toc) {
            (true, true) => "COALESCE(l.tocEntryId, lt.tocEntryId)",
            (true, false) => "l.tocEntryId",
            (false, true) => "lt.tocEntryId",
            (false, false) => "NULL",
        };
        let content = if has_content_table {
            "c.content"
        } else {
            "l.content"
        };
        let mut line_sql = format!("SELECT l.lineIndex, l.heRef, {toc}, {content} FROM line l ");
        if has_content_table {
            line_sql.push_str("LEFT JOIN line_content c ON c.id = l.id ");
        }
        if has_line_toc {
            line_sql.push_str("LEFT JOIN line_toc lt ON lt.lineId = l.id ");
        }
        line_sql
            .push_str("WHERE l.bookId = ?1 AND l.lineIndex BETWEEN ?2 AND ?3 ORDER BY l.lineIndex");
        let db_version = if has_table(&conn, "schema_meta")? {
            conn.query_row(
                "SELECT value FROM schema_meta WHERE key = 'db_version'",
                [],
                |r| r.get::<_, Value>(0),
            )
            .optional()?
            .map(|v| match v {
                Value::Integer(i) => i.to_string(),
                Value::Text(s) => s,
                _ => String::new(),
            })
        } else {
            None
        };
        Ok(Loaded {
            mtime: m,
            books,
            cats,
            closure,
            ranks,
            line_sql,
            db_version,
        })
    }

    fn loaded(&self) -> std::sync::Arc<Loaded> {
        self.state.read().unwrap().clone()
    }

    /// Reload when the file changed; true when it did.
    pub fn maybe_reopen(&self) -> bool {
        let m = mtime(&self.path);
        if m == self.loaded().mtime {
            return false;
        }
        match Self::load(&self.path, self.immutable) {
            Ok(l) => {
                *self.state.write().unwrap() = std::sync::Arc::new(l);
                *self.codec.lock().unwrap() = None;
                true
            }
            Err(e) => {
                eprintln!("otzaria-mcp: semantic catalogue reload failed: {e:#}");
                false
            }
        }
    }

    pub fn db_version(&self) -> Option<String> {
        self.loaded().db_version.clone()
    }

    pub fn book(&self, id: i64) -> Option<Book> {
        self.loaded().books.get(&id).cloned()
    }

    pub fn has_book(&self, id: i64) -> bool {
        self.loaded().books.contains_key(&id)
    }

    pub fn rank(&self, id: i64) -> Option<SourceRank> {
        self.loaded().ranks.get(&id).cloned()
    }

    pub fn cat_path(&self, cid: i64) -> Vec<String> {
        let l = self.loaded();
        id_path(&l.cats, cid)
            .into_iter()
            .filter_map(|c| l.cats.get(&c).map(|x| x.1.clone()))
            .collect()
    }

    /// Category titles or ids -> the books filed under them (descendant categories included).
    pub fn books_in_categories(&self, specs: &[String]) -> HashSet<i64> {
        let l = self.loaded();
        let mut want = HashSet::new();
        for s in specs {
            let s = s.trim();
            if s.is_empty() {
                continue;
            }
            if s.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(id) = s.parse() {
                    want.insert(id);
                }
            } else {
                want.extend(
                    l.cats
                        .iter()
                        .filter(|(_, (_, t))| t.trim() == s)
                        .map(|(id, _)| *id),
                );
            }
        }
        let mut cats = want.clone();
        for c in &want {
            if let Some(d) = l.closure.get(c) {
                cats.extend(d);
            }
        }
        l.books
            .iter()
            .filter(|(_, b)| cats.contains(&b.category))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Every category with its book count, as the plugin's /categories expects.
    pub fn categories(&self) -> Vec<serde_json::Value> {
        let l = self.loaded();
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for b in l.books.values() {
            *counts.entry(b.category).or_default() += 1;
        }
        let mut ids: Vec<&i64> = l.cats.keys().collect();
        ids.sort();
        ids.into_iter()
            .map(|id| {
                let (p, t) = &l.cats[id];
                serde_json::json!({"id": id, "parent": p, "title": t, "books": counts.get(id).copied().unwrap_or(0)})
            })
            .collect()
    }

    pub fn all_book_ids(&self) -> HashSet<i64> {
        self.loaded().books.keys().copied().collect()
    }

    pub fn book_count(&self) -> usize {
        self.loaded().books.len()
    }

    /// Lines `lo..=hi` of a book, raw (`line_content` as stored).
    pub fn lines(&self, conn: &Connection, book: i64, lo: i64, hi: i64) -> Result<Vec<RawLine>> {
        let l = self.loaded();
        let mut guard = self.codec.lock().unwrap();
        if guard.as_ref().map(|(m, _)| *m) != Some(l.mtime) {
            *guard = Some((l.mtime, LineCodec::load(conn)?));
        }
        let codec = &mut guard.as_mut().unwrap().1;
        let mut stmt = conn.prepare_cached(&l.line_sql)?;
        let mut rows = stmt.query(rusqlite::params![book, lo, hi])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let content: Value = r.get(3)?;
            out.push(RawLine {
                index: r.get(0)?,
                he_ref: r.get(1)?,
                toc: r.get(2)?,
                content: codec.text_lossy(&content),
            });
        }
        Ok(out)
    }

    /// TOC headings from the root down to `toc_id`, at most `limit`.
    pub fn toc_path(&self, conn: &Connection, toc_id: i64, limit: usize) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let mut cur = Some(toc_id);
        let mut stmt = conn.prepare_cached(
            "SELECT e.parentId, t.text FROM tocEntry e JOIN tocText t ON t.id = e.textId WHERE e.id = ?1",
        )?;
        while let Some(id) = cur {
            if out.len() >= limit {
                break;
            }
            let row = stmt
                .query_row([id], |r| {
                    Ok((r.get::<_, Option<i64>>(0)?, r.get::<_, String>(1)?))
                })
                .optional()?;
            match row {
                Some((parent, text)) => {
                    out.push(strip_html(&text));
                    cur = parent;
                }
                None => break,
            }
        }
        out.reverse();
        Ok(out)
    }
}

fn id_path(cats: &HashMap<i64, (Option<i64>, String)>, cid: i64) -> Vec<i64> {
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    let mut c = Some(cid);
    while let Some(id) = c {
        if !cats.contains_key(&id) || !seen.insert(id) {
            break;
        }
        ids.push(id);
        c = cats[&id].0;
    }
    ids.reverse();
    ids
}

pub fn connect_ro(path: &Path, immutable: bool) -> Result<Connection> {
    let conn = if immutable {
        let uri = format!("file:{}?immutable=1", uri_path(path));
        Connection::open_with_flags(
            uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )
    } else {
        Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
    }
    .with_context(|| format!("cannot open {}", path.display()))?;
    conn.execute_batch("PRAGMA query_only=ON;").ok();
    Ok(conn)
}

/// A path for a `file:` URI: forward slashes, and the characters a URI reserves escaped
/// (Hebrew, spaces, `#` or `?` in a folder name).
pub fn uri_path(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"/:-_.~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    if out.len() > 1 && out.as_bytes()[1] == b':' {
        out.insert(0, '/'); // file:/C:/...
    }
    out
}

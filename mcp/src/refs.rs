//! Reference resolution ("בראשית א א", "שו\"ע או\"ח א א", "ברכות ב:") → (book_id, line_index).
//!
//! Mirrors the app's find_ref approach: an in-memory index of NORMALIZED book
//! names (title + heRef + every book_acronym alias) → book ids, then the trailing
//! address tokens are matched against normalized `line.heRef` within each candidate
//! book. Normalization strips nikud/te'amim and gershayim, turns Talmud amud markers
//! `.`→" א" and `:`→" ב", and drops commas/maqaf — applied identically to query and
//! heRef so they line up. This is the biggest correctness-risk tool; it is covered by
//! a golden test corpus (see tests/).

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

use crate::line_text::LineText;

/// Normalize a reference/name for matching. Applied to book names, acronyms,
/// `line.heRef`, and the user query alike.
pub fn normalize_ref(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            // Separators → space. Maqaf (U+05BE) must be handled before the nikud
            // range below (it falls inside 0591–05C7) so words don't get glued.
            ',' | '־' | '-' | '\u{2013}' | '\u{2014}' => out.push(' '),
            '"' | '\u{05F4}' => {}      // gershayim (ASCII + HEBREW GERSHAYIM)
            '\'' | '\u{05F3}' => {}     // geresh
            '.' => out.push_str(" א "), // Talmud daf: amud alef
            ':' => out.push_str(" ב "), // Talmud daf: amud bet
            '\u{0591}'..='\u{05C7}' => {} // Hebrew nikud + te'amim
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Token-boundary prefix test on normalized (space-separated) strings.
fn starts_with_tokens(hay: &str, needle: &str) -> bool {
    hay == needle
        || (!needle.is_empty() && hay.starts_with(needle) && hay[needle.len()..].starts_with(' '))
}

pub struct RefIndex {
    /// normalized name → book ids (a name like "ברכות" can map to several books).
    name_to_books: HashMap<String, Vec<i64>>,
    book_heref: HashMap<i64, String>,
    book_title: HashMap<i64, String>,
    /// bookId → (is_base_book, orderIndex) for ranking candidates.
    book_meta: HashMap<i64, (bool, i64)>,
    /// Lazily built normalized heRef index for each requested book.
    line_cache: Mutex<HashMap<i64, BTreeMap<String, CachedLine>>>,
}

#[derive(Clone)]
struct CachedLine {
    line_index: i64,
    he_ref: String,
}

fn add_name(map: &mut HashMap<String, Vec<i64>>, name: &str, id: i64) {
    let n = normalize_ref(name);
    if n.is_empty() {
        return;
    }
    let v = map.entry(n).or_default();
    if !v.contains(&id) {
        v.push(id);
    }
}

impl RefIndex {
    pub fn build(conn: &Connection) -> RefIndex {
        let mut name_to_books: HashMap<String, Vec<i64>> = HashMap::new();
        let mut book_heref = HashMap::new();
        let mut book_title = HashMap::new();
        let mut book_meta = HashMap::new();

        if let Ok(mut stmt) =
            conn.prepare("SELECT id, title, heRef, isBaseBook, orderIndex FROM book")
        {
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            });
            if let Ok(rows) = rows {
                for (id, title, heref, is_base, order) in rows.flatten() {
                    let href = heref.clone().unwrap_or_else(|| title.clone());
                    add_name(&mut name_to_books, &title, id);
                    if let Some(h) = &heref {
                        add_name(&mut name_to_books, h, id);
                    }
                    book_title.insert(id, title);
                    book_heref.insert(id, href);
                    book_meta.insert(id, (is_base != 0, order));
                }
            }
        }

        if let Ok(mut stmt) = conn.prepare("SELECT bookId, term FROM book_acronym") {
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)));
            if let Ok(rows) = rows {
                for (id, term) in rows.flatten() {
                    add_name(&mut name_to_books, &term, id);
                }
            }
        }

        RefIndex {
            name_to_books,
            book_heref,
            book_title,
            book_meta,
            line_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Lines of `book_id` whose normalized heRef equals or extends `canonical` (token
    /// boundary), in line order.
    fn matching_lines(
        &self,
        conn: &Connection,
        book_id: i64,
        canonical: &str,
    ) -> Result<Vec<CachedLine>> {
        let mut cache = self.line_cache.lock().unwrap();
        if let Entry::Vacant(entry) = cache.entry(book_id) {
            let mut stmt = conn.prepare(
                "SELECT lineIndex, heRef FROM line \
                  WHERE bookId = ?1 AND heRef IS NOT NULL ORDER BY lineIndex",
            )?;
            let mut lines = BTreeMap::new();
            for row in stmt.query_map([book_id], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })? {
                let (line_index, he_ref) = row?;
                let normalized = normalize_ref(&he_ref);
                lines
                    .entry(normalized)
                    .or_insert(CachedLine { line_index, he_ref });
            }
            entry.insert(lines);
        }

        let Some(lines) = cache.get(&book_id) else {
            return Ok(vec![]);
        };
        let mut found: Vec<CachedLine> = lines
            .range(canonical.to_string()..)
            .take_while(|(he_ref, _)| he_ref.starts_with(canonical))
            .filter(|(he_ref, _)| starts_with_tokens(he_ref, canonical))
            .map(|(_, line)| line.clone())
            .collect();
        found.sort_by_key(|line| line.line_index);
        Ok(found)
    }
}

#[derive(Debug, Serialize)]
pub struct RefCandidate {
    pub book_id: i64,
    pub book_title: String,
    pub line_index: i64,
    pub he_ref: String,
    pub snippet: String,
    /// "exact" (normalized heRef == query) or "prefix" (query is a leading portion).
    pub match_kind: &'static str,
}

/// Resolve a human reference to candidate (book_id, line_index) locations.
///
/// The query is split into a book name (the longest leading run of tokens that names a
/// book; shorter runs are tried if it yields nothing) and an address. The address is
/// matched as written first; only if that finds nothing in a book are cleaned-up forms
/// tried: unit markers dropped (`פ"ב ה"א` → `ב א`, `סימן`, `סעיף`, `ס"ק`, `דף`),
/// `ע"א`/`ע"ב` read as an amud, a Shulchan Aruch part name dropped after a nosei kelim
/// (`מגן אברהם אורח חיים שח ב`), and leftover words of the book's own title dropped
/// (`קצות החושן כח א`, where only `קצות` is an alias). A trailing `ד"ה <words>` picks the
/// line of that dibbur hamatchil within the matched location.
pub fn resolve(
    conn: &Connection,
    line_text: &LineText,
    index: &RefIndex,
    query: &str,
    limit: i64,
) -> Result<Vec<RefCandidate>> {
    let (ref_part, dibbur) = split_dibbur(query);
    let raw: Vec<&str> = ref_part.split_whitespace().collect();
    let q_tokens: Vec<String> = normalize_ref(&ref_part)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if q_tokens.is_empty() {
        return Ok(vec![]);
    }

    let max_k = q_tokens.len().min(12);
    let mut out: Vec<RefCandidate> = Vec::new();
    let mut named_a_book = false;
    for k in (1..=max_k).rev() {
        let Some(book_ids) = index.name_to_books.get(&q_tokens[..k].join(" ")) else {
            continue;
        };
        named_a_book = true;
        let address = q_tokens[k..].join(" ");
        let cleaned = raw_rest(&raw, k)
            .map(|rest| normalize_ref(&clean_address(&rest).join(" ")))
            .unwrap_or_else(|| address.clone());
        let cleaned = drop_leading(&cleaned, |t| PART_NAMES.contains(&t));

        for &bid in book_ids {
            if out.len() as i64 >= limit {
                break;
            }
            let title = index.book_title.get(&bid).cloned().unwrap_or_default();
            let book_href = index.book_heref.get(&bid).cloned().unwrap_or_default();
            let book_norm = normalize_ref(&book_href);
            let own_words: Vec<String> = normalize_ref(&format!("{title} {book_href}"))
                .split_whitespace()
                .map(str::to_string)
                .collect();
            let without_title = drop_leading(&cleaned, |t| own_words.iter().any(|w| w == t));

            let mut variants: Vec<&str> = vec![address.as_str()];
            for v in [cleaned.as_str(), without_title.as_str()] {
                // A cleanup that leaves nothing would just point at the book's start.
                if !variants.contains(&v) && (!v.is_empty() || address.is_empty()) {
                    variants.push(v);
                }
            }
            for v in variants {
                let mut canonical = book_norm.clone();
                if !v.is_empty() {
                    canonical.push(' ');
                    canonical.push_str(v);
                }
                if let Some(c) = first_line_matching(
                    conn,
                    line_text,
                    index,
                    bid,
                    &title,
                    &canonical,
                    dibbur.as_deref(),
                )? {
                    out.push(c);
                    break;
                }
            }
        }
        if !out.is_empty() {
            break;
        }
    }

    if !named_a_book || out.is_empty() {
        return fallback_heref_like(conn, line_text, &ref_part, limit);
    }

    // Rank: exact before prefix; base texts before commentaries; then catalogue order.
    out.sort_by_key(|c| {
        let (is_base, order) = index
            .book_meta
            .get(&c.book_id)
            .copied()
            .unwrap_or((false, i64::MAX));
        let mk = match c.match_kind {
            "exact" | "dibbur" => 0,
            "prefix" => 1,
            _ => 2,
        };
        (mk, !is_base, order, c.book_id)
    });
    Ok(out)
}

/// Words that only say what kind of number follows ("פרק ב", "סימן רסה", "דף מג"), after
/// normalization. None of them is a well-formed Hebrew numeral.
const MARKER_WORDS: &[&str] = &[
    "פרק", "הלכה", "סימן", "סי", "סעיף", "סע", "סק", "דף", "עמוד", "משנה", "אות",
];

/// Shulchan Aruch / Tur part names, written after a nosei kelim that covers one part.
const PART_NAMES: &[&str] = &[
    "אורח", "חיים", "אוח", "יורה", "דעה", "יוד", "אבן", "העזר", "אהע", "חושן", "משפט", "חומ",
];

/// Split off a trailing dibbur hamatchil: `... ד"ה ופורסין` → (`...`, Some(words)).
fn split_dibbur(query: &str) -> (String, Option<Vec<String>>) {
    let raw: Vec<&str> = query.split_whitespace().collect();
    match raw
        .iter()
        .position(|t| matches!(normalize_ref(t).as_str(), "דה" | "בדה"))
    {
        Some(i) => {
            let words = normalize_words(&raw[i + 1..].join(" "));
            let dibbur = (!words.is_empty()).then(|| words.into_iter().take(4).collect());
            (raw[..i].join(" "), dibbur)
        }
        None => (query.to_string(), None),
    }
}

/// The raw tokens after the first `consumed` normalized tokens, when that boundary falls
/// between raw tokens.
fn raw_rest<'a>(raw: &[&'a str], consumed: usize) -> Option<Vec<&'a str>> {
    let mut n = 0;
    for (i, t) in raw.iter().enumerate() {
        if n == consumed {
            return Some(raw[i..].to_vec());
        }
        n += normalize_ref(t).split_whitespace().count();
        if n > consumed {
            return None;
        }
    }
    (n == consumed).then(Vec::new)
}

/// Rewrite an address the way a heRef spells it: unit markers dropped, marker-prefixed
/// numbers (`פ"ב`, `ה"א`, `מ"ג`, `סק"ב`, `פכ"ה`) reduced to the number, `ע"א`/`ע"ב`
/// after a number turned into an amud.
fn clean_address(raw: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (i, tok) in raw.iter().enumerate() {
        let bare = normalize_ref(tok);
        if bare.is_empty() || MARKER_WORDS.contains(&bare.as_str()) {
            continue;
        }
        if (bare == "עא" || bare == "עב")
            && i + 1 == raw.len()
            && out
                .last()
                .is_some_and(|p| is_hebrew_numeral(&normalize_ref(p)))
        {
            let last = out.last_mut().unwrap();
            last.push(if bare == "עא" { '.' } else { ':' });
            continue;
        }
        out.push(strip_marker_prefix(tok).unwrap_or_else(|| tok.to_string()));
    }
    out
}

/// `פ"ב` → `ב`, `ה"א` → `א`, `מ"ג` → `ג`, `סק"ב` → `ב`, `פ"ק` → `א`, `פכ"ה` → `כ"ה`,
/// `הי"ב` → `י"ב`; `None` when the token has no such prefix.
fn strip_marker_prefix(tok: &str) -> Option<String> {
    let q = tok.find(['"', '\u{05F4}', '\'', '\u{05F3}'])?;
    let head = &tok[..q];
    let quote_len = tok[q..].chars().next()?.len_utf8();
    let tail = &tok[q + quote_len..];
    if ["פ", "ה", "מ", "סק", "ס"].contains(&head) && !tail.is_empty() {
        let n = normalize_ref(tail);
        if n == "ק" {
            return Some("א".to_string());
        }
        return is_hebrew_numeral(&n).then(|| tail.to_string());
    }
    let mut chars = head.chars();
    let first = chars.next()?;
    if (first == 'פ' || first == 'ה') && chars.next().is_some() {
        let rest = &tok[first.len_utf8()..];
        return is_hebrew_numeral(&normalize_ref(rest)).then(|| rest.to_string());
    }
    None
}

/// Drop the leading tokens of a normalized string while `drop` holds and the token is not
/// a number.
fn drop_leading(s: &str, drop: impl Fn(&str) -> bool) -> String {
    let toks: Vec<&str> = s.split_whitespace().collect();
    let skip = toks
        .iter()
        .take_while(|t| drop(t) && !is_hebrew_numeral(t))
        .count();
    toks[skip..].join(" ")
}

/// A well-formed Hebrew numeral: any ת, then at most one of ק ר ש, one tens letter and one
/// units letter, with 15 and 16 written טו / טז (so `סי`, `יה`, `סק` are not numerals).
fn is_hebrew_numeral(s: &str) -> bool {
    let mut rest = s;
    while let Some(r) = rest.strip_prefix('ת') {
        rest = r;
    }
    for group in ["קרש", "יכךלמםנןסעפףצץ", "אבגדהוזחט"] {
        if let Some(c) = rest.chars().next() {
            if group.contains(c) {
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    if rest == "ו" || rest == "ז" {
        // טו / טז
        return s.ends_with("טו") || s.ends_with("טז");
    }
    rest.is_empty() && !s.is_empty() && !s.ends_with("יה") && !s.ends_with("יו")
}

/// Words of running text for comparison: nikud, gershayim and punctuation removed.
fn normalize_words(s: &str) -> Vec<String> {
    let mut clean = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' | '\u{05F4}' | '\'' | '\u{05F3}' => {}
            '\u{0591}'..='\u{05BD}' | '\u{05BF}'..='\u{05C7}' => {}
            c if c.is_alphanumeric() => clean.push(c),
            _ => clean.push(' '),
        }
    }
    clean.split_whitespace().map(str::to_string).collect()
}

/// The line of `book_id` whose normalized heRef equals (exact) or extends (prefix)
/// `canonical`, preferring an exact match. With a dibbur hamatchil, the first of those
/// lines whose opening words contain it ("dibbur"); if none does, the location itself.
fn first_line_matching(
    conn: &Connection,
    line_text: &LineText,
    index: &RefIndex,
    book_id: i64,
    title: &str,
    canonical: &str,
    dibbur: Option<&[String]>,
) -> Result<Option<RefCandidate>> {
    let lines = index.matching_lines(conn, book_id, canonical)?;
    let Some(first) = lines
        .iter()
        .find(|l| normalize_ref(&l.he_ref) == canonical)
        .or_else(|| lines.first())
        .cloned()
    else {
        return Ok(None);
    };
    let content_of = |line_index: i64| -> Result<String> {
        Ok(conn.query_row(
            &format!(
                "SELECT {} FROM line l WHERE l.bookId = ?1 AND l.lineIndex = ?2",
                line_text.column("l")
            ),
            rusqlite::params![book_id, line_index],
            |r| line_text.get(r, 0),
        )?)
    };

    if let Some(dh) = dibbur {
        for line in lines.iter().take(400) {
            let content = content_of(line.line_index)?;
            let words = normalize_words(&crate::db::strip_html_tags(&content));
            let head = &words[..words.len().min(dh.len() + 8)];
            if head.windows(dh.len()).any(|w| w == dh) {
                return Ok(Some(make_candidate(
                    book_id,
                    title,
                    line.line_index,
                    line.he_ref.clone(),
                    content,
                    "dibbur",
                )));
            }
        }
    }

    let content = content_of(first.line_index)?;
    let match_kind = if normalize_ref(&first.he_ref) == canonical {
        "exact"
    } else {
        "prefix"
    };
    Ok(Some(make_candidate(
        book_id,
        title,
        first.line_index,
        first.he_ref,
        content,
        match_kind,
    )))
}

/// Last resort: substring match on raw `line.heRef`.
fn fallback_heref_like(
    conn: &Connection,
    line_text: &LineText,
    query: &str,
    limit: i64,
) -> Result<Vec<RefCandidate>> {
    let like = format!("%{}%", query.replace(['%', '_'], ""));
    let mut stmt = conn.prepare(&format!(
        "SELECT l.bookId, b.title, l.lineIndex, l.heRef, {} \
           FROM line l JOIN book b ON l.bookId = b.id \
          WHERE l.heRef LIKE ?1 \
          ORDER BY b.orderIndex, l.lineIndex LIMIT ?2",
        line_text.column("l")
    ))?;
    let rows = stmt.query_map(rusqlite::params![like, limit], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
            line_text.get(r, 4)?,
        ))
    })?;
    Ok(rows
        .flatten()
        .map(|(bid, title, li, href, content)| {
            make_candidate(bid, &title, li, href, content, "fallback")
        })
        .collect())
}

fn make_candidate(
    book_id: i64,
    title: &str,
    line_index: i64,
    he_ref: String,
    content: String,
    match_kind: &'static str,
) -> RefCandidate {
    let plain = crate::db::strip_html_tags(&content);
    let snippet: String = plain.chars().take(160).collect();
    RefCandidate {
        book_id,
        book_title: title.to_string(),
        line_index,
        he_ref,
        snippet,
        match_kind,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clean_address, is_hebrew_numeral, normalize_ref, starts_with_tokens, LineText, RefIndex,
    };
    use rusqlite::Connection;

    #[test]
    fn strips_nikud_and_gershayim() {
        assert_eq!(normalize_ref("בְּרֵאשִׁית"), "בראשית");
        assert_eq!(normalize_ref("שו\"ע או\"ח"), "שוע אוח");
        assert_eq!(normalize_ref("רד\"ק"), "רדק");
    }

    #[test]
    fn expands_talmud_amud_markers() {
        assert_eq!(normalize_ref("ברכות ב."), "ברכות ב א");
        assert_eq!(normalize_ref("ברכות ב:"), "ברכות ב ב");
    }

    #[test]
    fn maqaf_and_commas_become_spaces() {
        assert_eq!(normalize_ref("על־פני"), "על פני");
        assert_eq!(normalize_ref("בראשית א, א"), "בראשית א א");
    }

    #[test]
    fn token_prefix_respects_boundaries() {
        assert!(starts_with_tokens("בראשית א א", "בראשית א"));
        assert!(starts_with_tokens("בראשית", "בראשית"));
        assert!(!starts_with_tokens("בראשית אב", "בראשית א"));
    }

    #[test]
    fn resolves_from_the_per_book_normalized_cache() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE book (id INTEGER PRIMARY KEY, title TEXT NOT NULL, heRef TEXT, isBaseBook INTEGER NOT NULL, orderIndex INTEGER NOT NULL);
             CREATE TABLE book_acronym (bookId INTEGER NOT NULL, term TEXT NOT NULL);
             CREATE TABLE line (bookId INTEGER NOT NULL, lineIndex INTEGER NOT NULL, heRef TEXT, content TEXT NOT NULL);
             INSERT INTO book VALUES (1, 'בראשית', 'בראשית', 1, 1);
             INSERT INTO line VALUES (1, 2, 'בראשית א, א', 'בתחילה');",
        )
        .unwrap();
        let index = RefIndex::build(&conn);

        let first = super::resolve(&conn, &LineText::inline(), &index, "בראשית א א", 1).unwrap();
        let second = super::resolve(&conn, &LineText::inline(), &index, "בראשית א א", 1).unwrap();

        assert_eq!(first[0].line_index, 2);
        assert_eq!(second[0].line_index, 2);
        assert_eq!(index.line_cache.lock().unwrap().len(), 1);
    }

    #[test]
    fn hebrew_numerals() {
        for n in ["א", "טו", "טז", "קמא", "תקכג", "תת", "עב", "שח"] {
            assert!(is_hebrew_numeral(n), "{n}");
        }
        for w in ["החושן", "סק", "סי", "הל", "יה", "אב", ""] {
            assert!(!is_hebrew_numeral(w), "{w}");
        }
    }

    #[test]
    fn cleans_marker_words_and_prefixes() {
        let c = |q: &str| clean_address(&q.split_whitespace().collect::<Vec<_>>()).join(" ");
        assert_eq!(c("פ\"ב ה\"א"), "ב א");
        assert_eq!(c("פכ\"ה הי\"ב"), "כ\"ה י\"ב");
        assert_eq!(c("סי' שח ס\"ק ב"), "שח ב");
        assert_eq!(c("סימן רסה סק\"ג"), "רסה ג");
        assert_eq!(c("פ\"ק מ\"ג"), "א ג");
        assert_eq!(c("דף מג ע\"ב"), "מג:");
    }

    /// heRefs in the shapes the live library uses.
    fn library() -> (Connection, RefIndex) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE book (id INTEGER PRIMARY KEY, title TEXT NOT NULL, heRef TEXT, isBaseBook INTEGER NOT NULL, orderIndex INTEGER NOT NULL);
             CREATE TABLE book_acronym (bookId INTEGER NOT NULL, term TEXT NOT NULL);
             CREATE TABLE line (bookId INTEGER NOT NULL, lineIndex INTEGER NOT NULL, heRef TEXT, content TEXT NOT NULL);
             INSERT INTO book VALUES
               (103, 'שבת', 'שבת', 1, 1),
               (144, 'תלמוד ירושלמי שביעית', 'תלמוד ירושלמי שביעית', 1, 2),
               (1907, 'תוספות על שבת', 'תוספות על שבת', 0, 3),
               (3839, 'מגן אברהם', 'מגן אברהם', 0, 4),
               (3862, 'קצות החושן על שולחן ערוך חושן משפט', 'קצות החושן על שולחן ערוך חושן משפט', 0, 5);
             INSERT INTO book_acronym VALUES
               (144, 'ירושלמי שביעית'), (1907, 'תוס'' שבת'), (1907, 'תוספות שבת'),
               (3839, 'מג\"א'), (3862, 'קצות'), (3862, 'קצות החושן חו\"מ');
             INSERT INTO line VALUES
               (103, 500, 'שבת, מג:, א', 'שורה'),
               (103, 900, 'שבת, פב., א', 'שורה'),
               (144, 40, 'תלמוד ירושלמי שביעית, ב, א, א', 'משנה'),
               (1907, 540, 'תוספות על שבת, מג:, א, א', '<b>דחשיב עלייהו - </b>וא\"ת'),
               (1907, 541, 'תוספות על שבת, מג:, ב, א', '<b>בעודן עליו - </b>אף'),
               (3839, 3267, 'מגן אברהם,  שח, ב', '(ב) שמקפידין עליו'),
               (3862, 171, 'קצות החושן על שולחן ערוך חושן משפט, כח, א, א', 'וחייב');",
        )
        .unwrap();
        let index = RefIndex::build(&conn);
        (conn, index)
    }

    #[test]
    fn resolves_citation_forms_by_cleaned_address() {
        let (conn, index) = library();
        let lt = LineText::inline();
        let first = |q: &str| {
            let c = super::resolve(&conn, &lt, &index, q, 3).unwrap();
            assert!(!c.is_empty(), "no candidate for {q:?}");
            (c[0].book_id, c[0].line_index, c[0].match_kind)
        };
        assert_eq!(first("ירושלמי שביעית פ\"ב ה\"א").1, 40);
        assert_eq!(first("מגן אברהם אורח חיים שח ב").1, 3267);
        assert_eq!(first("מג\"א או\"ח סי' שח ס\"ק ב").1, 3267);
        assert_eq!(first("קצות החושן כח א").1, 171);
        assert_eq!(first("שבת מג ע\"ב").1, 500);
        assert_eq!(first("שבת דף מג ע\"ב").1, 500);
        // A written-out number still wins over a marker reading: daf 82, not perek 2.
        assert_eq!(first("שבת פ\"ב").1, 900);
    }

    #[test]
    fn dibbur_hamatchil_picks_its_line() {
        let (conn, index) = library();
        let lt = LineText::inline();
        let c = super::resolve(&conn, &lt, &index, "תוס' שבת מג: ד\"ה בעודן", 3).unwrap();
        assert_eq!((c[0].line_index, c[0].match_kind), (541, "dibbur"));
        // Unknown dibbur: the amud itself.
        let c = super::resolve(&conn, &lt, &index, "תוס' שבת מג: ד\"ה ופורסין", 3).unwrap();
        assert_eq!((c[0].line_index, c[0].match_kind), (540, "prefix"));
    }
}

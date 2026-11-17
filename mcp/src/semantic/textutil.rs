//! Text helpers of the semantic tools, ported from otzaria-sources' Python server
//! (`otzaria_semantic_server.py`, `personal_index.py`). Each keeps the Python function's
//! name and behaviour, characters counted as code points as Python does; the fixture test
//! (`semantic::parity_tests`) holds them to the Python output.

use std::sync::LazyLock;

use fancy_regex::Regex as FRegex;
use regex::Regex;

pub const MAX_CHARS: usize = 512;
pub const WINDOW_CHARS: usize = 512;
pub const WINDOW_OVERLAP: usize = 128;
pub const MAX_WINDOWS: usize = 6;
pub const MIN_EMBEDDABLE: usize = 5;
pub const MIN_MEANINGFUL: usize = 20;
/// A hit on a later window of a long line scores this much lower, as in Otzaria (the
/// default: a calibration file can set another, see [`super::calibration::window_penalty`]).
pub const WINDOW_PENALTY: f32 = 0.04;

static WS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
/// personal_index.NIKUD_RE: nikud and teamim, keeping maqaf (05BE), paseq (05C0),
/// sof pasuq (05C3) and nun hafukha (05C6).
static NIKUD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new("[\u{0591}-\u{05BD}\u{05BF}\u{05C1}\u{05C2}\u{05C4}\u{05C5}\u{05C7}]").unwrap()
});
/// The server's MARKS_RE: every Hebrew mark and point.
static MARKS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new("[\u{0591}-\u{05C7}]").unwrap());
static HEADING_RE: LazyLock<FRegex> =
    LazyLock::new(|| FRegex::new(r"(?is)^\s*<h([1-6])\b[^>]*>(.*?)</h\1>").unwrap());

/// `strip_html`: a line for display — tags removed as Otzaria's indexer removes them (`<br>`,
/// the block tags and `<sup>`/`<sub>` separate words, so a footnote marker is not glued to the
/// word before it), entities decoded, whitespace collapsed; nikud kept.
pub fn strip_html(s: &str) -> String {
    let no_tags = strip_tags(s);
    let text = htmlize::unescape(no_tags.as_str());
    WS_RE.replace_all(&text, " ").trim().to_string()
}

/// `personal_index.line_text`: a line as Otzaria's index stores it — the engine's own
/// `normalize_text_for_indexing` (line_text_version 4), the text every semantic key is cut from.
pub fn line_text(raw: &str) -> String {
    search_engine::api::search_engine::normalize_text_for_indexing(raw.to_string())
}

/// `personal_index.strip_data_uris`: the app's `stripDataUrisForIndex` — every `data:` followed
/// by 64 or more payload characters is removed with them before a book is indexed.
pub fn strip_data_uris(text: &str) -> std::borrow::Cow<'_, str> {
    const SCHEME: &str = "data:";
    let is_payload = |b: u8| b.is_ascii_alphanumeric() || b"+/;,=.-".contains(&b);
    let bytes = text.as_bytes();
    let Some(mut start) = text.find(SCHEME) else {
        return std::borrow::Cow::Borrowed(text);
    };
    let (mut out, mut copied) = (String::new(), 0);
    loop {
        let mut end = start + SCHEME.len();
        while end < bytes.len() && is_payload(bytes[end]) {
            end += 1;
        }
        if end - start - SCHEME.len() >= 64 {
            out.push_str(&text[copied..start]);
            copied = end;
        }
        match text[end..].find(SCHEME) {
            Some(next) => start = end + next,
            None => break,
        }
    }
    if copied == 0 {
        return std::borrow::Cow::Borrowed(text);
    }
    out.push_str(&text[copied..]);
    std::borrow::Cow::Owned(out)
}

/// Tags removed as `strip_html_for_indexing` removes them, entities left as they are.
fn strip_tags(text: &str) -> String {
    if !text.contains('<') {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let (mut out, mut copied, mut i) = (String::with_capacity(text.len()), 0, 0);
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        let Some(close) = bytes[i + 1..].iter().position(|&b| b == b'>') else {
            i += 1; // unterminated: kept
            continue;
        };
        out.push_str(&text[copied..i]);
        if tags::tag_breaks_word(text, i, i + 1 + close) {
            out.push(' ');
        }
        i += close + 2;
        copied = i;
    }
    out.push_str(&text[copied..]);
    out
}

/// Which tags separate words: verbatim from otzaria_search_engine's `rust/src/hebrew_query.rs`
/// (MIT, Copyright (c) 2026 otzaria), private there, so copied rather than called.
mod tags {
    const CGJ: char = '\u{034F}';

    fn is_attached_mark(c: char) -> bool {
        matches!(c, '\u{0591}'..='\u{05C7}')
            && !matches!(c, '\u{05BE}' | '\u{05C0}' | '\u{05C3}' | '\u{05C6}')
    }

    fn is_general_combining_mark(c: char) -> bool {
        matches!(c, '\u{0300}'..='\u{036F}')
    }

    fn is_word_mark(c: char) -> bool {
        is_attached_mark(c) || is_general_combining_mark(c) || c == '\u{FB1E}'
    }

    fn is_invisible_char(c: char) -> bool {
        matches!(
            c,
            '\u{200B}'..='\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{FEFF}'
        )
    }

    fn fold_presentation_form(c: char) -> Option<&'static str> {
        Some(match c {
            '\u{FB1D}' => "\u{05D9}\u{05B4}",
            '\u{FB1E}' => "", // varika — סימן צמוד ללא פירוק, מושמט
            '\u{FB1F}' => "\u{05F2}\u{05B7}",
            '\u{FB20}' => "\u{05E2}",
            '\u{FB21}' => "\u{05D0}",
            '\u{FB22}' => "\u{05D3}",
            '\u{FB23}' => "\u{05D4}",
            '\u{FB24}' => "\u{05DB}",
            '\u{FB25}' => "\u{05DC}",
            '\u{FB26}' => "\u{05DD}",
            '\u{FB27}' => "\u{05E8}",
            '\u{FB28}' => "\u{05EA}",
            '\u{FB2A}' => "\u{05E9}\u{05C1}",
            '\u{FB2B}' => "\u{05E9}\u{05C2}",
            '\u{FB2C}' => "\u{05E9}\u{05BC}\u{05C1}",
            '\u{FB2D}' => "\u{05E9}\u{05BC}\u{05C2}",
            '\u{FB2E}' => "\u{05D0}\u{05B7}",
            '\u{FB2F}' => "\u{05D0}\u{05B8}",
            '\u{FB30}' => "\u{05D0}\u{05BC}",
            '\u{FB31}' => "\u{05D1}\u{05BC}",
            '\u{FB32}' => "\u{05D2}\u{05BC}",
            '\u{FB33}' => "\u{05D3}\u{05BC}",
            '\u{FB34}' => "\u{05D4}\u{05BC}",
            '\u{FB35}' => "\u{05D5}\u{05BC}",
            '\u{FB36}' => "\u{05D6}\u{05BC}",
            '\u{FB38}' => "\u{05D8}\u{05BC}",
            '\u{FB39}' => "\u{05D9}\u{05BC}",
            '\u{FB3A}' => "\u{05DA}\u{05BC}",
            '\u{FB3B}' => "\u{05DB}\u{05BC}",
            '\u{FB3C}' => "\u{05DC}\u{05BC}",
            '\u{FB3E}' => "\u{05DE}\u{05BC}",
            '\u{FB40}' => "\u{05E0}\u{05BC}",
            '\u{FB41}' => "\u{05E1}\u{05BC}",
            '\u{FB43}' => "\u{05E3}\u{05BC}",
            '\u{FB44}' => "\u{05E4}\u{05BC}",
            '\u{FB46}' => "\u{05E6}\u{05BC}",
            '\u{FB47}' => "\u{05E7}\u{05BC}",
            '\u{FB48}' => "\u{05E8}\u{05BC}",
            '\u{FB49}' => "\u{05E9}\u{05BC}",
            '\u{FB4A}' => "\u{05EA}\u{05BC}",
            '\u{FB4B}' => "\u{05D5}\u{05B9}",
            '\u{FB4C}' => "\u{05D1}\u{05BF}",
            '\u{FB4D}' => "\u{05DB}\u{05BF}",
            '\u{FB4E}' => "\u{05E4}\u{05BF}",
            '\u{FB4F}' => "\u{05D0}\u{05DC}",
            _ => return None,
        })
    }

    const BREAKING_TAG_NAMES: &[&str] = &[
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ];

    const SCRIPT_TAG_NAMES: &[&str] = &["sub", "sup"];

    fn tag_name(body: &str) -> Option<(bool, &str)> {
        let (closing, name) = match body.strip_prefix('/') {
            Some(name) => (true, name),
            None => (false, body),
        };
        let len = name.bytes().take_while(u8::is_ascii_alphanumeric).count();
        (1..=10).contains(&len).then(|| (closing, &name[..len]))
    }

    fn is_listed(names: &[&str], name: &str) -> bool {
        names.iter().any(|listed| listed.eq_ignore_ascii_case(name))
    }

    fn is_hebrew_letter(c: char) -> bool {
        matches!(c, '\u{05D0}'..='\u{05EA}')
    }

    fn is_script_letter(c: char) -> bool {
        match fold_presentation_form(c) {
            Some(folded) => {
                let mut folded = folded.chars();
                folded.next().is_some_and(is_hebrew_letter) && folded.all(is_attached_mark)
            }
            None => is_hebrew_letter(c),
        }
    }

    pub(super) fn tag_breaks_word(source: &str, lt: usize, gt: usize) -> bool {
        let Some((closing, name)) = tag_name(&source[lt + 1..gt]) else {
            return false;
        };
        if is_listed(BREAKING_TAG_NAMES, name) {
            return true;
        }
        if !is_listed(SCRIPT_TAG_NAMES, name) {
            return false;
        }
        let open = if closing {
            script_open_before(source, lt)
        } else {
            Some((lt, gt))
        };
        !open.is_some_and(|(lt, gt)| is_word_internal_script(source, lt, gt))
    }

    fn script_open_before(source: &str, lt: usize) -> Option<(usize, usize)> {
        let mut before = trim_script_context_end(&source[..lt]).chars();
        before.next_back().filter(|&c| is_script_letter(c))?;
        let gt = trim_script_formatting_end(before.as_str())
            .len()
            .checked_sub(1)?;
        if source.as_bytes()[gt] != b'>' {
            return None;
        }
        let lt = source[..gt].rfind('<')?;
        (!source[lt + 1..gt].contains('>')).then_some((lt, gt))
    }

    fn is_inline_tag(body: &str) -> bool {
        !tag_name(body).is_some_and(|(_, name)| {
            is_listed(BREAKING_TAG_NAMES, name) || is_listed(SCRIPT_TAG_NAMES, name)
        })
    }

    fn is_script_invisible(c: char) -> bool {
        is_invisible_char(c) || c == CGJ || c == '\u{FB1E}'
    }

    fn trim_script_formatting_start(mut source: &str) -> &str {
        loop {
            source = source.trim_start_matches(is_script_invisible);
            let Some(body) = source.strip_prefix('<') else {
                break;
            };
            let Some(gt) = body.find('>') else { break };
            if !is_inline_tag(&body[..gt]) {
                break;
            }
            source = &body[gt + 1..];
        }
        source
    }

    fn trim_script_formatting_end(mut source: &str) -> &str {
        loop {
            source = source.trim_end_matches(is_script_invisible);
            let Some(body) = source.strip_suffix('>') else {
                break;
            };
            let Some(lt) = body.rfind('<') else { break };
            if body[lt + 1..].contains('>') || !is_inline_tag(&body[lt + 1..]) {
                break;
            }
            source = &body[..lt];
        }
        source
    }

    fn trim_script_context_end(mut source: &str) -> &str {
        loop {
            let trimmed = trim_script_formatting_end(source).trim_end_matches(is_word_mark);
            if trimmed.len() == source.len() {
                return source;
            }
            source = trimmed;
        }
    }

    fn trim_script_context_start(mut source: &str) -> &str {
        loop {
            let trimmed = trim_script_formatting_start(source).trim_start_matches(is_word_mark);
            if trimmed.len() == source.len() {
                return source;
            }
            source = trimmed;
        }
    }

    fn marks_footnote(attributes: &str) -> bool {
        let lower = attributes.to_ascii_lowercase();
        let word = |b: Option<&u8>| b.is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
        lower.match_indices("footnote-marker").any(|(i, found)| {
            !word(lower.as_bytes()[..i].last()) && !word(lower.as_bytes().get(i + found.len()))
        })
    }

    fn is_word_internal_script(source: &str, lt: usize, gt: usize) -> bool {
        let Some((false, name)) = tag_name(&source[lt + 1..gt]) else {
            return false;
        };
        if marks_footnote(&source[lt + 1..gt]) {
            return false;
        }
        if !trim_script_context_end(&source[..lt])
            .chars()
            .next_back()
            .is_some_and(is_script_letter)
        {
            return false;
        }
        let mut content = trim_script_formatting_start(&source[gt + 1..]).chars();
        if !content.next().is_some_and(is_script_letter) {
            return false;
        }
        let rest = trim_script_context_start(content.as_str());
        let Some(close) = rest.strip_prefix('<') else {
            return false;
        };
        let Some(close_gt) = close.find('>') else {
            return false;
        };
        matches!(tag_name(&close[..close_gt]), Some((true, closing)) if closing.eq_ignore_ascii_case(name))
            && trim_script_context_start(&close[close_gt + 1..])
                .chars()
                .next()
                .is_some_and(is_script_letter)
    }
}

pub fn strip_nikud(s: &str) -> String {
    NIKUD_RE.replace_all(s, "").into_owned()
}

pub fn strip_marks(s: &str) -> String {
    MARKS_RE.replace_all(s, "").into_owned()
}

/// Character spans `[a, b)` (code points) of a line's windows, Otzaria's chunker.rs
/// `window_spans`: the first 512 characters, then 512-character windows overlapping the
/// one before by 128, the start moved to the next word and the end back to a space; at
/// most [`MAX_WINDOWS`].
pub fn window_spans(text: &str) -> Vec<(usize, usize)> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut spans = vec![(0, n.min(MAX_CHARS))];
    while spans.last().unwrap().1 < n && spans.len() < MAX_WINDOWS {
        let (ps, pe) = *spans.last().unwrap();
        let mut a = pe.saturating_sub(WINDOW_OVERLAP).max(ps + 1);
        if c[a - 1] != ' ' {
            // text.find(" ", a, pe)
            if let Some(sp) = (a..pe).find(|&i| c[i] == ' ') {
                a = sp + 1;
            }
        }
        let mut b = n.min(a + WINDOW_CHARS);
        if b < n {
            // text.rfind(" ", a + WINDOW_CHARS // 2, b + 1)
            let lo = a + WINDOW_CHARS / 2;
            let hi = (b + 1).min(n);
            if lo < hi {
                if let Some(sp) = (lo..hi).rev().find(|&i| c[i] == ' ') {
                    b = sp;
                }
            }
        }
        spans.push((a, b));
    }
    spans
}

pub fn window_span(text: &str, ordinal: usize) -> (usize, usize) {
    let sp = window_spans(text);
    sp[ordinal.min(sp.len() - 1)]
}

/// Characters `[a, b)` of `s`, counted as code points.
pub fn char_slice(s: &str, a: usize, b: usize) -> String {
    s.chars().skip(a).take(b.saturating_sub(a)).collect()
}

pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// The first `n` characters (code points) of `s`.
pub fn char_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `personal_index.heading_path`: headings in force at line `upto` (h1..h6), outermost first.
pub fn heading_path(raw_lines: &[String], upto: usize) -> Vec<String> {
    let mut stack: Vec<(u8, String)> = Vec::new();
    for x in raw_lines.iter().take(upto + 1) {
        if let Ok(Some(m)) = HEADING_RE.captures(x) {
            let level: u8 = m[1].parse().unwrap_or(1);
            stack.retain(|(k, _)| *k < level);
            stack.push((level, line_text(&m[2])));
        }
    }
    stack.sort_by_key(|(k, _)| *k);
    stack.into_iter().map(|(_, v)| v).collect()
}

pub fn is_heading(raw: &str) -> bool {
    matches!(HEADING_RE.is_match(raw), Ok(true))
}

/// `norm_word`: the comparison form of a word — no nikud or teamim, no punctuation, no
/// geresh/gershayim (Python's `[^\wא-ת]` removed, then `_`).
pub fn norm_word(w: &str) -> String {
    w.chars()
        .filter(|&ch| !('\u{0591}'..='\u{05C7}').contains(&ch))
        .filter(|&ch| ch.is_alphanumeric() || ('\u{05D0}'..='\u{05EA}').contains(&ch))
        .collect()
}

pub fn otzaria_url(book_id: i64, line_index: i64) -> String {
    format!("otzaria://open/book/{book_id}?index={line_index}")
}

/// Where a book comes from: `o` the library, `u` personal books, `d:<slug>` an attached
/// database. The same keys the Python server used.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub enum SourceKey {
    Official,
    User,
    Attached(String),
}

impl SourceKey {
    pub fn code(&self) -> String {
        match self {
            SourceKey::Official => "o".into(),
            SourceKey::User => "u".into(),
            SourceKey::Attached(s) => format!("d:{s}"),
        }
    }

    pub fn parse(s: &str) -> Option<SourceKey> {
        let t = s.trim();
        match t {
            "" | "o" => Some(SourceKey::Official),
            "u" => Some(SourceKey::User),
            _ => t
                .strip_prefix("d:")
                .map(|slug| SourceKey::Attached(slug.to_string())),
        }
    }

    /// The link a hit of this source opens in Otzaria.
    pub fn url(&self, book_id: i64, line_index: i64) -> String {
        match self {
            SourceKey::Official => otzaria_url(book_id, line_index),
            SourceKey::User => {
                format!("otzaria://open/book/{book_id}?source=user&index={line_index}")
            }
            SourceKey::Attached(slug) => {
                format!("otzaria://open/book/{book_id}?source=db:{slug}&index={line_index}")
            }
        }
    }
}

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"otzaria://open/book/(\d+)\?([^\s#)\]>"']*)"#).unwrap());

/// `parse_book_url`: `otzaria://open/book/<id>?[source=user|db:<slug>&]index=<n>` ->
/// (source, book id, line index).
pub fn parse_book_url(url: &str) -> Option<(SourceKey, i64, i64)> {
    let m = URL_RE.captures(url)?;
    let book: i64 = m[1].parse().ok()?;
    let mut index = None;
    let mut source = None;
    for pair in m[2].split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let v = percent_decode(v);
        if v.is_empty() {
            continue; // parse_qs drops blank values
        }
        match percent_decode(k).as_str() {
            "index" if index.is_none() => index = Some(v),
            "source" if source.is_none() => source = Some(v),
            _ => {}
        }
    }
    let li: i64 = match index {
        Some(v) => v.trim().parse().ok()?,
        None => 0,
    };
    let src = source.unwrap_or_default();
    let src = src.trim();
    let key = if src.is_empty() || src.eq_ignore_ascii_case("official") {
        SourceKey::Official
    } else if src.eq_ignore_ascii_case("user") {
        SourceKey::User
    } else if src.len() >= 3 && src[..3].eq_ignore_ascii_case("db:") {
        SourceKey::Attached(src[3..].to_string())
    } else {
        return None;
    };
    Some((key, book, li))
}

/// `urllib.parse.unquote_plus`.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok();
            if let Some(v) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `trunc16`: truncate to `n` UTF-16 code units without splitting a surrogate pair.
pub fn trunc16(s: &str, n: usize) -> String {
    let mut units = 0;
    let mut out = String::new();
    for ch in s.chars() {
        let w = ch.len_utf16();
        if units + w > n {
            break;
        }
        units += w;
        out.push(ch);
    }
    out
}

// ── the repetition ("echo") test ────────────────────────────────────────────

static ECHO_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[\u{05D0}-\u{05EA}A-Za-z0-9]+").unwrap());

fn is_heb_letter(c: char) -> bool {
    ('\u{05D0}'..='\u{05EA}').contains(&c)
}

/// `echo_words`: words for the repetition test — no nikud or teamim; maqaf, paseq, sof
/// pasuq and every other mark separate words; gershayim/geresh inside a word are dropped
/// (ר"ן = רן).
pub fn echo_words(text: &str) -> Vec<String> {
    // ECHO_DROP_RE: teamim 0591-05AF, nikud 05B0-05BD, 05BF, 05C1, 05C2, 05C4, 05C5, 05C7
    let dropped: Vec<char> = text
        .chars()
        .filter(|&c| {
            !(('\u{0591}'..='\u{05BD}').contains(&c)
                || matches!(
                    c,
                    '\u{05BF}' | '\u{05C1}' | '\u{05C2}' | '\u{05C4}' | '\u{05C5}' | '\u{05C7}'
                ))
        })
        .collect();
    // ECHO_JOIN_RE: a quote mark between two Hebrew letters
    let mut joined = String::with_capacity(dropped.len());
    for (i, &c) in dropped.iter().enumerate() {
        let quote = matches!(
            c,
            '"' | '\'' | '\u{05F3}' | '\u{05F4}' | '\u{2019}' | '\u{201D}'
        );
        if quote
            && i > 0
            && i + 1 < dropped.len()
            && is_heb_letter(dropped[i - 1])
            && is_heb_letter(dropped[i + 1])
        {
            continue;
        }
        joined.push(c);
    }
    ECHO_WORD_RE
        .find_iter(&joined)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct EchoStats {
    pub containment: f64,
    pub run: usize,
    pub words: usize,
    pub n: usize,
}

/// `EchoReference`: lexical overlap of a hit with the note. Containment is the share of the
/// hit's word n-grams that occur in the note; run the longest stretch of hit words found
/// contiguously in the note.
pub struct EchoReference {
    n: usize,
    grams2: std::collections::HashSet<Vec<String>>,
    gramsn: std::collections::HashSet<Vec<String>>,
    empty: bool,
}

impl EchoReference {
    pub fn new(text: &str, n: usize) -> EchoReference {
        let n = n.clamp(2, 8);
        let words = echo_words(text);
        let grams = |k: usize| -> std::collections::HashSet<Vec<String>> {
            if words.len() < k {
                return Default::default();
            }
            (0..=words.len() - k)
                .map(|i| words[i..i + k].to_vec())
                .collect()
        };
        EchoReference {
            n,
            grams2: grams(2),
            gramsn: grams(n),
            empty: words.is_empty(),
        }
    }

    pub fn measure(&self, text: &str) -> Option<EchoStats> {
        if self.empty {
            return None;
        }
        let words = echo_words(text);
        if words.len() < 2 {
            return None;
        }
        let k = if words.len() >= self.n { self.n } else { 2 };
        let set = if k == self.n {
            &self.gramsn
        } else {
            &self.grams2
        };
        let hits: Vec<bool> = (0..=words.len() - k)
            .map(|i| set.contains(&words[i..i + k].to_vec()))
            .collect();
        let (mut best, mut cur) = (0usize, 0usize);
        for &h in &hits {
            cur = if h { cur + 1 } else { 0 };
            best = best.max(cur);
        }
        let containment = hits.iter().filter(|&&h| h).count() as f64 / hits.len() as f64;
        Some(EchoStats {
            containment: round_to(containment, 3),
            run: if best > 0 { best + k - 1 } else { 0 },
            words: words.len(),
            n: k,
        })
    }
}

/// Python's `round(x, nd)`: to the nearest, ties to even, on the decimal value.
pub fn round_to(x: f64, nd: i32) -> f64 {
    let p = 10f64.powi(nd);
    let y = x * p;
    let r = y.round();
    let r = if (y - y.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        r - y.signum()
    } else {
        r
    };
    r / p
}

// ── a note cut into query windows ───────────────────────────────────────────

static FM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\A---\s*\n.*?\n---\s*(\n|\z)").unwrap());
static CODE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").unwrap());
static CODE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`[^`\n]*`").unwrap());
static COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)%%.*?%%").unwrap());
static HTML_COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());
static EMBED_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"!\[\[[^\]]*\]\]").unwrap());
static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]|#^]*)(?:[#^][^\]|]*)?(?:\|([^\]]*))?\]\]").unwrap());
static MDLINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!?\[([^\]]*)\]\([^)]*\)").unwrap());
static URLTEXT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+|otzaria://\S+").unwrap());
static FOOTNOTE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\^[^\]]*\]").unwrap());
static TAGWORD_RE: LazyLock<FRegex> =
    LazyLock::new(|| FRegex::new(r"(?<![\w&/])#[\w\-/\u{0590}-\u{05FF}]+").unwrap());
static CALLOUT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*>\s*\[![^\]]*\][+-]?").unwrap());
static BLOCK_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+(\[.\]\s+)?|\d+[.)]\s+)").unwrap()
});
static EMPHASIS_RE: LazyLock<FRegex> =
    LazyLock::new(|| FRegex::new(r"(\*\*|__|==|~~|\*|_)(?=\S)(.+?)(?<=\S)\1").unwrap());
static INLINE_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>\n]+>").unwrap());

fn freplace(re: &FRegex, text: &str, rep: &str) -> String {
    re.replace_all(text, rep).into_owned()
}

/// `clean_markdown`: a note's text without frontmatter, code, comments, link syntax, tags
/// and block markers.
pub fn clean_markdown(text: &str) -> String {
    let t = FM_RE.replace(text, "").into_owned();
    let t = CODE_BLOCK_RE.replace_all(&t, " ").into_owned();
    let t = CODE_RE.replace_all(&t, " ").into_owned();
    let t = COMMENT_RE.replace_all(&t, " ").into_owned();
    let t = HTML_COMMENT_RE.replace_all(&t, " ").into_owned();
    let t = EMBED_RE.replace_all(&t, " ").into_owned();
    let t = WIKILINK_RE
        .replace_all(&t, |c: &regex::Captures| {
            let alias = c.get(2).map(|m| m.as_str()).unwrap_or("");
            if !alias.is_empty() {
                alias.to_string()
            } else {
                c.get(1)
                    .map(|m| m.as_str())
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .to_string()
            }
        })
        .into_owned();
    let t = MDLINK_RE.replace_all(&t, "$1").into_owned();
    let t = URLTEXT_RE.replace_all(&t, " ").into_owned();
    let t = FOOTNOTE_RE.replace_all(&t, "").into_owned();
    let t = freplace(&TAGWORD_RE, &t, " ");
    let t = CALLOUT_RE.replace_all(&t, "").into_owned();
    let t = BLOCK_PREFIX_RE.replace_all(&t, "").into_owned();
    let t = freplace(&EMPHASIS_RE, &t, "$2");
    INLINE_TAG_RE.replace_all(&t, " ").into_owned()
}

pub fn hebrew_enough(s: &str, min_heb: usize) -> bool {
    let he = s.chars().filter(|&c| is_heb_letter(c)).count();
    let la = s.chars().filter(|c| c.is_ascii_alphabetic()).count();
    he >= min_heb && (he as f64) >= 0.4 * (he + la) as f64
}

/// `re.split(r"\n\s*\n|\n(?=\S)", body)`.
fn split_paragraphs(body: &str) -> Vec<String> {
    let c: Vec<char> = body.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '\n' {
            // \n\s*\n : the longest whitespace run that ends in \n
            let mut j = i + 1;
            let mut last_nl = None;
            while j < c.len() && c[j].is_whitespace() {
                if c[j] == '\n' {
                    last_nl = Some(j);
                }
                j += 1;
            }
            if let Some(end) = last_nl {
                out.push(std::mem::take(&mut cur));
                i = end + 1;
                continue;
            }
            // \n(?=\S)
            if i + 1 < c.len() && !c[i + 1].is_whitespace() {
                out.push(std::mem::take(&mut cur));
                i += 1;
                continue;
            }
        }
        cur.push(c[i]);
        i += 1;
    }
    out.push(cur);
    out
}

/// `SENT_SPLIT.split(p)`: `(?<=[.:;?!׃])\s+`.
fn split_sentences(p: &str) -> Vec<String> {
    let c: Vec<char> = p.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_whitespace()
            && i > 0
            && matches!(c[i - 1], '.' | ':' | ';' | '?' | '!' | '\u{05C3}')
        {
            let mut j = i;
            while j < c.len() && c[j].is_whitespace() {
                j += 1;
            }
            out.push(std::mem::take(&mut cur));
            i = j;
            continue;
        }
        cur.push(c[i]);
        i += 1;
    }
    out.push(cur);
    out
}

/// numpy's `round()` on a float: to the nearest, ties to even.
fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

/// `note_windows`: a note cut into the query windows `related` embeds. `count_tokens` is the
/// model tokenizer's count without special tokens.
pub fn note_windows(
    count_tokens: &dyn Fn(&str) -> usize,
    text: &str,
    title: &str,
    window_tokens: usize,
    max_windows: usize,
    min_heb: usize,
) -> Vec<String> {
    let body = clean_markdown(text);
    let paras: Vec<String> = split_paragraphs(&body)
        .into_iter()
        .map(|p| WS_RE.replace_all(&p, " ").trim().to_string())
        .filter(|p| !p.is_empty() && hebrew_enough(p, min_heb))
        .collect();
    let mut pieces: Vec<(String, usize)> = Vec::new();
    for p in paras {
        let n = count_tokens(&p);
        if n <= window_tokens {
            pieces.push((p, n));
            continue;
        }
        let (mut cur, mut cur_n): (Vec<String>, usize) = (Vec::new(), 0);
        for s in split_sentences(&p) {
            let sn = count_tokens(&s);
            if !cur.is_empty() && cur_n + sn > window_tokens {
                pieces.push((cur.join(" "), cur_n));
                cur.clear();
                cur_n = 0;
            }
            cur.push(s);
            cur_n += sn;
        }
        if !cur.is_empty() {
            pieces.push((cur.join(" "), cur_n));
        }
    }
    let mut merged: Vec<(String, usize)> = Vec::new();
    for (p, n) in pieces {
        if let Some(last) = merged.last_mut() {
            if (last.1 < 24 || n < 24) && last.1 + n <= window_tokens {
                last.0 = format!("{} {}", last.0, p);
                last.1 += n;
                continue;
            }
        }
        merged.push((p, n));
    }
    let mut windows: Vec<String> = merged.into_iter().map(|(p, _)| p).collect();
    if !title.is_empty() && hebrew_enough(title, 3) {
        windows.insert(0, title.trim().to_string());
    }
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = windows
        .into_iter()
        .filter(|w| seen.insert(w.clone()))
        .collect();
    if out.len() > max_windows {
        let n = out.len();
        let mut keep: Vec<usize> = if max_windows <= 1 {
            vec![0]
        } else {
            (0..max_windows)
                .map(|i| {
                    round_half_even(i as f64 * (n - 1) as f64 / (max_windows - 1) as f64) as usize
                })
                .collect()
        };
        keep.sort_unstable();
        keep.dedup();
        out = keep.into_iter().map(|i| out[i].clone()).collect();
    }
    out
}

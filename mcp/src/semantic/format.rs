//! The text the semantic MCP tools return, as otzaria_mcp.py wrote it.

use serde_json::Value;

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

/// `clip`: whitespace collapsed, at most `n` characters with an ellipsis.
pub fn clip(text: &str, n: usize) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() <= n {
        t
    } else {
        let mut out: String = t.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn last2(v: &Value, k: &str) -> String {
    let parts: Vec<&str> = v
        .get(k)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    parts[parts.len().saturating_sub(2)..].join(" › ")
}

fn context(h: &Value, side: &str, chars: usize) -> Vec<String> {
    let mark = if side == "before" { "↑" } else { "↓" };
    h.get(format!("context_{side}"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|c| format!("   {mark} {}", clip(s(c, "text"), chars)))
                .collect()
        })
        .unwrap_or_default()
}

fn text_or_snippet(h: &Value) -> &str {
    let t = s(h, "text");
    if t.is_empty() {
        s(h, "snippet")
    } else {
        t
    }
}

fn score(h: &Value) -> f64 {
    h.get("score").and_then(Value::as_f64).unwrap_or(0.0)
}

pub fn hits(hits: &[Value], snippet_chars: usize) -> String {
    let mut lines = Vec::new();
    for (n, h) in hits.iter().enumerate() {
        let flag = if h.get("cited").and_then(Value::as_bool).unwrap_or(false) {
            "  [already cited]"
        } else {
            ""
        };
        lines.push(format!(
            "{}. [{:.2}] {}  ({}){}\n   {}",
            n + 1,
            score(h),
            s(h, "he_ref"),
            last2(h, "category"),
            flag,
            s(h, "url")
        ));
        lines.extend(context(h, "before", snippet_chars));
        lines.push(format!("   {}", clip(text_or_snippet(h), snippet_chars)));
        lines.extend(context(h, "after", snippet_chars));
        if let Some(also) = h
            .get("also_in")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
        {
            let items: Vec<String> = also
                .iter()
                .take(4)
                .map(|a| format!("{} <{}>", s(a, "book_title"), s(a, "url")))
                .collect();
            lines.push(format!("   also in: {}", items.join(", ")));
        }
    }
    lines.join("\n")
}

pub fn search(query: &str, out: &Value, snippet_chars: usize) -> String {
    let hs = out
        .get("hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if hs.is_empty() {
        return "No passages above the similarity threshold.".into();
    }
    format!(
        "Semantic matches for: {}\n\n{}",
        clip(query, 120),
        hits(&hs, snippet_chars)
    )
}

pub fn get_text(out: &Value) -> String {
    let mut head = format!("{}  ({})", s(out, "book_title"), last2(out, "category"));
    if let Some(toc) = out
        .get("toc")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        head.push_str(&format!(
            "\nlocation: {}",
            toc.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" › ")
        ));
    }
    if !s(out, "note").is_empty() {
        head.push_str(&format!("\nnote: {}", s(out, "note")));
    }
    let pattern = match s(out, "url_pattern") {
        "" => format!(
            "otzaria://open/book/{}?index=<line>",
            out.get("book_id").cloned().unwrap_or_default()
        ),
        p => p.to_string(),
    };
    head.push_str(&format!("\nlink pattern: {pattern}"));
    let center = out.get("center").and_then(Value::as_i64);
    let rows: Vec<String> = out
        .get("lines")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|ln| {
                    let li = ln.get("line_index").and_then(Value::as_i64);
                    let mark = if li == center { ">>" } else { "  " };
                    let r = s(ln, "he_ref");
                    let r = if r.is_empty() {
                        String::new()
                    } else {
                        format!(" ({r})")
                    };
                    let h = if ln.get("heading").and_then(Value::as_bool).unwrap_or(false) {
                        "## "
                    } else {
                        ""
                    };
                    format!("{mark}[{}]{r} {h}{}", li.unwrap_or(0), s(ln, "text"))
                })
                .collect()
        })
        .unwrap_or_default();
    format!("{head}\n\n{}", rows.join("\n"))
}

fn where_(x: &Value) -> String {
    [s(x, "era").to_string(), last2(x, "category")]
        .into_iter()
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

fn pct(x: &Value) -> String {
    format!(
        "{:.0}%",
        x.get("ratio").and_then(Value::as_f64).unwrap_or(0.0) * 100.0
    )
}

fn diff_lines(hit: &Value) -> Vec<String> {
    let diffs = hit
        .get("diffs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (abbr, real): (Vec<&Value>, Vec<&Value>) = diffs.iter().partition(|d| s(d, "op") == "abbr");
    let mut out = Vec::new();
    if !abbr.is_empty() {
        let items: Vec<String> = abbr
            .iter()
            .map(|d| format!("{} = {}", s(d, "quote"), s(d, "library")))
            .collect();
        out.push(format!(
            "Abbreviations resolved (not variants): {}",
            items.join("; ")
        ));
    }
    if !real.is_empty() {
        out.push("Differences (quote → library):".into());
        for d in real.iter().take(30) {
            out.push(match s(d, "op") {
                "replace" => format!("  - {} → {}", s(d, "quote"), s(d, "library")),
                "delete" => format!("  - {} → (absent in library)", s(d, "quote")),
                _ => format!("  - (absent in quote) → {}", s(d, "library")),
            });
        }
    } else {
        out.push("No differences beyond nikud/punctuation.".into());
    }
    out
}

pub fn check_quote(quote: &str, out: &Value) -> String {
    let b = match out
        .get("source")
        .filter(|v| !v.is_null())
        .or_else(|| out.get("best").filter(|v| !v.is_null()))
    {
        Some(b) => b,
        None => {
            let wh = if s(out, "located_by") == "reference" {
                " at the given location"
            } else {
                ""
            };
            let note = if s(out, "note").is_empty() {
                String::new()
            } else {
                format!("\nnote: {}", s(out, "note"))
            };
            return format!(
                "Verdict: not found{wh}. No library line resembles this quotation.\nQuote: {}{note}",
                clip(quote, 200)
            );
        }
    };
    let located = s(out, "located_by") == "reference";
    let mut lines = vec![
        format!("Verdict: {}", s(out, "verdict")),
        format!(
            "{}: {}  ({})  word match {}",
            if located {
                "Best match"
            } else {
                "Earliest source"
            },
            s(b, "he_ref"),
            where_(b),
            pct(b)
        ),
        format!("Link: {}", s(b, "url")),
        format!("Library text: {}", s(b, "library_text")),
    ];
    if !s(out, "note").is_empty() {
        lines.push(format!("note: {}", s(out, "note")));
    }
    lines.extend(diff_lines(b));
    if let Some(c) = out.get("closest").filter(|v| !v.is_null()) {
        lines.push(String::new());
        lines.push(format!(
            "Closest wording (a later work matches better): {}  ({})  word match {}",
            s(c, "he_ref"),
            where_(c),
            pct(c)
        ));
        lines.push(format!("Link: {}", s(c, "url")));
        lines.push(format!("Library text: {}", s(c, "library_text")));
        lines.extend(diff_lines(c));
    }
    let alts: Vec<&Value> = out
        .get("alternatives")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|x| x.get("ratio").and_then(Value::as_f64).unwrap_or(0.0) >= 0.5)
                .collect()
        })
        .unwrap_or_default();
    if !alts.is_empty() {
        lines.push("Other places with this text (earliest first within equal match):".into());
        for x in alts {
            lines.push(format!(
                "  - {} ({}; {}) {}",
                s(x, "he_ref"),
                where_(x),
                pct(x),
                s(x, "url")
            ));
        }
    }
    lines.join("\n")
}

pub fn parallels(out: &Value, snippet_chars: usize) -> String {
    let hs = out
        .get("hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if hs.is_empty() {
        return match s(out, "note") {
            "" => "No candidate parallels above the threshold.".into(),
            n => n.to_string(),
        };
    }
    let wins: Vec<&str> = out
        .get("windows")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut lines = vec![
        "CANDIDATE parallels (unverified; for the report only, not for note bodies or links)."
            .to_string(),
        String::new(),
    ];
    for (n, h) in hs.iter().enumerate() {
        let via = h
            .get("windows")
            .and_then(Value::as_array)
            .and_then(|w| w.first())
            .and_then(Value::as_u64)
            .and_then(|i| wins.get(i as usize).copied())
            .unwrap_or("");
        let flag = if h.get("cited").and_then(Value::as_bool).unwrap_or(false) {
            "  [already cited]"
        } else {
            ""
        };
        lines.push(format!(
            "{}. [{:.2}] {}  ({}){}\n   {}",
            n + 1,
            score(h),
            s(h, "he_ref"),
            last2(h, "category"),
            flag,
            s(h, "url")
        ));
        lines.extend(context(h, "before", snippet_chars));
        lines.push(format!("   {}", clip(text_or_snippet(h), snippet_chars)));
        lines.extend(context(h, "after", snippet_chars));
        if !via.is_empty() {
            lines.push(format!("   matched via: {}", clip(via, 90)));
        }
    }
    lines.join("\n")
}

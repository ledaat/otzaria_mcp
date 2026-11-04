//! Find-ref across the library, personal books and attached databases — the one
//! implementation behind the `resolve_ref` / `resolve_refs` tools and the HTTP daemon's
//! `POST /resolve_refs`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Value};

use crate::db::Db;
use crate::refs::{rank_candidates, RefCandidate, ResolveOptions};
use crate::secondary::{SecondaryConfig, SecondarySet, Source};

/// Generation of resolve_ref's answers. Bumped only when the same reference can come back
/// with different candidates (not with a new package version), so a client caching
/// answers can key its cache on it. "0.3": the app-parity find-ref of 0.3.0, unchanged
/// through 0.4.x (0.4.2 adds `book_level`, set only when true, and opt-in filters).
pub const RESOLVER: &str = "0.3";

/// Most references one `resolve_refs` call takes.
pub const MAX_BATCH: usize = 2000;

/// What a find-ref call asks for besides the reference itself.
#[derive(Debug, Clone)]
pub struct FindRefRequest {
    pub limit: i64,
    /// `sources` as passed by the client ("library", "user", "attached[:slug]", "all").
    pub filters: Vec<String>,
    /// For the library; secondary databases never take the heRef fallback.
    pub opts: ResolveOptions,
}

impl FindRefRequest {
    pub fn new(
        limit: Option<u32>,
        sources: Option<Vec<String>>,
        exclude_book_level: Option<bool>,
        heref_fallback: Option<bool>,
        snippets: Option<bool>,
    ) -> FindRefRequest {
        let d = ResolveOptions::default();
        FindRefRequest {
            limit: limit.unwrap_or(10).clamp(1, 50) as i64,
            filters: sources
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| vec!["library".to_string()]),
            opts: ResolveOptions {
                heref_fallback: heref_fallback.unwrap_or(d.heref_fallback),
                app_detection: true,
                exclude_book_level: exclude_book_level.unwrap_or(false),
                snippets: snippets.unwrap_or(true),
            },
        }
    }

    fn wants(&self, src: &Source) -> bool {
        self.filters.iter().any(|f| f.trim() == "all")
            || self.filters.iter().any(|f| src.matches_filter(f))
    }
}

/// One reference, every source asked for, one ranking (see `refs::rank_candidates`).
pub fn resolve(
    db: &Db,
    secondary: &SecondarySet,
    reference: &str,
    req: &FindRefRequest,
) -> Result<Vec<RefCandidate>> {
    let mut out: Vec<RefCandidate> = Vec::new();
    if req.wants(&Source::Library) {
        out.extend(db.resolve_ref_with(reference, req.limit, req.opts)?);
    }
    let sec_opts = ResolveOptions {
        heref_fallback: false,
        ..req.opts
    };
    for sec in &secondary.dbs {
        if !req.wants(&sec.spec.source) {
            continue;
        }
        let Some(sdb) = sec.db() else { continue };
        // one broken database must not sink the lookup
        match sdb.resolve_ref_with(reference, req.limit, sec_opts) {
            Ok(cands) => out.extend(cands.into_iter().map(|mut c| {
                c.set_source(&sec.spec.source);
                c
            })),
            Err(e) => eprintln!(
                "otzaria-mcp: resolve_ref in {} failed: {e:#}",
                sec.spec.path.display()
            ),
        }
    }
    let mut out = rank_candidates(reference, out);
    out.truncate(req.limit as usize);
    Ok(out)
}

/// Many references in one go: one entry per reference, in order —
/// `{reference, count, candidates, ms}`, or `{reference, error}` for one that failed
/// (it does not sink the others). Repeats are answered once.
pub fn resolve_batch(
    db: &Db,
    secondary: &SecondarySet,
    references: &[String],
    req: &FindRefRequest,
) -> Vec<Value> {
    let mut seen: HashMap<&str, Value> = HashMap::new();
    let mut out = Vec::with_capacity(references.len());
    for r in references {
        let reference = r.trim();
        if let Some(v) = seen.get(reference) {
            out.push(v.clone());
            continue;
        }
        let v = if reference.is_empty() {
            json!({ "reference": r, "count": 0, "candidates": [] })
        } else {
            let t = Instant::now();
            match resolve(db, secondary, reference, req) {
                Ok(c) => json!({
                    "reference": reference,
                    "count": c.len(),
                    "candidates": c,
                    "ms": t.elapsed().as_millis() as u64,
                }),
                Err(e) => json!({ "reference": reference, "error": format!("{e:#}") }),
            }
        };
        seen.insert(reference, v.clone());
        out.push(v);
    }
    out
}

/// The HTTP daemon's find-ref: the library it opened, and the secondary databases of the
/// moment — rebuilt when the app's list of them, or one of them, changes.
pub struct FindRefService {
    pub db: Arc<Db>,
    pub data_version: String,
    config: SecondaryConfig,
    secondary: RwLock<Arc<SecondarySet>>,
}

impl FindRefService {
    pub fn new(db: Arc<Db>, data_version: String, config: SecondaryConfig) -> FindRefService {
        let set = SecondarySet::build(&config);
        FindRefService {
            db,
            data_version,
            config,
            secondary: RwLock::new(Arc::new(set)),
        }
    }

    fn secondary(&self) -> Arc<SecondarySet> {
        let fp = self.config.fingerprint();
        {
            let cur = self.secondary.read().unwrap();
            if cur.fingerprint == fp {
                return Arc::clone(&cur);
            }
        }
        let set = Arc::new(SecondarySet::build(&self.config));
        *self.secondary.write().unwrap() = Arc::clone(&set);
        set
    }

    /// `POST /resolve_refs` {references: [...], limit, sources, exclude_book_level,
    /// heref_fallback, snippets} — the stdio tool's parameters and answer.
    pub fn handle(&self, body: &serde_json::Map<String, Value>) -> Result<Value, String> {
        let refs: Vec<String> = match body.get("references") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            _ => return Err("`references` must be an array of strings".into()),
        };
        if refs.len() > MAX_BATCH {
            return Err(format!("at most {MAX_BATCH} references per call"));
        }
        let b = |k: &str| body.get(k).and_then(Value::as_bool);
        let sources = body.get("sources").and_then(Value::as_array).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        });
        let limit = body
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n.min(50) as u32);
        let req = FindRefRequest::new(
            limit,
            sources,
            b("exclude_book_level"),
            b("heref_fallback"),
            b("snippets"),
        );
        let t = Instant::now();
        let results = resolve_batch(&self.db, &self.secondary(), &refs, &req);
        Ok(json!({
            "count": results.len(),
            "results": results,
            "resolver": RESOLVER,
            "data_version": self.data_version,
            "schema_version": self.db.schema_version(),
            "timing_ms": { "total": t.elapsed().as_millis() as u64 },
        }))
    }
}

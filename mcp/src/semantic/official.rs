//! Otzaria's own vectors, through Otzaria's own code: the semantic sidecar's
//! `OfficialSemanticIndex`, opened read-only (vendored patch, see VENDORED.md). It embeds a
//! query by the installed family's recipe — QA491's `query.int8.onnx` capped at
//! `query_max_tokens`, and its `document.fp32.onnx` for a text embedded as a passage — and
//! scans the installed set exactly.
//!
//! The model follows the vectors: the folder in Otzaria's books folder whose `model.json`
//! declares the set's model family. A new generation is picked up by `reload()`, and the old
//! one's mapping released, so Windows lets Otzaria delete its files; a generation built by
//! another model reopens the index with that model.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::semantic::embedding::EmbeddingDeployment;
use otzaria_semantic_search::semantic::official_index::{
    LocalModel, OfficialIndexConfig, OfficialSemanticIndex, ReloadOutcome,
};
use otzaria_semantic_search::semantic::resolve::{BookSet, SlotRef, VectorHit};
use otzaria_semantic_search::semantic::versioning::{ModelIdentity, TextIdentity};

/// What the installed set declares about itself (`vectors/CURRENT` -> `set.json`).
#[derive(Debug, Clone)]
pub struct SetDeclaration {
    pub generation: u64,
    pub family: String,
    pub model: ModelIdentity,
    pub text: TextIdentity,
    pub library_version: u32,
    pub release_tag: String,
}

pub fn read_set(vectors_dir: &Path) -> Result<SetDeclaration> {
    let cur: serde_json::Value = serde_json::from_slice(
        &std::fs::read(vectors_dir.join("CURRENT"))
            .with_context(|| format!("no vector set in {}", vectors_dir.display()))?,
    )?;
    let set_rel = cur["set"]
        .as_str()
        .ok_or_else(|| anyhow!("CURRENT names no set"))?;
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(vectors_dir.join(set_rel))?)?;
    let model: ModelIdentity = serde_json::from_value(doc["identity"]["model"].clone())
        .context("the set's model identity is unreadable")?;
    let text: TextIdentity = serde_json::from_value(doc["identity"]["text"].clone())
        .context("the set's text identity is unreadable")?;
    Ok(SetDeclaration {
        generation: doc["generation"].as_u64().unwrap_or(0),
        family: model.family_id.clone(),
        model,
        text,
        library_version: doc["library_version"].as_u64().unwrap_or(0) as u32,
        release_tag: doc["library_release_tag"]
            .as_str()
            .unwrap_or("")
            .to_string(),
    })
}

/// The model folder Otzaria downloaded for `family` (meivin-round2-onnx, meivin-qa491-onnx,
/// ...): the one whose model.json declares it.
pub fn find_model_dir(books_dir: &Path, family: &str) -> Result<PathBuf> {
    let mut found = Vec::new();
    if let Ok(rd) = std::fs::read_dir(books_dir) {
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs {
            let Ok(bytes) = std::fs::read(d.join("model.json")) else {
                continue;
            };
            let bytes = bytes
                .strip_prefix(b"\xef\xbb\xbf")
                .unwrap_or(&bytes)
                .to_vec();
            let fam = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v["family_id"].as_str().map(String::from));
            if fam.as_deref() == Some(family) {
                return Ok(d);
            }
            found.push(format!(
                "{}: {}",
                d.file_name().unwrap_or_default().to_string_lossy(),
                fam.unwrap_or_default()
            ));
        }
    }
    bail!(
        "no model folder in {} for the vector set's model {family:?}{}. Open Otzaria once so it \
         downloads the model, or pass --model-dir.",
        books_dir.display(),
        if found.is_empty() {
            String::new()
        } else {
            format!(" (found {})", found.join("; "))
        }
    )
}

/// The query graph of a model folder and its precision: QA491's `query.int8.onnx`, or the
/// int8 graph of a single-graph family (round 2).
fn query_graph(model_dir: &Path) -> Result<(PathBuf, &'static str)> {
    let q = model_dir.join("query.int8.onnx");
    if q.exists() {
        return Ok((q, "int8"));
    }
    let mut graphs: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "onnx"))
        .collect();
    graphs.sort();
    if let Some(g) = graphs.iter().find(|g| {
        g.file_name()
            .unwrap()
            .to_string_lossy()
            .to_lowercase()
            .contains("int8")
    }) {
        return Ok((g.clone(), "int8"));
    }
    match graphs.first() {
        Some(g) => Ok((g.clone(), "fp32")),
        None => bail!("no .onnx graph in {}", model_dir.display()),
    }
}

pub struct OfficialConfig {
    pub vectors_dir: PathBuf,
    pub books_dir: PathBuf,
    /// Fixed by --model-dir: the model does not follow the vectors then.
    pub model_dir: Option<PathBuf>,
    pub onnx_runtime: Option<PathBuf>,
    pub scan_threads: Option<usize>,
}

pub struct Official {
    pub index: OfficialSemanticIndex,
    pub vectors_dir: PathBuf,
    pub declared: SetDeclaration,
    pub model_dir: PathBuf,
    pub query_graph: PathBuf,
}

impl Official {
    pub fn open(cfg: &OfficialConfig) -> Result<Official> {
        let declared = read_set(&cfg.vectors_dir)?;
        let model_dir = match &cfg.model_dir {
            Some(d) => d.clone(),
            None => find_model_dir(&cfg.books_dir, &declared.family)?,
        };
        let (graph, quantization) = query_graph(&model_dir)?;
        let mut model = LocalModel::of_family(graph.clone(), &declared.model, quantization);
        model.model_quantization = quantization.to_string();
        let index = OfficialSemanticIndex::open_read_only(OfficialIndexConfig {
            vectors_dir: cfg.vectors_dir.clone(),
            text: declared.text.clone(),
            model,
            deployment: EmbeddingDeployment {
                onnx_runtime: cfg.onnx_runtime.clone(),
            },
            scan_threads: cfg.scan_threads.and_then(std::num::NonZeroUsize::new),
        })
        .map_err(|e| {
            anyhow!(
                "cannot open Otzaria's vectors at {}: {e}",
                cfg.vectors_dir.display()
            )
        })?;
        Ok(Official {
            index,
            vectors_dir: cfg.vectors_dir.clone(),
            declared,
            model_dir,
            query_graph: graph,
        })
    }

    /// The model family the open index embeds with.
    pub fn family(&self) -> &str {
        &self.declared.family
    }

    pub fn slots(&self) -> u64 {
        self.index.set_info().slots_live
    }

    pub fn segments(&self) -> usize {
        self.index.set_info().segments.len()
    }

    /// The query's vector, by the set's recipe (role prefix, normalization, query cap).
    pub fn embed_query(&self, q: &str) -> Result<Vec<f32>> {
        self.index
            .embed_query(q)
            .map_err(|e| anyhow!("cannot embed the query: {e}"))
    }

    /// The text's vector as a stored passage (QA491: the FP32 document graph).
    pub fn embed_passage(&self, t: &str) -> Result<Vec<f32>> {
        let cancel = CancellationToken::new();
        let mut v = self
            .index
            .embed_passages(&[t], &cancel)
            .map_err(|e| anyhow!("cannot embed the passage: {e}"))?;
        v.pop().ok_or_else(|| anyhow!("no vector"))
    }

    pub fn scan(
        &self,
        vector: &[f32],
        k: usize,
        books: Option<&BookSet>,
    ) -> Result<Vec<VectorHit>> {
        let cancel = CancellationToken::new();
        self.index
            .search_hits_with(
                vector,
                k,
                books,
                &[],
                super::calibration::window_penalty(),
                &cancel,
            )
            .map_err(|e| anyhow!("the vector scan failed: {e}"))
    }

    /// The exact score of each of `slots` (of this generation) against `vector`, as the scan
    /// scores them (the long-window penalty included): a scan that admits no book and weighs
    /// these slots besides. A slot no longer live, or holding another text, is left out.
    pub fn score_slots(&self, vector: &[f32], slots: &[SlotRef]) -> Result<Vec<VectorHit>> {
        if slots.is_empty() {
            return Ok(Vec::new());
        }
        let none = BookSet::default();
        let cancel = CancellationToken::new();
        self.index
            .search_hits_with(
                vector,
                slots.len(),
                Some(&none),
                slots,
                super::calibration::window_penalty(),
                &cancel,
            )
            .map_err(|e| anyhow!("the rescore failed: {e}"))
    }

    /// Pick up a new generation. `Ok(true)` when the set changed.
    pub fn reload(&mut self) -> Result<bool> {
        match self.index.reload_vectors() {
            Ok(ReloadOutcome::Unchanged { .. }) => Ok(false),
            Ok(_) => {
                if let Ok(d) = read_set(&self.vectors_dir) {
                    self.declared = d;
                }
                Ok(true)
            }
            Err(e) => Err(anyhow!("{e}")),
        }
    }
}

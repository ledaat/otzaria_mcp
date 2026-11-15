# Vendored: otzaria-semantic-search (the semantic sidecar)
`otzaria-semantic-search` at rev `a79901ff1a34bc9d57b994b905de7ec9ad69bb7a`
(the QA491 role caps: queries through `query.int8.onnx` capped at `query_max_tokens`, passages
through `document.fp32.onnx`), copied as `Cargo.toml`, `src/`, `config/`, `benches/` and the two
licences. The workspace `[patch."https://github.com/Otzaria/otzaria-semantic-search"]` points every
git dependency on it here (the MCP's own; the search core's is optional and off). The MCP uses its `OfficialSemanticIndex` directly, with the
`onnx-backend` feature: the query is embedded and the installed vector set scanned by Otzaria's
own code, and ONNX Runtime is loaded at run time (`--onnx-runtime`).

Two local changes, in `src/semantic/official_index.rs`:

- **`OfficialSemanticIndex::open_read_only(config)`**: [`open`] with a `read_only` flag kept on
  the index, so the first open and every `reload_vectors` go through
  `SegmentSet::open_without_recovery`. Upstream `open` takes the set's lock when it can and then
  recovers `staging/` and collects garbage in Otzaria's `vectors/` folder; a reader beside the
  application must never do either, nor make an install wait for the lock.
- **`OfficialSemanticIndex::embed_query`** is `pub` (upstream `pub(crate)`): the MCP scans its
  personal books' vectors, built with the same model, with the same query vector.

## Upgrading
A family with a newer text recipe (Pardes-QA491: embedding text 3, normalization 2, added to the
sidecar after this revision) needs a newer sidecar: copy it here, re-apply the two changes above,
and move the `rev` in `mcp/Cargo.toml` with it.

The search engine itself is no longer vendored: `mcp/Cargo.toml` pins Otzaria's
`otzaria-search-core` by git revision (see the workspace `Cargo.toml`).

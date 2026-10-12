# Vendored: otzaria_search_engine (rust crate `search_engine`) — v3.1.0

Copied from `github.com/Otzaria/otzaria_search_engine`, tag **v3.1.0**, `rust/` only.
It matches the index current Otzaria builds: `otzaria_index_meta.json` →
`schema_version: 5`, `engine_version: 3.1.0`, `tantivy_version: 0.26.2`
(`INDEX_SCHEMA_VERSION = MIN_READABLE_SCHEMA_VERSION = 5`).

(Replaces the earlier 0.6.7 copy, which only reads schema-2 indexes; against a
schema-5 index the MCP started with "full-text search disabled".)

## Local modifications (all marked "otzaria-mcp local patch")
1. `src/read_only_dir.rs` (new): `ReadOnlyDirectory`, an `MmapDirectory` that refuses
   every write and takes no lock file — lifted verbatim from upstream's
   `smart_search_bench.rs`, where it is test-only.
2. `src/lib.rs`: `mod read_only_dir;`
3. `src/api/search_engine.rs`:
   - new `SearchEngine::open_read_only(path) -> Result<Self>`: checks
     `check_index_compatibility`, opens through `ReadOnlyDirectory`, never panics.
   - `from_read_only_index` un-gated from `#[cfg(test)]` (body unchanged).

## Workspace-level requirements
The engine's own `[patch.crates-io]` (Otzaria's tantivy fork at rev 85b76fe…, and
the vendored tantivy-fst with the raised DFA state limit) must be repeated in the
MCP workspace root `Cargo.toml`, because `[patch]` only applies at the root.

## MCP-side changes for the 3.x API (`mcp/src/search.rs`, `mcp/src/reload.rs`)
- `search_and_count_exact/fuzzy` gained `match_nikud, match_taamim, grouping`;
  `search_and_count_advanced` gained negative-query, scope, grouping and
  word-match parameters. The MCP passes neutral defaults (nikud/taamim-insensitive,
  no negative query, `SearchScope::WordDistance`, no grouping) — the old behaviour.
- 3.x stores most official-book line text in `seforim.db` (`TextStorage::LibraryDb`),
  so `Search::open` now takes the db path and calls `configure_line_source` (the
  engine opens it read-only, lazily) so snippets are filled in.

## Upgrading again
Re-copy the matching tag's `rust/`, re-apply the three patches above, re-sync the
root `[patch.crates-io]` with the engine's own, then `cargo test -p otzaria-mcp`.

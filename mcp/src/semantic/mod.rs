//! Semantic search: the tools and HTTP API that otzaria-sources' Python server provided
//! (`otzaria_semantic_server.py` + `otzaria_mcp.py`), now inside otzaria-mcp.
//!
//! - [`official`]: Otzaria's installed vector set, opened read-only through the sidecar.
//! - [`personal`]: the personal-books index built by `build_personal_index.py`.
//! - [`service`]: search, related (a whole note), check_quote, text, feedback, status.
//! - [`http`]: the `--http` daemon the Obsidian plugin and the stdio MCP talk to.
//! - [`client`]: the stdio MCP's side: find or start the daemon, forward a call.

pub mod align;
pub mod catalog;
pub mod client;
pub mod feedback;
pub mod format;
pub mod http;
pub mod official;
pub mod personal;
pub mod service;
pub mod textutil;

#[cfg(test)]
mod parity_tests;

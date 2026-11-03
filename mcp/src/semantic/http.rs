//! `otzaria-mcp --http 127.0.0.1:8765`: the semantic service over HTTP, for the Obsidian plugin
//! (otzaria-sources), the Otzaria plugin מקבילות, and the stdio MCP's semantic tools. The routes
//! and their JSON are the Python server's:
//!
//! ```text
//!   GET  /health                 status, model, library version, slot counts, version
//!   GET  /categories             the library's category tree (ids, titles, book counts)
//!   GET  /personal/status        the personal index: what is indexed, what changed since
//!   POST /search   {query, k, ...}
//!   POST /related  {text, title, k, ...}
//!   POST /text     {url | book_id+line_index[+source] | ref, before, after}
//!   POST /check_quote {quote, [url | book_id+line_index | ref, span]}
//!   POST /feedback {type: like|dislike|entry, ...} or {event: {...}}
//!   POST /shutdown               (a newer plugin or MCP replacing this server)
//! ```


use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde_json::{json, Map, Value};
use tiny_http::{Header, Method, Request, Response, Server};

use super::service::{BadRequest, SemanticService, VERSION};

fn respond(req: Request, code: u16, body: &Value) {
    let data = serde_json::to_vec(body).unwrap_or_default();
    let mut resp = Response::from_data(data).with_status_code(code);
    resp.add_header(Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap());
    resp.add_header(Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap());
    let _ = req.respond(resp);
}

fn handle(svc: &SemanticService, mut req: Request, server: &Server) -> bool {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    if method == Method::Options {
        let mut resp = Response::empty(204);
        for (k, v) in [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type"),
        ] {
            resp.add_header(Header::from_bytes(k, v).unwrap());
        }
        let _ = req.respond(resp);
        return false;
    }
    if method == Method::Get {
        let out = if path.starts_with("/health") {
            Some(svc.health())
        } else if path.starts_with("/categories") {
            Some(svc.categories())
        } else if path.starts_with("/personal/status") {
            Some(svc.personal_status())
        } else {
            None
        };
        match out {
            Some(v) => respond(req, 200, &v),
            None => respond(req, 404, &json!({"error": "not found"})),
        }
        return false;
    }
    if method != Method::Post {
        respond(req, 405, &json!({"error": "method not allowed"}));
        return false;
    }
    let mut body = String::new();
    if req.as_reader().read_to_string(&mut body).is_err() {
        respond(req, 400, &json!({"error": "the body is not UTF-8"}));
        return false;
    }
    let opts: Map<String, Value> = match serde_json::from_str::<Value>(if body.trim().is_empty() { "{}" } else { &body }) {
        Ok(Value::Object(m)) => m,
        Ok(_) => {
            respond(req, 400, &json!({"error": "the body must be a JSON object"}));
            return false;
        }
        Err(e) => {
            respond(req, 400, &json!({"error": format!("invalid JSON: {e}")}));
            return false;
        }
    };
    if path.starts_with("/shutdown") {
        respond(req, 200, &json!({"ok": true, "version": VERSION}));
        server.unblock();
        return true;
    }
    let t = Instant::now();
    let result: Result<Value> = if path.starts_with("/search") {
        svc.search(&opts)
    } else if path.starts_with("/related") {
        svc.related(&opts)
    } else if path.starts_with("/text") {
        svc.text(&opts)
    } else if path.starts_with("/check_quote") {
        svc.check_quote(&opts)
    } else if path.starts_with("/feedback") {
        svc.feedback(&opts)
    } else {
        respond(req, 404, &json!({"error": "not found"}));
        return false;
    };
    match result {
        Ok(mut v) => {
            if let Some(Value::Object(tm)) = v.get_mut("timing_ms") {
                tm.insert("total".into(), json!(t.elapsed().as_millis()));
            }
            respond(req, 200, &v);
        }
        Err(e) => {
            if let Some(b) = e.downcast_ref::<BadRequest>() {
                respond(req, 400, &json!({"error": b.0}));
            } else {
                eprintln!("otzaria-mcp: {path} failed: {e:#}");
                respond(req, 500, &json!({"error": format!("{e:#}")}));
            }
        }
    }
    false
}

/// Serve until /shutdown. Requests are answered by a few worker threads, so a long `related`
/// does not hold up /health.
pub fn serve(svc: Arc<SemanticService>, addr: &str, workers: usize) -> Result<()> {
    let server = Arc::new(Server::http(addr).map_err(|e| anyhow::anyhow!("cannot listen on {addr}: {e}"))?);
    eprintln!("otzaria-mcp: semantic server listening on http://{addr}");
    svc.watch();
    let mut handles = Vec::new();
    for i in 0..workers.max(1) {
        let (svc, server) = (svc.clone(), server.clone());
        handles.push(
            std::thread::Builder::new().name(format!("http-{i}")).spawn(move || {
                while let Ok(req) = server.recv() {
                    if handle(&svc, req, &server) {
                        // wake the other workers too
                        for _ in 0..workers {
                            server.unblock();
                        }
                        break;
                    }
                }
            })?,
        );
    }
    for h in handles {
        let _ = h.join();
    }
    eprintln!("otzaria-mcp: semantic server stopped");
    Ok(())
}

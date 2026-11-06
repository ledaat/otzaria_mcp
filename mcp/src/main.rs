//! otzaria-mcp — an offline MCP server over the Otzaria library.
//! Thin binary: parse args, resolve paths, open the library, serve over stdio — or, with
//! `--http`, run the semantic server the Obsidian plugin and the semantic tools talk to.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rmcp::transport::stdio;
use rmcp::ServiceExt;

use otzaria_mcp::paths::PathArgs;
use otzaria_mcp::reload::Library;
use otzaria_mcp::server::OtzariaServer;

const USAGE: &str = "\
otzaria-mcp — MCP server over the Otzaria library (read-only, offline)

USAGE:
    otzaria-mcp [OPTIONS]

Transport: stdio (MCP JSON-RPC on stdin/stdout; logs go to stderr).

PATH OPTIONS (precedence: flags > env > --data-dir > per-OS defaults):
    --db <path>           Path to seforim.db            [env: OTZARIA_DB]
    --index <path>        Path to the Tantivy index dir [env: OTZARIA_INDEX]
    --lexical-db <path>   Path to lexical.db (fuzzy)    [env: OTZARIA_LEXICAL_DB]
    --dictionaries-dir <dir>
                          Folder with dictionary.json / Acronyms.json (advanced-search
                          translation and acronym options) [env: OTZARIA_DICTIONARIES_DIR]
    --data-dir <root>     Otzaria data root; two layouts auto-detected:
                            <root>/otzaria/seforim.db + <root>/index   (download root)
                            <root>/seforim.db          + <root>/index   (library dir)
                                                        [env: OTZARIA_DATA_DIR]
    (defaults: %APPDATA%\\otzaria then %ProgramData%\\otzaria on Windows;
     ~/Library/Application Support/otzaria then /Library/Application Support/otzaria
     on macOS; ~/.local/share/otzaria then /var/lib/otzaria on Linux —
     <root>/books/seforim.db and <root>/index)
    --snapshot            Open the DB with immutable=1 (closed snapshot; ignores WAL)

SECONDARY SOURCES (personal books and attached databases, as the app's find-ref sees them;
read only by resolve_ref when asked for — see its `sources` parameter):
    --app-data-dir <dir>  The app's data folder (app_preferences.hive, databases/)
                                                        [env: OTZARIA_APP_DATA_DIR]
                          (default: the folder holding books/, else the per-OS root)
    --user-books-db <path>
                          Personal books database        [env: OTZARIA_USER_BOOKS_DB]
                          (default: <app data>/databases/user_books.db)
    --attached <slug>=<path>
                          An attached database; repeatable. Replaces the app's own list
                          (read from app_preferences.hive and followed live)
                                                        [env: OTZARIA_ATTACHED, `;`-separated]
    --no-secondary        Library only

SEMANTIC SEARCH (Otzaria's own model and vectors, plus the personal-books index):
    --http <host:port>    Run the semantic server (the Obsidian plugin's API) instead of the
                          stdio MCP. Default address 127.0.0.1:8765 (--port <n> sets the port).
    --semantic-url <url>  Where the stdio MCP's semantic tools find that server; it is started
                          (this binary with --http) when none answers there
                                                        [default: http://127.0.0.1:8765]
    --no-semantic         No semantic tools
    --vectors-dir <dir>   Otzaria's vector set         [default: <app data>/vectors]
    --model-dir <dir>     The model folder (default: the one in the books folder whose
                          model.json matches the vector set's model; it follows the vectors)
    --onnx-runtime <path> onnxruntime.dll / libonnxruntime.so [env: OTZARIA_ONNX_RUNTIME]
                          (default: Otzaria's own, <install dir>/onnxruntime/)
    --personal-index <dir>
                          The personal-books index of build_personal_index.py
                          [default: %LOCALAPPDATA%\\otzaria-sources\\personal-index]
    --builder <path>      build_personal_index.py, for /personal/status
    --python <cmd>        Python for the builder's check [default: python (Windows), python3]
    --plugin-data <path>  The Obsidian plugin's data.json (context defaults of the tools)
    --semantic-log <path> Where a started server writes its log
    --scan-threads <n>    Threads per vector scan       [default: all cores]
    --workers <n>         HTTP worker threads           [default: 4]
    --no-feedback         Do not relay relevance feedback (HTTP mode)
    -h, --help            Print this help
    -V, --version         Print version
";

#[derive(Default)]
struct SemanticArgs {
    http: Option<String>,
    semantic_url: Option<String>,
    no_semantic: bool,
    vectors_dir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    onnx_runtime: Option<PathBuf>,
    personal_index: Option<PathBuf>,
    builder: Option<PathBuf>,
    python: Option<String>,
    plugin_data: Option<PathBuf>,
    semantic_log: Option<PathBuf>,
    scan_threads: Option<usize>,
    workers: Option<usize>,
    no_feedback: bool,
    /// Every argument a started daemon gets after `--http <addr>`.
    passthrough: Vec<OsString>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let (args, sem) = match parse_args()? {
        Some(a) => a,
        None => return Ok(()), // --help / --version already printed
    };

    let paths = args.with_env().resolve()?;
    if let Some(addr) = sem.http.clone() {
        return tokio::task::spawn_blocking(move || run_http(paths, sem, &addr)).await?;
    }
    eprintln!(
        "otzaria-mcp: db={} index={} lexical={} dictionaries={} snapshot={}",
        paths.db.display(),
        paths
            .index
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into()),
        paths
            .lexical_db
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into()),
        paths
            .dictionaries_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into()),
        paths.snapshot,
    );
    for spec in paths.secondary.discover() {
        eprintln!(
            "otzaria-mcp: {} source {} = {}",
            spec.source.kind(),
            spec.name,
            spec.path.display()
        );
    }

    // Library holds the current DB+index snapshot and hot-reloads it when the
    // underlying files change (opening the index logs to stderr internally).
    let lib = Library::open(paths)?;
    {
        let snap = lib.current();
        eprintln!(
            "otzaria-mcp: db_version={:?} db_schema_version={:?} data_version={}",
            snap.schema_version, snap.db_schema_version, snap.fingerprint
        );
    }
    let watcher = tokio::spawn(Arc::clone(&lib).watch());

    let mut server = OtzariaServer::new(lib.clone());
    if !sem.no_semantic {
        let url = sem
            .semantic_url
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:8765".into());
        let log = sem.semantic_log.clone().or_else(|| {
            sem.plugin_data
                .as_ref()
                .and_then(|d| d.parent())
                .map(|d| d.join("otzaria_server.log"))
        });
        eprintln!("otzaria-mcp: semantic tools through {url}");
        server = server.with_semantic(otzaria_mcp::server::SemanticTools {
            client: otzaria_mcp::semantic::client::DaemonClient::new(
                &url,
                sem.passthrough.clone(),
                log,
            ),
            plugin_data: sem.plugin_data.clone(),
        });
    }
    let result = async {
        let service = server.serve(stdio()).await?;
        service.waiting().await?;
        Ok(())
    }
    .await;
    // rmcp drains responses at EOF; once the transport is closed, no queued
    // request or reload should start. Running synchronous operations finish.
    lib.shutdown();
    watcher.abort();
    result
}

/// The semantic server: Otzaria's vectors, the personal index and the library catalogue
/// behind the HTTP API, until /shutdown.
fn run_http(paths: otzaria_mcp::paths::Paths, sem: SemanticArgs, addr: &str) -> Result<()> {
    use otzaria_mcp::semantic::{http, service};
    let app =
        otzaria_mcp::secondary::app_data_dir(paths.secondary.app_data_dir.clone(), &paths.db, &[])
            .or_else(|| {
                paths
                    .db
                    .parent()
                    .and_then(Path::parent)
                    .map(Path::to_path_buf)
            })
            .ok_or_else(|| {
                anyhow::anyhow!("cannot tell Otzaria's data folder; pass --app-data-dir")
            })?;
    let books_dir = paths
        .db
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| app.join("books"));
    let personal_dir = sem
        .personal_index
        .clone()
        .unwrap_or_else(default_personal_index);
    let cfg = service::SemanticConfig {
        otzaria_dir: app.clone(),
        books_dir,
        db: paths.db.clone(),
        snapshot: paths.snapshot,
        vectors_dir: sem
            .vectors_dir
            .clone()
            .unwrap_or_else(|| app.join("vectors")),
        model_dir: sem.model_dir.clone(),
        onnx_runtime: sem
            .onnx_runtime
            .clone()
            .or_else(|| std::env::var_os("OTZARIA_ONNX_RUNTIME").map(PathBuf::from))
            .or_else(default_onnx_runtime),
        scan_threads: sem.scan_threads,
        feedback_dir: if sem.no_feedback {
            None
        } else {
            personal_dir.parent().map(|d| d.join("feedback-outbox"))
        },
        personal_dir,
        user_books_db: paths
            .secondary
            .user_books_db
            .clone()
            .unwrap_or_else(|| app.join("databases").join("user_books.db")),
        builder: sem.builder.clone(),
        python: sem.python.clone().unwrap_or_else(|| {
            if cfg!(windows) {
                "python".into()
            } else {
                "python3".into()
            }
        }),
    };
    eprintln!(
        "otzaria-mcp {} --http {addr}: db={} vectors={} personal={} onnxruntime={}",
        env!("CARGO_PKG_VERSION"),
        cfg.db.display(),
        cfg.vectors_dir.display(),
        cfg.personal_dir.display(),
        cfg.onnx_runtime
            .as_ref()
            .map_or("(sidecar default)".into(), |p| p.display().to_string())
    );
    let refs = otzaria_mcp::db::Db::open(&paths)
        .map(Arc::new)
        .map_err(|e| eprintln!("otzaria-mcp: references unavailable: {e:#}"))
        .ok();
    // find-ref over the same library, plus personal books and attached databases
    let findref = refs.clone().map(|db| {
        Arc::new(otzaria_mcp::findref::FindRefService::new(
            db,
            otzaria_mcp::reload::fingerprint(&paths),
            paths.secondary.clone(),
        ))
    });
    let svc = Arc::new(service::SemanticService::open(cfg, refs)?);
    http::serve(svc, findref, addr, sem.workers.unwrap_or(4))
}

fn default_personal_index() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("otzaria-sources").join("personal-index")
}

/// Otzaria's own ONNX Runtime: `<install dir>/onnxruntime/onnxruntime.dll`, the installer's
/// default install folders first.
fn default_onnx_runtime() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
        if let Some(v) = std::env::var_os(var) {
            dirs.push(PathBuf::from(v).join("Otzaria"));
        }
    }
    if let Some(v) = std::env::var_os("LOCALAPPDATA") {
        dirs.push(PathBuf::from(v).join("Programs").join("Otzaria"));
    }
    let lib = if cfg!(windows) {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    dirs.into_iter()
        .map(|d| d.join("onnxruntime").join(lib))
        .find(|p| p.is_file())
}

/// Flags followed by a value.
const VALUE_FLAGS: &[&str] = &[
    "--app-data-dir",
    "--user-books-db",
    "--attached",
    "--db",
    "--index",
    "--lexical-db",
    "--data-dir",
    "--dictionaries-dir",
    "--port",
    "--semantic-url",
    "--vectors-dir",
    "--model-dir",
    "--onnx-runtime",
    "--personal-index",
    "--builder",
    "--python",
    "--plugin-data",
    "--semantic-log",
    "--scan-threads",
    "--workers",
];

/// Flags a started daemon does not get (they are about this process, not the data).
const NOT_PASSED: &[&str] = &[
    "--http",
    "--port",
    "--semantic-url",
    "--no-semantic",
    "--warm",
    "--plugin-data",
    "--semantic-log",
];

/// Parse CLI args into `PathArgs` and the semantic options. Returns `Ok(None)` when
/// help/version was printed.
fn parse_args() -> Result<Option<(PathArgs, SemanticArgs)>> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = PathArgs::default();
    let mut sem = SemanticArgs::default();
    let mut port: Option<String> = None;
    let mut http_addr: Option<String> = None;
    let mut i = 0;
    while i < raw.len() {
        let arg = raw[i].as_str();
        // `--http` takes an optional address
        let value = if VALUE_FLAGS.contains(&arg) {
            i += 1;
            Some(
                raw.get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing value for {arg}"))?,
            )
        } else if arg == "--http" && raw.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
            i += 1;
            Some(raw[i].clone())
        } else {
            None
        };
        if !NOT_PASSED.contains(&arg) {
            sem.passthrough.push(arg.into());
            if let Some(v) = &value {
                sem.passthrough.push(v.into());
            }
        }
        let path = || PathBuf::from(value.clone().unwrap_or_default());
        match arg {
            "-h" | "--help" => {
                eprint!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                // stdout, so the Obsidian plugin can read the version it will run
                println!("otzaria-mcp {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--snapshot" => args.snapshot = true,
            "--no-secondary" => args.no_secondary = true,
            "--app-data-dir" => args.app_data_dir = Some(path()),
            "--user-books-db" => args.user_books_db = Some(path()),
            "--attached" => {
                let v = value.clone().unwrap_or_default();
                let pair = otzaria_mcp::paths::parse_attached_arg(&v)
                    .ok_or_else(|| anyhow::anyhow!("--attached expects <slug>=<path>, got {v}"))?;
                args.attached.push(pair);
            }
            "--db" => args.db = Some(path()),
            "--index" => args.index = Some(path()),
            "--lexical-db" => args.lexical_db = Some(path()),
            "--data-dir" => args.data_dir = Some(path()),
            "--dictionaries-dir" => args.dictionaries_dir = Some(path()),
            "--http" => http_addr = Some(value.clone().unwrap_or_default()),
            "--port" => port = value.clone(),
            "--semantic-url" => sem.semantic_url = value.clone(),
            "--no-semantic" => sem.no_semantic = true,
            "--warm" => {} // accepted for the old plugin's command line; the sidecar maps lazily
            "--vectors-dir" => sem.vectors_dir = Some(path()),
            "--model-dir" => sem.model_dir = Some(path()),
            "--onnx-runtime" => sem.onnx_runtime = Some(path()),
            "--personal-index" => sem.personal_index = Some(path()),
            "--builder" => sem.builder = Some(path()),
            "--python" => sem.python = value.clone(),
            "--plugin-data" => sem.plugin_data = Some(path()),
            "--semantic-log" => sem.semantic_log = Some(path()),
            "--scan-threads" => sem.scan_threads = Some(value.clone().unwrap_or_default().parse()?),
            "--workers" => sem.workers = Some(value.clone().unwrap_or_default().parse()?),
            "--no-feedback" => sem.no_feedback = true,
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
        i += 1;
    }
    if let Some(a) = http_addr {
        sem.http = Some(if !a.is_empty() {
            a
        } else {
            format!("127.0.0.1:{}", port.as_deref().unwrap_or("8765"))
        });
    }
    Ok(Some((args, sem)))
}

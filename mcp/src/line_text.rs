//! Reading a line's text across `seforim.db` layouts.
//!
//! Older databases keep the text in `line.content`. Current ones (Otzaria's split layout)
//! move it to `line_content(id, content)`, keyed by `line.id`, where each value is either
//! TEXT or a zstd frame compressed with one of the dictionaries in `zstd_dict` (chosen by
//! the dictionary id in the frame header). [`LineText`] detects the layout once when the
//! database is opened, gives the SQL expression for a line's content, and decodes the value.

use anyhow::{Context, Result};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashMap;
use std::sync::Mutex;

/// Largest decoded line accepted (the same bound Otzaria's search engine uses).
const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

pub struct LineText {
    /// `line_content` exists: the text is there, not in `line.content`.
    split: bool,
    /// Present when the database has a `zstd_dict` table.
    codec: Option<Mutex<ZstdCodec>>,
}

impl LineText {
    /// Inspect the database layout (and load the zstd dictionaries, if any).
    pub fn detect(conn: &Connection) -> Result<Self> {
        let has_table = |name: &str| -> Result<bool> {
            Ok(conn
                .query_row(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    [name],
                    |_| Ok(()),
                )
                .optional()?
                .is_some())
        };
        let split = has_table("line_content")?;
        let codec = if has_table("zstd_dict")? {
            Some(Mutex::new(ZstdCodec::load(conn)?))
        } else {
            None
        };
        Ok(Self { split, codec })
    }

    /// The old layout: text in `line.content`, no compression.
    pub fn inline() -> Self {
        Self {
            split: false,
            codec: None,
        }
    }

    /// SQL expression for the content of the `line` row aliased `alias`.
    pub fn column(&self, alias: &str) -> String {
        if self.split {
            format!("(SELECT lc.content FROM line_content lc WHERE lc.id = {alias}.id)")
        } else {
            format!("{alias}.content")
        }
    }

    /// Column `idx` of a row as text: TEXT as is, NULL as empty, BLOB decompressed.
    pub fn get(&self, row: &rusqlite::Row<'_>, idx: usize) -> rusqlite::Result<String> {
        self.decode(row.get_ref(idx)?).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(idx, row_type(row, idx), e.into())
        })
    }

    fn decode(&self, value: ValueRef<'_>) -> Result<String> {
        match value {
            ValueRef::Text(bytes) => Ok(String::from_utf8_lossy(bytes).into_owned()),
            ValueRef::Null => Ok(String::new()),
            ValueRef::Blob(frame) => {
                let codec = self
                    .codec
                    .as_ref()
                    .context("compressed line text in a database without zstd_dict")?;
                let bytes = codec.lock().unwrap().decode(frame)?;
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
            _ => anyhow::bail!("line content is neither TEXT nor BLOB"),
        }
    }
}

fn row_type(row: &rusqlite::Row<'_>, idx: usize) -> rusqlite::types::Type {
    row.get_ref(idx)
        .map(|v| v.data_type())
        .unwrap_or(rusqlite::types::Type::Null)
}

/// zstd decoder holding every dictionary of `zstd_dict`, by dictionary id.
struct ZstdCodec {
    dicts: HashMap<u32, zstd::zstd_safe::DDict<'static>>,
    dctx: zstd::zstd_safe::DCtx<'static>,
}

impl ZstdCodec {
    fn load(conn: &Connection) -> Result<Self> {
        let mut dicts = HashMap::new();
        let mut stmt = conn.prepare("SELECT dict FROM zstd_dict")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0).context("zstd_dict.dict is not a BLOB")?;
            let id = zstd::zstd_safe::get_dict_id_from_dict(&bytes).map_or(0, |id| id.get());
            let ddict = zstd::zstd_safe::DDict::try_create(&bytes)
                .context("cannot load a zstd_dict dictionary")?;
            dicts.insert(id, ddict);
        }
        let dctx = zstd::zstd_safe::DCtx::try_create().context("cannot create a zstd context")?;
        Ok(Self { dicts, dctx })
    }

    fn decode(&mut self, frame: &[u8]) -> Result<Vec<u8>> {
        let size = match zstd::zstd_safe::get_frame_content_size(frame) {
            Ok(Some(size)) if size <= MAX_LINE_BYTES => size as usize,
            other => anyhow::bail!("invalid zstd frame (content size {other:?})"),
        };
        // SAFETY: reads at most `frame.len()` bytes of a valid slice.
        let dict_id = unsafe {
            zstd::zstd_safe::zstd_sys::ZSTD_getDictID_fromFrame(frame.as_ptr().cast(), frame.len())
        };
        let ddict = self
            .dicts
            .get(&dict_id)
            .with_context(|| format!("zstd dictionary {dict_id} is missing from zstd_dict"))?;
        let mut out: Vec<u8> = Vec::with_capacity(size);
        let written = self
            .dctx
            .decompress_using_ddict(&mut out, frame, ddict)
            .map_err(|code| {
                anyhow::anyhow!(
                    "zstd decode failed: {}",
                    zstd::zstd_safe::get_error_name(code)
                )
            })?;
        if written != size {
            anyhow::bail!("zstd frame declared {size} bytes and produced {written}");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::LineText;
    use rusqlite::Connection;

    /// A dictionary-compressed frame as Otzaria writes it (content size in the header).
    fn compress(text: &str, dict: &[u8]) -> Vec<u8> {
        let mut cctx = zstd::zstd_safe::CCtx::create();
        cctx.load_dictionary(dict).unwrap();
        cctx.set_pledged_src_size(Some(text.len() as u64)).unwrap();
        let mut out = Vec::with_capacity(zstd::zstd_safe::compress_bound(text.len()) + 64);
        cctx.compress2(&mut out, text.as_bytes()).unwrap();
        out
    }

    #[test]
    fn reads_inline_split_text_and_compressed_rows() {
        let dict = "בראשית ברא אלהים את השמים ואת הארץ "
            .repeat(20)
            .into_bytes();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE line (id INTEGER PRIMARY KEY, bookId INTEGER, lineIndex INTEGER);
             CREATE TABLE line_content (id INTEGER PRIMARY KEY, content);
             CREATE TABLE zstd_dict (id INTEGER PRIMARY KEY, dict BLOB);
             INSERT INTO line VALUES (1, 1, 0), (2, 1, 1), (3, 1, 2);
             INSERT INTO line_content (id, content) VALUES (1, 'טקסט פשוט');",
        )
        .unwrap();
        conn.execute("INSERT INTO zstd_dict VALUES (1, ?1)", [&dict])
            .unwrap();
        let frame = compress("<b>בראשית ברא</b> אלהים", &dict);
        conn.execute("INSERT INTO line_content VALUES (2, ?1)", [&frame])
            .unwrap();

        let lt = LineText::detect(&conn).unwrap();
        let sql = format!(
            "SELECT {} FROM line l WHERE l.bookId = 1 ORDER BY l.lineIndex",
            lt.column("l")
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let texts: Vec<String> = stmt
            .query_map([], |r| lt.get(r, 0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(texts, ["טקסט פשוט", "<b>בראשית ברא</b> אלהים", ""]);
    }

    #[test]
    fn old_layout_reads_line_content_column() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE line (id INTEGER PRIMARY KEY, content TEXT);
             INSERT INTO line VALUES (1, 'ישן');",
        )
        .unwrap();
        let lt = LineText::detect(&conn).unwrap();
        assert_eq!(lt.column("l"), "l.content");
        let s: String = conn
            .query_row(&format!("SELECT {} FROM line l", lt.column("l")), [], |r| {
                lt.get(r, 0)
            })
            .unwrap();
        assert_eq!(s, "ישן");
    }
}

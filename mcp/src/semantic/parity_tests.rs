//! Parity with the Python server: `parity_fixtures.json` holds what the Python helpers
//! (otzaria-sources' `otzaria_semantic_server.py`, `personal_index.py`) return for the same
//! inputs; generate it again with `tests/gen_semantic_fixtures.py`.

use serde_json::Value;

use super::align::{align_words, Op};
use super::personal::text_hash;
use super::textutil::*;

fn fixtures() -> Value {
    serde_json::from_str(include_str!("parity_fixtures.json")).unwrap()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn window_spans_match_python() {
    for case in fixtures()["window_spans"].as_array().unwrap() {
        let t = case["text"].as_str().unwrap();
        let want: Vec<(usize, usize)> = case["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(
            window_spans(t),
            want,
            "spans of a {}-character line",
            char_len(t)
        );
        if let Some(s3) = case["server_span_3"].as_array() {
            let got = window_span(t, 3);
            assert_eq!(
                (got.0 as u64, got.1 as u64),
                (s3[0].as_u64().unwrap(), s3[1].as_u64().unwrap())
            );
        }
    }
}

#[test]
fn text_helpers_match_python() {
    for case in fixtures()["text"].as_array().unwrap() {
        let s = case["s"].as_str().unwrap();
        assert_eq!(
            strip_html(s),
            case["strip_html"].as_str().unwrap(),
            "strip_html({s:?})"
        );
        assert_eq!(
            line_text(s),
            case["line_text"].as_str().unwrap(),
            "line_text({s:?})"
        );
        let got: Vec<String> = s.split_whitespace().map(norm_word).collect();
        assert_eq!(got, strs(&case["norm_words"]), "norm_word over {s:?}");
    }
}

#[test]
fn align_matches_python() {
    let mut checked = 0;
    for case in fixtures()["align"].as_array().unwrap() {
        let (qo, qw, co, cw) = (
            strs(&case["qo"]),
            strs(&case["qw"]),
            strs(&case["co"]),
            strs(&case["cw"]),
        );
        let got = align_words(&qw, &cw, &qo, &co);
        match (&case["result"], got) {
            (Value::Null, None) => {}
            (Value::Null, Some(g)) => {
                panic!("Python found no alignment, Rust {g:?} for {qo:?} / {co:?}")
            }
            (_, None) => panic!("Rust found no alignment for {qo:?} / {co:?}"),
            (want, Some(g)) => {
                assert!(
                    (g.ratio - want["ratio"].as_f64().unwrap()).abs() < 1e-12,
                    "ratio for {qo:?} / {co:?}"
                );
                assert_eq!(
                    g.start as u64,
                    want["start"].as_u64().unwrap(),
                    "start for {qo:?} / {co:?}"
                );
                assert_eq!(
                    g.end as u64,
                    want["end"].as_u64().unwrap(),
                    "end for {qo:?} / {co:?}"
                );
                let ops: Vec<(String, u64, u64, u64, u64)> = g
                    .ops
                    .iter()
                    .map(|(op, a, b, c, d)| {
                        let name = match op {
                            Op::Replace => "replace",
                            Op::Delete => "delete",
                            Op::Insert => "insert",
                            Op::Abbr => "abbr",
                        };
                        (name.to_string(), *a as u64, *b as u64, *c as u64, *d as u64)
                    })
                    .collect();
                let want_ops: Vec<(String, u64, u64, u64, u64)> = want["ops"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| {
                        (
                            o[0].as_str().unwrap().to_string(),
                            o[1].as_u64().unwrap(),
                            o[2].as_u64().unwrap(),
                            o[3].as_u64().unwrap(),
                            o[4].as_u64().unwrap(),
                        )
                    })
                    .collect();
                assert_eq!(ops, want_ops, "ops for {qo:?} / {co:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 300);
}

#[test]
fn echo_matches_python() {
    for case in fixtures()["echo"].as_array().unwrap() {
        let note = case["note"].as_str().unwrap();
        assert_eq!(
            echo_words(note),
            strs(&case["words"]),
            "echo_words({note:?})"
        );
        let r = EchoReference::new(note, case["n"].as_u64().unwrap() as usize);
        for m in case["measures"].as_array().unwrap() {
            let got = r.measure(m["text"].as_str().unwrap());
            match (&m["m"], got) {
                (Value::Null, None) => {}
                (want, Some(g)) if !want.is_null() => {
                    assert_eq!(g.run as u64, want["run"].as_u64().unwrap());
                    assert_eq!(g.words as u64, want["words"].as_u64().unwrap());
                    assert_eq!(g.n as u64, want["n"].as_u64().unwrap());
                    assert!((g.containment - want["containment"].as_f64().unwrap()).abs() < 1e-9);
                }
                (want, got) => panic!("echo measure differs: python {want}, rust {got:?}"),
            }
        }
    }
}

#[test]
fn markdown_and_note_windows_match_python() {
    let fx = fixtures();
    for case in fx["markdown"].as_array().unwrap() {
        assert_eq!(
            clean_markdown(case["md"].as_str().unwrap()),
            case["clean"].as_str().unwrap()
        );
    }
    let count =
        |t: &str| t.split_whitespace().count() + t.chars().filter(|c| ".,:;".contains(*c)).count();
    for case in fx["note_windows"].as_array().unwrap() {
        let got = note_windows(
            &count,
            case["md"].as_str().unwrap(),
            case["title"].as_str().unwrap(),
            case["window_tokens"].as_u64().unwrap() as usize,
            case["max_windows"].as_u64().unwrap() as usize,
            case["min_heb"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            got,
            strs(&case["windows"]),
            "note_windows {} / {}",
            case["window_tokens"],
            case["max_windows"]
        );
    }
}

#[test]
fn urls_headings_and_small_helpers_match_python() {
    let fx = fixtures();
    for case in fx["urls"].as_array().unwrap() {
        let got = parse_book_url(case["url"].as_str().unwrap());
        match (&case["parsed"], got) {
            (Value::Null, None) => {}
            (want, Some((src, b, l))) if !want.is_null() => {
                assert_eq!(src.code(), want[0].as_str().unwrap(), "{}", case["url"]);
                assert_eq!(b, want[1].as_i64().unwrap());
                assert_eq!(l, want[2].as_i64().unwrap());
            }
            (want, got) => panic!("{}: python {want}, rust {got:?}", case["url"]),
        }
    }
    for case in fx["headings"].as_array().unwrap() {
        let lines = strs(&case["lines"]);
        assert_eq!(
            heading_path(&lines, case["upto"].as_u64().unwrap() as usize),
            strs(&case["path"])
        );
    }
    for case in fx["trunc16"].as_array().unwrap() {
        assert_eq!(
            trunc16(
                case["s"].as_str().unwrap(),
                case["n"].as_u64().unwrap() as usize
            ),
            case["out"].as_str().unwrap()
        );
    }
    for case in fx["text_hash"].as_array().unwrap() {
        let (h, n) = text_hash(&strs(&case["lines"]));
        assert_eq!(h, case["hash"][0].as_str().unwrap());
        assert_eq!(n as u64, case["hash"][1].as_u64().unwrap());
    }
}

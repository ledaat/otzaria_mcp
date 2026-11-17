"""otzaria_personal_vectors.py's passage recipe held to Otzaria's own engine: line_text_fixtures.json
was produced by otzaria_search_engine 7.2.0 itself (normalize_text_for_indexing; the keys by
semantic_keys::column_values and the sidecar's Chunker::embedded_windows) on 3,000 synthetic lines
(random mixes of tags, entities, marks and presentation forms, and sentence shapes with footnote
markers, footnotes, paragraphs and headings) and on one small book. The same fixtures check otzaria-sources' personal_index.py."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import otzaria_personal_vectors as V


def main():
    fx = json.loads((HERE / "line_text_fixtures.json").read_text(encoding="utf-8"))
    bad = [c for c in fx["lines"] if V.line_text(c["raw"]) != c["text"]]
    for c in bad[:5]:
        print(f"line_text({c['raw']!r}) = {V.line_text(c['raw'])!r}, Otzaria: {c['text']!r}")
    book = fx["book"]
    want = {(i, 0): int(k) for i, k in enumerate(book["keys"]) if k != "0"}
    want.update({(i, w): int(k) for i, w, k in book["windows"]})
    ps = V.windowed_passages(book["lines"])
    got = {(i, w): int(k) for (i, _t, w), k in zip(ps, V.passage_keys([t for _i, t, _w in ps]))}
    print(f"{len(fx['lines']) - len(bad):,}/{len(fx['lines']):,} lines as Otzaria stores them; the book's "
          f"passage keys {'match' if got == want else 'DIFFER'}")
    sys.exit(1 if bad or got != want else 0)


if __name__ == "__main__":
    main()

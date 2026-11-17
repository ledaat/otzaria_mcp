"""End-to-end test of otzaria_personal_vectors.py in a synthetic Otzaria folder (a stand-in
embedder: the model is not here). Checked afterwards by otzaria-semantic-search's own reader."""
import hashlib, json, os, shutil, sqlite3, struct, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import otzaria_personal_vectors as V

def main():
    # otzaria-semantic-search's examples/check_set, when given: Otzaria's own reader checks the result
    CHECK = os.environ.get("OXV_CHECK_SET", "")
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "e2e").resolve(); shutil.rmtree(root, ignore_errors=True)
    otz = root / "otzaria"; vdir = otz / "vectors"; (vdir / "segments").mkdir(parents=True)
    (otz / "databases").mkdir(); (otz / "books").mkdir()
    rng = np.random.default_rng(3)
    IDENT = json.loads('{"identity_digest": "a31f76271e9979e11f72685b9ced3287a6ddb2013f3c666f090ea71802d692cc", "identity": {"text": {"line_text_version": 4, "key_version": 1}, "model": {"family_id": "otzaria/Meivin-Embed-QA491-native256", "tokenizer_checksum": "0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9", "embedding_dim": 256, "pooling": "in-graph", "max_tokens": 256, "query_max_tokens": 128, "embedding_text_version": 2, "normalization_version": 1, "chunking_identity": 320260894818558297, "query_packages": [{"checksum": "e90a67be02c077990c7bc2a6485f6872769ac2a71eb9779d25ceb1fcd97ac9b3", "quantization": "int8"}, {"checksum": "5f80c47d3310a21fcc0c2f541fbb5732ba254555ef45afc2712cab63c8a33059", "quantization": "fp32"}]}, "store": {"backend_id": "otzaria-oxv", "store_format_version": 3, "vector_precision": "rot192-i8i4"}}}')
    IDENT["identity"]["model"]["chunking_identity"] = V.recipe_identity()


    def fake_vec(texts):
        """Deterministic 'embedding' of a passage: a unit vector seeded by its key."""
        out = np.zeros((len(texts), 256), np.float32)
        for i, t in enumerate(texts):
            r = np.random.default_rng(V.passage_key(t) % (2**32))
            v = r.standard_normal(256).astype(np.float32); out[i] = v / np.linalg.norm(v)
        return out


    class FakeEmbedder:
        device, graph, batch_tokens = "CPU", "fake.onnx", 4096
        calls = 0
        def __init__(self, *a, **k): pass
        def embed(self, texts):
            FakeEmbedder.calls += len(texts); return fake_vec(texts)
        def check_batching(self, texts): pass


    V.Embedder = FakeEmbedder
    V.live_vectors = lambda live, keys: {}  # the synthetic codec is too coarse for the model check

    # ---- books: words from a small vocabulary; one long line (windows), short lines (context), headings
    words = ["אמר", "רבא", "הלכה", "כרבי", "יהודה", "בשבת", "מותר", "אסור", "לטלטל", "כלי", "שמלאכתו", "לאיסור",
             "דתניא", "ומשום", "שנאמר", "וכתב", "הרמב\"ם", "בפירוש", "המשנה", "דברים", "אלו"]
    def line(n): return " ".join(rng.choice(words, n))
    def book(nlines, long_at=None):
        out = ["<h1>כותרת</h1>"]
        for i in range(nlines):
            if i == long_at: out.append(line(400))
            elif i % 9 == 4: out.append("סימן " + str(i))          # short: takes context
            else: out.append(line(int(rng.integers(6, 40))))
        return out
    lib_book = book(300)                      # a library book: its passages are in the base set
    books_txt = {
        2500: book(400, long_at=17),
        2501: book(250),
        2502: lib_book[:120] + book(80),     # half of it is the library's text
        2503: book(400, long_at=17)[:1] + book(30),
    }
    books_txt[2504] = list(books_txt[2501])   # an identical copy: one vector, two records
    folder = root / "personal"; folder.mkdir()
    con = sqlite3.connect(otz / "databases" / "user_books.db")
    con.execute("create table book (id integer primary key, title text, filePath text, fileType text, categoryId int)")
    con.execute("create table category (id integer primary key, parentId int, title text)")
    for bid, ls in books_txt.items():
        p = folder / f"ספר {bid}.txt"; p.write_text("\n".join(ls), encoding="utf-8")
        con.execute("insert into book values (?,?,?,?,1)", (bid, f"ספר {bid}", str(p), "txt"))
    con.execute("insert into book values (9,'pdf',?,'pdf',1)", (str(folder / "x.pdf"),))
    con.commit(); con.close()
    # ---- an attached database (seforim.db layout, plain line text)
    dbp = root / "unofficial.db"
    con = sqlite3.connect(dbp)
    con.execute("create table book (id integer primary key, title text, categoryId int)")
    con.execute("create table category (id integer primary key, parentId int, title text)")
    con.execute("create table line (id integer primary key, bookId int, lineIndex int, content text)")
    db_books = {7: book(200, long_at=50), 12: lib_book[100:220]}
    for bid, ls in db_books.items():
        con.execute("insert into book values (?,?,1)", (bid, f"מסד {bid}"))
        con.executemany("insert into line (bookId, lineIndex, content) values (?,?,?)", [(bid, i, t) for i, t in enumerate(ls)])
    con.commit(); con.close()
    # ---- Otzaria's settings (a Hive box with one string frame)
    val = json.dumps([{"path": str(dbp), "slug": "unofficial-books", "displayName": "מאגר", "status": "ok"},
                      {"path": str(dbp), "slug": "hidden-one", "hidden": True}])
    key = b"key-attached-libraries"; vb = val.encode()
    frame = bytes([1, len(key)]) + key + bytes([4]) + struct.pack("<I", len(vb)) + vb
    (otz / "app_preferences.hive").write_bytes(struct.pack("<I", len(frame) + 8) + frame + b"\0\0\0\0")

    # ---- the base set: the library book's passages (real codec layout, random rotation)
    dim, head, tail = 256, 128, 64
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    params = struct.pack("<5I", 1, dim, 192, head, tail) + np.linspace(0.3, 0.1, head).astype("<f4").tobytes() \
        + np.full(tail, -0.25, "<f4").tobytes() + np.full(tail, 0.25, "<f4").tobytes() + q[:192].astype("<f4").tobytes()
    rot = V.Rot(params)
    ps = V.windowed_passages(lib_book)
    bkeys = V.passage_keys([t for _i, t, _w in ps]); bl = np.array([i for i, _t, _w in ps])
    u, first = np.unique(bkeys, return_index=True); first.sort()
    bkeys, bl, btexts = bkeys[first], bl[first], [ps[i][1] for i in first]
    nb = len(bkeys)
    books = np.zeros(1, V.BOOK_DT); books["name_len"] = 4; books["slot_count"] = nb
    occ = np.zeros(nb, V.OCC_DT); occ["key"] = bkeys; occ["anchor"] = bkeys; occ["hint"] = bl
    occ = occ[np.lexsort((occ["hint"], occ["key"].astype(np.uint64)))]
    def fill(vec, scales):
        enc, s = rot.encode(fake_vec(btexts)); vec[:] = enc; scales[:] = s
    w = V.write_segment(vdir / "segments" / "t.oxv", kind_flag=V.FLAG_BASE, identity_digest=bytes.fromhex(IDENT["identity_digest"]),
                        params=params, from_version=0, to_version=38, tag="v38-test", dim=dim, books=books, names=b"id:1",
                        keys=bkeys.astype(">u8"), hints=bl.astype("<u4"), windows=np.zeros(nb, np.uint8),
                        occurrences=occ.tobytes(), book_hashes=np.zeros(1, "<u8"), fill_vectors=fill,
                        segment_id=hashlib.sha256(b"base").digest()[:16])
    os.replace(vdir / "segments" / "t.oxv", vdir / "segments" / f"{w['id']}.oxv")
    g = vdir / "gen-000001"; g.mkdir()
    delb = V.DEL_MAGIC + struct.pack("<QQ", nb, 0) + b"\0" * (8 * (-(-nb // 64))); delb += struct.pack("<I", V.crc(delb))
    (g / f"{w['id']}.del").write_bytes(delb)
    prov = {"passage_package": {"checksum": "x", "quantization": "fp32"}, "worker": {"backend": "test", "device": "test"}}
    doc = {"format": "otzaria-vector-set", "format_version": 1, "generation": 1, "identity": IDENT["identity"],
           "identity_digest": IDENT["identity_digest"], "codec_params_sha256": hashlib.sha256(params).hexdigest(),
           "library_version": 38, "library_release_tag": "v38-test",
           "segments": [{"id": w["id"], "file": f"segments/{w['id']}.oxv", "kind": "base", "from": 0, "to": 38,
                         "sha256": w["sha256"], "size": w["size"], "slots": nb, "provenance": prov,
                         "del": {"file": f"{w['id']}.del", "crc32": V.crc(delb[:-4]), "count": 0}}],
           "stats": {"slots": nb, "slots_dead": 0, "foreign": 0, "foreign_unresolved": 0, "bytes": w["size"]},
           "created_at": "2026-11-16T00:00:00Z"}
    raw = json.dumps(doc, indent=2).encode(); (g / "set.json").write_bytes(raw)
    (vdir / "CURRENT").write_text(json.dumps({"generation": 1, "set": "gen-000001/set.json",
                                              "set_sha256": hashlib.sha256(raw).hexdigest()}, separators=(",", ":")))
    # ---- a model folder (only model.json and tokenizer.json are read before the embedder)
    md = otz / "books" / "meivin-test-onnx"; md.mkdir()
    (md / "tokenizer.json").write_text("{}")
    (md / "model.json").write_text(json.dumps(dict(IDENT["identity"]["model"], tokenizer_checksum=hashlib.sha256(b"{}").hexdigest())))


    def run(*extra):
        sys.argv = ["x", "--otzaria-dir", str(otz), "--cache-dir", str(root / "cache"), "--workers", "2", *extra]
        try:
            V.main()
        except SystemExit as e:
            return e.code
        return 0


    print("== check"); assert run("--check") == 0
    print("== interrupted run (group of 100, stop after the first embedding)")
    real_add = V.Cache.add
    def stop_add(self, k, c, s):
        real_add(self, k, c, s); V.CANCEL.set()
    V.Cache.add = stop_add
    print("exit", run("--group", "100"))
    V.Cache.add = real_add; V.CANCEL.clear()
    n1 = FakeEmbedder.calls
    print("== run")
    rc = run("--group", "100"); n2 = FakeEmbedder.calls - n1
    print("exit", rc, "embedded:", n1, "+", n2)
    assert rc == 0 and n1 > 0 and n1 + n2 == 976, (rc, n1, n2)   # resumed: nothing embedded twice
    print("== run again (nothing new)"); FakeEmbedder.calls = 0; run(); print("embedded:", FakeEmbedder.calls)
    assert FakeEmbedder.calls == 0
    cur = json.loads((vdir / "CURRENT").read_text())
    assert cur["generation"] == 2, cur
    st = json.loads((vdir / cur["set"]).read_text())
    seg = st["segments"][-1]
    assert seg["slots"] == 976 and (vdir / seg["file"]).stat().st_size == seg["size"], seg
    assert hashlib.sha256((vdir / seg["file"]).read_bytes()).hexdigest() == seg["sha256"]
    assert V.self_test(vdir / seg["file"])

    # ---- Otzaria's reader: open, scrub, and scan for passages of each kind
    def q(texts): return fake_vec(texts).astype("<f4")
    def ptext(ls, li, w=0):
        return next(t for i, t, ww in V.windowed_passages(ls) if i == li and ww == w)
    queries = [ptext(books_txt[2500], 1), ptext(books_txt[2500], 18, 2), ptext(books_txt[2500], 14),   # line, window, short line
               ptext(books_txt[2501], 5), ptext(db_books[7], 51, 1), ptext(lib_book, 30)]
    (root / "q.f32").write_bytes(q(queries).tobytes())
    if CHECK:
        out = subprocess.run([CHECK, str(vdir), str(root / "q.f32")], capture_output=True, text=True)
        print(out.stdout[-2500:], out.stderr[-1500:])
    print("== remove"); assert run("--remove") == 0
    cur = json.loads((vdir / "CURRENT").read_text()); assert cur["generation"] == 3
    assert len(json.loads((vdir / cur["set"]).read_text())["segments"]) == 1
    print("ALL OK")
    if CHECK:
        out = subprocess.run([CHECK, str(vdir)], capture_output=True, text=True); print(out.stdout[:400], out.stderr[-500:])


def hive_frame(key, value):
    k, v = key.encode(), value.encode()
    f = bytes([1, len(k)]) + k + bytes([4]) + struct.pack("<I", len(v)) + v
    return struct.pack("<I", len(f) + 8) + f + b"\0\0\0\0"


def layouts(root):
    """Otzaria's library elsewhere than its settings, as in an installation for all users
    (C:\\ProgramData\\otzaria): found through the library-path setting, through %ProgramData%,
    and through a folder the user names."""
    otz = root / "otzaria"
    pd = root / "programdata" / "otzaria"
    shutil.rmtree(root / "programdata", ignore_errors=True)
    pd.mkdir(parents=True)
    for d in ("books", "vectors", "databases"):
        shutil.move(str(otz / d), str(pd / d))
    hive = (otz / "app_preferences.hive").read_bytes()

    def check(label, *extra, env=None):
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            sys.argv = ["x", "--check", "--cache-dir", str(root / "cache"), "--workers", "2", *extra]
            events = []
            V.HOOKS["event"] = lambda kind, **d: events.append((kind, d))
            try:
                V.main()
                rc = 0
            except SystemExit as e:
                rc = e.code
        finally:
            os.environ.clear(); os.environ.update(old)
            V.HOOKS["event"] = None
        loc = next((d for k, d in events if k == "locations"), {})
        summ = next((d for k, d in events if k == "summary"), {})
        print(f"== {label}: exit {rc}, vectors {loc.get('vectors')}, books {summ.get('books')}")
        assert rc == 0 and loc.get("vectors") == str(pd / "vectors") and summ.get("books") == 7, (label, rc, loc, summ)

    # 1. the setting key-library-path names the library (an installation for all users)
    (otz / "app_preferences.hive").write_bytes(hive + hive_frame("key-library-path", str(pd / "books")))
    check("library-path setting", "--otzaria-dir", str(otz))
    # 2. no setting: %ProgramData%\\otzaria
    (otz / "app_preferences.hive").write_bytes(hive)
    check("ProgramData", "--otzaria-dir", str(otz), env={"ProgramData": str(root / "programdata")})
    # 3. no setting, nothing standard: the folder the user picks
    check("chosen folder", "--otzaria-dir", str(otz), "--library-dir", str(pd))
    # and the user picks the folder holding the settings too (a portable copy: otzaria_data)
    port = root / "portable"; shutil.rmtree(port, ignore_errors=True); (port / "otzaria_data").mkdir(parents=True)
    shutil.copytree(otz, port / "otzaria_data", dirs_exist_ok=True)
    (port / "otzaria_data" / "app_preferences.hive").write_bytes(hive + hive_frame("key-library-path", str(pd / "books")))
    sys.argv = ["x"]
    loc = V.Locations(root=port)
    assert loc.data == port / "otzaria_data" and loc.vectors == pd / "vectors", (loc.data, loc.vectors)
    print("LAYOUTS OK")


if __name__ == "__main__":
    where = Path(sys.argv[1] if len(sys.argv) > 1 else "e2e").resolve()
    main()
    layouts(where)

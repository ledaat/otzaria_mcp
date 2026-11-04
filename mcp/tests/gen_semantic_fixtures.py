#!/usr/bin/env python3
"""Fixtures for mcp/src/semantic/parity_tests.rs: the Python server's own helpers
(otzaria-sources' otzaria_semantic_server.py and personal_index.py) run on many inputs.

    python3 -I gen_semantic_fixtures.py <otzaria-sources folder> > ../src/semantic/parity_fixtures.json
"""

import json
import random
import sys
import types

sys.path.insert(0, sys.argv[1])
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("srv", sys.argv[1] + "/otzaria_semantic_server.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
P = S.PI

rng = random.Random(5785)

BASE = [
    "שמעון הצדיק היה משירי כנסת הגדולה. הוא היה אומר, על שלשה דברים העולם עומד, על התורה ועל העבודה ועל גמילות חסדים.",
    "הַמּוֹצִיא יַיִן, כְּדֵי מְזִיגַת הַכּוֹס. חָלָב, כְּדֵי גְמִיעָה. דְּבַשׁ, כְּדֵי לִתֵּן עַל הַכָּתִית. שֶׁמֶן, כְּדֵי לָסוּךְ אֵבֶר קָטָן.",
    "אמר רב הונא אמר רב אין כלי ניטל אלא לדבר הניטל בשבת, ורבי יוחנן אמר אפילו לדבר שאינו ניטל.",
    "ואע\"ג דאמרינן בעלמא דבר שאינו מתכוין מותר, הכא פסיק רישיה ולא ימות הוא ואסור לכו\"ע.",
    "ר\"ה אמר הלכה כר\"ש ור\"י אמר הלכה כר\"מ, וקי\"ל כר\"י לגבי ר\"ה.",
    "בָּרוּךְ אַתָּה ה' אֱלֹהֵינוּ מֶלֶךְ הָעוֹלָם, אֲשֶׁר קִדְּשָׁנוּ בְּמִצְוֹתָיו וְצִוָּנוּ עַל נְטִילַת יָדָיִם.",
    "כתב הרמב\"ם בפ\"ה מהל' שבת דהמדליק את הנר בשבת חייב, ועיין בכסף משנה שם שהקשה עליו מדברי הגמרא.",
    "<b>מתני'</b> אין עומדין להתפלל אלא מתוך כובד ראש. חסידים הראשונים היו שוהין שעה אחת ומתפללין.",
]
ABBR = {"אע\"ג": "אף על גב", "לכו\"ע": "לכולי עלמא", "קי\"ל": "קיימא לן", "ר\"ה": "רב הונא",
        "ר\"י": "רבי יוחנן", "ר\"ש": "רבי שמעון", "פ\"ה": "פרק ה", "מתני'": "מתניתין"}


def words(t):
    return t.split()


def perturb(ws):
    ws = list(ws)
    for _ in range(rng.randint(0, 3)):
        if not ws:
            break
        op = rng.choice(["del", "ins", "rep", "abbr", "expand"])
        i = rng.randrange(len(ws))
        if op == "del":
            ws.pop(i)
        elif op == "ins":
            ws.insert(i, rng.choice(["דהא", "וכו'", "נמי", "ע\"ש", "ז\"ל", "והנה"]))
        elif op == "rep":
            ws[i] = rng.choice(["אמר", "תניא", "הכי", "ולא", "שבת", "דבר"])
        elif op == "abbr":
            for k, v in ABBR.items():
                vs = v.split()
                for j in range(len(ws) - len(vs) + 1):
                    if [S.norm_word(x) for x in ws[j:j + len(vs)]] == [S.norm_word(x) for x in vs]:
                        ws[j:j + len(vs)] = [k]
                        break
        elif op == "expand":
            k = S.norm_word(ws[i])
            for a, v in ABBR.items():
                if S.norm_word(a) == k:
                    ws[i:i + 1] = v.split()
                    break
    return ws


def align_case(q, c):
    qw_orig = [w for w in q if S.norm_word(w)]
    qw = [S.norm_word(w) for w in qw_orig]
    co = [w for w in c if S.norm_word(w)]
    cw = [S.norm_word(w) for w in co]
    al = S.align_words(qw, cw, qw_orig, co)
    out = None
    if al is not None:
        ratio, a0, a1, ops = al
        out = {"ratio": ratio, "start": a0, "end": a1, "ops": [list(o) for o in ops]}
    return {"qo": qw_orig, "qw": qw, "co": co, "cw": cw, "result": out}


fx = {}

# window spans: lines of words up to ~3500 characters, spaces in odd places too
spans = []
for n in [0, 5, 100, 511, 512, 513, 600, 900, 1024, 1500, 2048, 2600, 3500]:
    for _ in range(3):
        t = ""
        while len(t) < n:
            t += rng.choice(["אבג", "דהוזח", "טיכלמנס", "עפצקרשת", "ב", "x" * rng.randint(1, 40)]) + (" " if rng.random() > 0.05 else "")
        t = t[:n]
        spans.append({"text": t, "spans": [list(s) for s in P.window_spans(t)],
                      "server_span_3": list(S.window_span(t, 3)) if t else None})
fx["window_spans"] = spans

samples = BASE + ["<h2>פרק <i>א</i></h2>", "a&amp;b &lt;c&gt; &quot;d&quot; &#1488;&#x5d1; &nbsp;x", "  רב   הונא\tאמר<br/>שלום  ",
                  "שִׁמְעוֹן׀ הַצַּדִּיק־הָיָה׃", "<span class=\"x\">ט</span><br>ק", "", "   ", "׆נ׀"]
fx["text"] = [{"s": s, "strip_html": S.strip_html(s), "line_text": P.line_text(s),
               "norm_words": [S.norm_word(w) for w in s.split()]} for s in samples]

al = []
for _ in range(400):
    base = words(rng.choice(BASE))
    a = rng.randrange(len(base))
    b = min(len(base), a + rng.randint(3, 14))
    quote = perturb(base[a:b])
    ctx = base[max(0, a - rng.randint(0, 8)):min(len(base), b + rng.randint(0, 8))]
    if rng.random() < 0.3:
        ctx = perturb(ctx) + words(rng.choice(BASE))[: rng.randint(0, 10)]
    al.append(align_case(quote, ctx))
# abbreviation-specific
for q, c in [("ואף על גב דאמרינן", "ואע\"ג דאמרינן בעלמא"), ("ר\"ה אמר הלכה", "רב הונא אמר הלכה כר\"ש"),
             ("קיימא לן כרבי יוחנן", "וקי\"ל כר\"י לגבי"), ("מתניתין אין עומדין", "מתני' אין עומדין להתפלל"),
             ("דרב הונא", "דר\"ה אמר"), ("א ב ג", "ד ה ו"), ("", "א"), ("שלום", "")]:
    al.append(align_case(q.split(), c.split()))
fx["align"] = al

echo_notes = [BASE[0] + " " + BASE[2], "ר\"ן כתב שאין כלי ניטל אלא לדבר הניטל", "", "x y"]
fx["echo"] = []
for note in echo_notes:
    for n in (2, 3, 4, 9):
        ref = S.EchoReference(note, n)
        fx["echo"].append({"note": note, "n": n, "words": S.echo_words(note),
                           "measures": [{"text": t, "m": ref.measure(t)} for t in BASE + ["שלום", "רן כתב שאין כלי ניטל"]]})

MD = [
    "---\ntags: [a]\nmekorot: []\n---\n# כותרת\n\nאמר רב הונא אין כלי ניטל אלא לדבר הניטל בשבת. ורבי יוחנן אמר אפילו לדבר שאינו ניטל; וקי\"ל כוותיה.\n\n- [[שבת קכא|שבת]] ו-[[מוקצה]]\n> [!note] הערה\n> **חשוב** ==מאוד== `code`\n\n```py\nx=1\n```\n%%comment%% <!-- c --> ![[img.png]] [link](http://x.y) #tag otzaria://open/book/1?index=2 [^1]\n1. אחד\n2) שנים\n",
    "",
    "Just English text with no Hebrew at all, long enough to be a paragraph.\n\nשורה עברית קצרה",
    "\n".join(BASE),
    "\n\n".join(BASE * 3),
]
fx["markdown"] = [{"md": m, "clean": S.clean_markdown(m)} for m in MD]


class Tok:
    @staticmethod
    def count_tokens(t):
        return len(t.split()) + sum(1 for c in t if c in ".,:;")


fx["note_windows"] = []
for m in MD:
    for title in ("", "אין כלי ניטל אלא לדבר הניטל", "Title"):
        for wt, mw in ((160, 12), (20, 12), (12, 3), (8, 1), (30, 5)):
            fx["note_windows"].append({"md": m, "title": title, "window_tokens": wt, "max_windows": mw, "min_heb": 12,
                                       "windows": S.note_windows(Tok, m, title, wt, mw, 12)})

urls = ["otzaria://open/book/12?index=34", "x otzaria://open/book/5?source=user&index=7 y",
        "otzaria://open/book/9?source=db:otzarya-unofficial-books&index=1", "otzaria://open/book/9?index=",
        "otzaria://open/book/9?source=weird&index=1", "otzaria://open/book/9?source=official&index=3",
        "otzaria://open/book/9?index=4)", "https://x", "otzaria://open/book/9?source=db%3Afoo&index=2"]
fx["urls"] = [{"url": u, "parsed": list(S.parse_book_url(u)) if S.parse_book_url(u) else None} for u in urls]

lines = ["<h1>ספר</h1>", "a", "<h2>פרק א</h2>", "b", "<H3 class=x>סימן <b>ב</b></h3>", "c", "<h2>פרק ב</h2>", "d", "<h5>x"]
fx["headings"] = [{"lines": lines, "upto": i, "path": P.heading_path(lines, i)} for i in range(len(lines))]

fx["trunc16"] = [{"s": s, "n": n, "out": S.trunc16(s, n)} for s in ("abc", "אבג", "a😀b", "😀😀") for n in (0, 1, 2, 3, 5)]
fx["text_hash"] = [{"lines": ls, "hash": list(P.text_hash(ls))} for ls in (["a", "b"], ["a\r", "b\r", "", "  "], ["שלום"], [])]

json.dump(fx, sys.stdout, ensure_ascii=False, indent=0)

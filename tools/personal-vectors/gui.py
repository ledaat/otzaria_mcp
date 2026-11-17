"""OtzariaPersonalVectors.exe: a window over otzaria_personal_vectors.py for users who do not
use a command line. On start it checks what is missing (nothing is changed); "התחל" embeds and
publishes; "הסר" takes the personal segment out again."""

import argparse
import json
import multiprocessing
import os
import queue
import sys
import threading
import traceback


def smoke(path):
    """CI: the frozen program's parts work (imports, the GPU providers, worker processes)."""
    import numpy as np
    import onnxruntime as ort
    import tokenizers
    import zstandard
    from concurrent.futures import ProcessPoolExecutor
    import otzaria_personal_vectors as V
    with ProcessPoolExecutor(2) as ex:
        keys = list(ex.map(V.passage_key, ["[PASSAGE] אמר רבא", "[PASSAGE] תניא"]))
    out = {"numpy": np.__version__, "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
           "tokenizers": tokenizers.__version__, "zstandard": zstandard.__version__, "keys": keys,
           "recipe": V.recipe_identity()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


RLM = "\u200f"
LRE, PDF = "\u202a", "\u202c"   # a left-to-right stretch (a path) inside right-to-left text


def R(text):
    """Right-to-left text for Tk and Windows message boxes, which lay a line out left to right:
    a right-to-left mark at both ends of every line keeps punctuation and numbers at the edges
    in their places (a sentence's final period on the left)."""
    return "\n".join(RLM + line + RLM if line else line for line in str(text).split("\n"))


def hours(n_passages, rate):
    secs = n_passages / max(rate, 1e-6)
    if secs < 90:
        return "פחות משתי דקות"
    if secs < 3600:
        return f"כ-{secs / 60:.0f} דקות"
    return f"כ-{secs / 3600:.1f} שעות"


HEBREW_ERRORS = [
    ("no vector set", "לא נמצאו הווקטורים של החיפוש לפי משמעות של אוצריא.\n\n"
                      "אם הספרייה של אוצריא נמצאת בתיקייה אחרת, לחצו על 'בחר תיקיית אוצריא' ובחרו את התיקייה "
                      "שבתוכה נמצאת התיקייה books. למשל:\n"
                      + "\u202a" + "C:\\ProgramData\\otzaria" + "\u202c" + "\n\n"
                      "אחרת, פתחו את אוצריא, הפעילו פעם אחת חיפוש לפי משמעות, חכו שההורדה תסתיים ונסו שוב."),
    ("no model folder", "לא נמצא מודל החיפוש לפי משמעות בתיקיית הספרים של אוצריא. פתחו את אוצריא, "
                        "הפעילו פעם אחת חיפוש לפי משמעות וחכו שההורדה תסתיים, ואז נסו שוב."),
    ("is locked", "אוצריא מעדכנת כעת את הווקטורים. חכו שהעדכון יסתיים ונסו שוב."),
    ("changed during the run", "אוצריא עדכנה את הווקטורים בזמן הריצה. הפעילו שוב (מה שכבר חושב נשמר)."),
    ("newer version of the script", "אוצריא שינתה את שיטת החלוקה לקטעים. נדרשת גרסה חדשה של הכלי."),
    ("is not stored as segment format", "אוצריא שינתה את פורמט הווקטורים. נדרשת גרסה חדשה של הכלי."),
    ("not Otzaria's", "המודל שנמצא אינו נותן את הווקטורים של אוצריא. לא נכתב דבר."),
]


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import otzaria_personal_vectors as V

    for name in ("askokcancel", "askyesno", "showinfo", "showerror"):
        orig = getattr(messagebox, name)
        setattr(messagebox, name, lambda title, msg, orig=orig: orig(R(title), R(msg)))

    root = tk.Tk()
    root.title("חיפוש לפי משמעות בספרים אישיים – אוצריא")
    root.geometry("720x520")
    root.minsize(560, 420)
    font = ("Segoe UI", 11)
    big = ("Segoe UI", 13, "bold")
    style = ttk.Style()
    style.configure("TButton", font=font, padding=6)

    prefs_path = V.default_cache_dir() / "settings.json"

    def load_prefs():
        try:
            return json.loads(prefs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_prefs(d):
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        prefs_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    prefs = load_prefs()
    ui = queue.Queue()            # (function, args) to run on the window's thread
    state = {"check": None, "busy": False, "total": 0, "details": False}

    def on_ui(fn, *a):
        ui.put((fn, a))

    def pump():
        try:
            while True:
                fn, a = ui.get_nowait()
                fn(*a)
        except queue.Empty:
            pass
        root.after(100, pump)

    head = tk.Label(root, text=R("חיפוש לפי משמעות בספרים האישיים ובמסדי הספרים המצורפים"), font=big, anchor="e",
                    justify="right")
    head.pack(fill="x", padx=16, pady=(14, 6))
    status = tk.Label(root, text=R("בודק מה יש במחשב…"), font=font, anchor="ne", justify="right", wraplength=680)
    status.pack(fill="x", padx=16, pady=6)
    bar = ttk.Progressbar(root, mode="indeterminate", length=680)
    bar.pack(fill="x", padx=16, pady=(6, 2))
    bar.start(12)
    progress = tk.Label(root, text="", font=font, anchor="e", justify="right")
    progress.pack(fill="x", padx=16)

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=16, pady=10)
    start_b = ttk.Button(buttons, text="התחל", state="disabled")
    stop_b = ttk.Button(buttons, text="עצור", state="disabled")
    remove_b = ttk.Button(buttons, text="הסר מאוצריא", state="disabled")
    details_b = ttk.Button(buttons, text="פרטים")
    choose_b = ttk.Button(buttons, text=R("בחר תיקיית אוצריא…"))
    for b in (start_b, stop_b, remove_b):
        b.pack(side="right", padx=4)
    details_b.pack(side="left", padx=4)
    choose_b.pack(side="left", padx=4)
    where = tk.Label(root, text="", font=("Segoe UI", 9), fg="#666", anchor="e", justify="right")
    where.pack(fill="x", padx=16)

    log_frame = tk.Frame(root)
    log_text = tk.Text(log_frame, height=12, font=("Consolas", 9), wrap="word", state="disabled")
    scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    log_text.pack(side="left", fill="both", expand=True)

    def toggle_details():
        state["details"] = not state["details"]
        if state["details"]:
            log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        else:
            log_frame.pack_forget()

    details_b.configure(command=toggle_details)

    def add_log(msg):
        log_text.configure(state="normal")
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")

    def set_status(text):
        status.configure(text=R(text))

    def set_prog(text):
        progress.configure(text=R(text) if text else "")

    def set_progress(done, rate):
        total = max(state["total"], done)
        if str(bar["mode"]) != "determinate":
            bar.stop()
            bar.configure(mode="determinate", maximum=max(total, 1))
        bar.configure(maximum=max(total, 1), value=done)
        left = max(total - done, 0)
        set_prog(f"חושבו {done:,} מתוך {total:,} קטעים ({rate:,.0f} בשנייה) – נותרו {hours(left, rate)}")

    def busy(on, indeterminate=True):
        state["busy"] = on
        start_b.configure(state="disabled")
        remove_b.configure(state="disabled")
        choose_b.configure(state="disabled" if on else "normal")
        stop_b.configure(state="normal" if on else "disabled")
        if on and indeterminate:
            bar.configure(mode="indeterminate", value=0)
            bar.start(12)
        if not on:
            bar.stop()

    # ---- hooks (called from the worker thread)
    def on_event(kind, **d):
        if kind == "locations":
            on_ui(where.configure, {"text": R(f"תיקיית אוצריא: {LRE}{d['library']}{PDF}")})
        elif kind == "books":
            on_ui(set_status, f"נמצאו {d['personal']:,} ספרים אישיים (txt) ו-{d['attached']:,} ספרים במסדים מצורפים. "
                              f"קורא את הספרים…")
        elif kind == "reading":
            on_ui(set_prog, f"נקראו {d['done']:,} מתוך {d['total']:,} ספרים")
        elif kind == "model":
            dev = "בכרטיס הגרפי" if d["device"] != "CPU" else "במעבד"
            on_ui(set_status, f"מחשב את הקטעים החסרים {dev}. אפשר לעצור בכל רגע ולהמשיך אחר כך: מה שחושב נשמר.")
        elif kind == "published":
            on_ui(set_status, "הסתיים. פתחו את אוצריא: החיפוש לפי משמעות מוצא עכשיו גם את הספרים האלה.")
        elif kind in ("nothing", "unchanged"):
            on_ui(set_status, "אין מה להוסיף: כל הקטעים של הספרים האלה כבר נמצאים בחיפוש של אוצריא.")
        elif kind == "removed":
            on_ui(set_status, "הוסר. החיפוש לפי משמעות באוצריא חזר לספריה בלבד (מה שחושב נשמר להבא).")
        elif kind == "no_segment":
            on_ui(set_status, "אין תוספת אישית בחיפוש של אוצריא; אין מה להסיר.")

    def confirm_closed():
        answer, done = [False], threading.Event()

        def ask():
            answer[0] = messagebox.askokcancel(
                "אוצריא פתוחה", "כדי להוסיף את הספרים לחיפוש צריך לסגור את אוצריא.\n\nסגרו את אוצריא ולחצו 'אישור'.")
            done.set()
        on_ui(ask)
        done.wait()
        return answer[0]

    V.HOOKS.update(log=lambda m: on_ui(add_log, m), event=on_event,
                   progress=lambda n, r: on_ui(set_progress, n, r), confirm_closed=confirm_closed)

    def args_for(check=False, remove=False):
        return argparse.Namespace(
            check=check, remove=remove, gpu="auto", otzaria_dir="", library_dir=prefs.get("library_dir", ""),
            vectors_dir="",
            model_dir="", cache_dir=str(V.default_cache_dir()), threads=0,
            workers=max(1, min(4, (os.cpu_count() or 2) - 1)), group=20000, force=False, self_test=None)

    def worker(kind):
        V.CANCEL.clear()
        try:
            if kind == "check":
                rep = V.run(args_for(check=True))
                on_ui(after_check, rep)
            elif kind == "start":
                rep = V.run(args_for())
                on_ui(after_run, rep)
            else:
                V.run(args_for(remove=True))
                on_ui(after_remove)
        except V.Cancelled as e:
            n = e.args[0] if e.args else 0
            on_ui(after_stop, n)
        except Exception as e:
            text = str(e)
            heb = next((h for k, h in HEBREW_ERRORS if k in text), None)
            on_ui(add_log, traceback.format_exc())
            on_ui(after_error, heb, text)

    def launch(kind):
        busy(True)
        threading.Thread(target=worker, args=(kind,), daemon=True).start()

    def after_check(rep):
        busy(False)
        state["check"] = rep
        bar.configure(mode="determinate", value=0)
        set_prog("")
        if rep is None:
            set_status("לא נמצאו ספרים אישיים (txt) או מסדי ספרים מצורפים באוצריא.")
            return
        state["total"] = rep["to_embed"]
        if not rep["to_publish"]:
            set_status(f"נבדקו {rep['books']:,} ספרים: כל הקטעים שלהם כבר נמצאים בחיפוש לפי משמעות של אוצריא. "
                       f"אין צורך לעשות דבר.")
        else:
            msg = (f"נבדקו {rep['books']:,} ספרים ({rep['passages']:,} קטעים). "
                   f"{rep['to_publish']:,} קטעים עדיין אינם בחיפוש לפי משמעות של אוצריא.")
            if rep["to_embed"]:
                msg += (f"\nצריך לחשב {rep['to_embed']:,} מהם: במעבד רגיל {hours(rep['to_embed'], 50)}, "
                        f"בכרטיס גרפי מהר יותר. אפשר לעצור ולהמשיך אחר כך.")
            msg += "\n\nלחצו 'התחל'. בסוף תתבקשו לסגור את אוצריא."
            set_status(msg)
            start_b.configure(state="normal")
        remove_b.configure(state="normal")

    def after_run(rep):
        busy(False)
        bar.configure(mode="determinate", maximum=1, value=1)
        remove_b.configure(state="normal")
        if rep and rep.get("published"):
            set_prog("")
            messagebox.showinfo("הסתיים", "הספרים נוספו לחיפוש לפי משמעות.\n\nפתחו את אוצריא ונסו.")

    def after_remove():
        busy(False)
        remove_b.configure(state="normal")
        start_b.configure(state="normal" if (state["check"] or {}).get("to_publish") else "disabled")

    def after_stop(n):
        busy(False)
        state["total"] = max(state["total"] - n, 0)
        set_status(f"נעצר. {n:,} קטעים שחושבו נשמרו; לחצו 'התחל' כדי להמשיך מאותה נקודה." if n else "נעצר.")
        start_b.configure(state="normal")
        remove_b.configure(state="normal")

    def after_error(heb, text):
        busy(False)
        bar.configure(mode="determinate", value=0)
        set_status(heb or "אירעה שגיאה. לחצו 'פרטים' כדי לראות אותה.")
        if not heb:
            messagebox.showerror("שגיאה", text)
        start_b.configure(state="normal" if (state["check"] or {}).get("to_publish") else "disabled")
        remove_b.configure(state="normal")

    def on_choose():
        folder = filedialog.askdirectory(
            title="בחרו את התיקייה של אוצריא (זו שיש בה books ו-vectors)",
            initialdir=prefs.get("library_dir") or os.environ.get("ProgramData") or "C:\\")
        if not folder:
            return
        folder = os.path.normpath(folder)
        loc = V.Locations(root=folder)
        if not (loc.vectors / "CURRENT").exists():
            messagebox.showerror("לא נמצא", f"בתיקייה הזו אין את הווקטורים של אוצריא:\n{LRE}{folder}{PDF}\n\n"
                                             f"בחרו את התיקייה שבתוכה נמצאות התיקיות books ו-vectors.")
            return
        prefs["library_dir"] = folder
        save_prefs(prefs)
        set_status("בודק מה יש במחשב…")
        launch("check")

    def on_start():
        set_status("מתחיל…")
        set_prog("")
        launch("start")

    def on_stop():
        V.CANCEL.set()
        stop_b.configure(state="disabled")
        set_status("עוצר… (מה שחושב נשמר)")

    def on_remove():
        if messagebox.askyesno("הסרה", "להסיר את הספרים האישיים מהחיפוש לפי משמעות של אוצריא?\n"
                                       "(מה שחושב נשמר, ואפשר להוסיף שוב בלי לחשב מחדש)"):
            launch("remove")

    def on_close():
        if state["busy"] and not messagebox.askyesno("יציאה", "הפעולה עדיין רצה. לעצור ולצאת? (מה שחושב נשמר)"):
            return
        V.CANCEL.set()
        root.destroy()
        os._exit(0)

    start_b.configure(command=on_start)
    stop_b.configure(command=on_stop)
    remove_b.configure(command=on_remove)
    choose_b.configure(command=on_choose)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, pump)
    root.after(300, lambda: launch("check"))
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke":
        smoke(sys.argv[2])
        sys.exit(0)
    main()

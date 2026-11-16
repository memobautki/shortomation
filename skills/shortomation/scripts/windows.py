"""Sätze mit gemessenen Zeiten und Fenster zum Bewerten (Schritt 2).

    python -I windows.py --dir <ordner> [--win 75] [--step 35]

Liest <ordner>/words.json (aus transcribe.py) und schreibt <ordner>/sentences.json und <ordner>/transcript.md mit
Satz-IDs, Zeiten und Fenstern, die der Agent nach references/rubric.md bewertet. Satzanfang und -ende sind die
gemessenen Zeiten des ersten und letzten Wortes, nicht geschätzt.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import StepError, mmss, norm_word, read_json, setup_console, write_json

ABBR = {"z.b", "bzw", "ca", "dr", "nr", "st", "u.a", "inkl", "evtl", "vgl", "mr", "mrs", "vs", "etc"}
MAX_WORDS = 40      # ohne Satzzeichen spätestens hier trennen
SPLIT_PAUSE = 1.0   # ... oder an einer Sprechpause ab dieser Länge (s)


def ends_sentence(word: str) -> bool:
    """Satzende bei . ! ? … ; nicht nach Ordinalzahlen („18.“) und üblichen Abkürzungen."""
    core = word.rstrip("\"'»«“”’)")
    if not core or core[-1] not in ".!?…":
        return False
    if core[-1] == ".":
        stem = core[:-1].lower()
        if re.fullmatch(r"\d{1,2}", stem) or stem in ABBR:
            return False
    return True


def merge_punctuation(words: list[dict]) -> list[dict]:
    """Tokens ohne Buchstaben und Ziffern („–“, „…“) an das Wort davor hängen (am Anfang: an das danach)."""
    out: list[dict] = []
    pending = ""
    for w in words:
        if not norm_word(w["w"]):
            if out:
                out[-1] = {**out[-1], "w": f"{out[-1]['w']} {w['w']}", "t1": max(out[-1]["t1"], w["t1"])}
            else:
                pending = f"{pending}{w['w']} "
            continue
        out.append({**w, "w": pending + w["w"]} if pending else dict(w))
        pending = ""
    return out


def build_sentences(words: list[dict], split_on_pause: bool = True) -> list[dict]:
    """Wörter -> Sätze. Zeiten = erstes und letztes Wort des Satzes.

    split_on_pause trennt zusätzlich an langen Sprechpausen; nur nötig, wenn die Satzzeichen aus der Erkennung
    stammen. Mit dem Wortlaut einer SRT bleiben die Sätze so, wie sie dort stehen.
    """
    words = merge_punctuation(words)
    sents: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            sents.append({"id": f"S{len(sents):03d}", "t0": round(cur[0]["t0"], 2), "t1": round(cur[-1]["t1"], 2),
                          "text": " ".join(w["w"] for w in cur), "n_words": len(cur),
                          "est_words": sum(1 for w in cur if w.get("est")),
                          "tokens": [[w["w"], round(w["t0"], 3), round(w["t1"], 3)] for w in cur]})
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        long_pause = split_on_pause and nxt is not None and nxt["t0"] - w["t1"] >= SPLIT_PAUSE
        if ends_sentence(w["w"]) or len(cur) >= MAX_WORDS or long_pause:
            flush()
    flush()
    return sents


def build_windows(sents: list[dict], win: float, step: float) -> list[dict]:
    """Überlappende Fenster von etwa win Sekunden, Start alle step Sekunden."""
    res, i = [], 0
    while i < len(sents):
        j = i
        while j + 1 < len(sents) and sents[j + 1]["t1"] - sents[i]["t0"] <= win:
            j += 1
        res.append({"id": f"W{len(res):02d}", "s0": sents[i]["id"], "s1": sents[j]["id"],
                    "t0": sents[i]["t0"], "t1": sents[j]["t1"]})
        if j == len(sents) - 1:
            break
        nxt = i
        while nxt < len(sents) and sents[nxt]["t0"] < sents[i]["t0"] + step:
            nxt += 1
        i = max(i + 1, min(nxt, j + 1))  # nie über einen Satz springen, der in keinem Fenster liegt
    return res


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Arbeitsordner mit words.json")
    ap.add_argument("--win", type=float, default=75.0, help="Fensterlänge in s (Standard 75)")
    ap.add_argument("--step", type=float, default=35.0, help="Versatz zwischen Fenstern in s (Standard 35)")
    args = ap.parse_args()
    if args.win <= 0 or args.step <= 0:
        raise StepError("--win und --step müssen größer als 0 sein.")
    d = Path(args.dir)
    wj = read_json(d / "words.json")
    sents = build_sentences(wj["words"], split_on_pause=wj.get("source") != "srt+whisper")
    if not sents:
        raise StepError("words.json enthält keine Wörter.")
    wins = build_windows(sents, args.win, args.step)
    write_json(d / "sentences.json", {"sentences": sents, "windows": wins})
    lines = ["# Transkript in Sätzen", "", "Format: `ID [Start–Ende] Text`. Fenster unten.", ""]
    lines += [f"{s['id']} [{mmss(s['t0'])}–{mmss(s['t1'])}] {s['text']}" for s in sents]
    lines += ["", "# Fenster (jedes bewerten, references/rubric.md)", ""]
    lines += [f"{w['id']}: {w['s0']}–{w['s1']} [{mmss(w['t0'])}–{mmss(w['t1'])}]" for w in wins]
    (d / "transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(sents)} Sätze, {len(wins)} Fenster -> {d / 'transcript.md'}")


if __name__ == "__main__":
    main()

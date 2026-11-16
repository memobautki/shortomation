"""Prüft fertige Shorts, bevor sie jemand zu sehen bekommt (Schritt 7).

    python -I check.py --dir <ordner> [--ids k01,k04] [--words] [--model small]

Je Short:
  Bild      1080x1920, Ton vorhanden, Länge wie geplant
  Lautheit  integriert -14 LUFS (±1,5)
  Schnitte  Direkt an jeder Stückgrenze darf keine Sprache klingen, sonst ist ein Wort angeschnitten (genau der Fehler
            „letztes Wort fehlt“). Pegel plus Sprach-Erkennung (Silero-VAD aus faster-whisper), damit
            Hintergrundmusik nicht als Sprache zählt. Unklare leise Laute: Ergebnis „prüfen“ (anhören).
  Wörter    (mit --words) Der Short wird neu erkannt. Je Stück müssen das erste und das letzte Wort der dort
            geplanten Sätze zu hören sein, und es dürfen keine zwei Wörter am Stück fehlen.
Schreibt <ordner>/CHECK.md. Exit-Code 1, wenn ein Short durchfällt.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import (FINAL_FADE, FIRST_FADE, PIECE_FADE, SR_LEVEL, VAD_FRAME, StepError, level_db, load_audio,
                    norm_word, probe, read_json, run, setup_console, speech_probability)
from transcribe import same_spoken

LUFS_TARGET, LUFS_TOL = -14.0, 1.5
DUR_TOL = 0.15
HOP = 80              # 10 ms bei 8 kHz: fein genug für die kurzen Ruhefenster an den Schnitten
OVER_FLOOR = 8.0      # dB über dem Grundpegel des Shorts gilt als Laut (erfasst auch Geflüstertes) -> anhören
SPEECH_NEAR = 8.0     # höchstens so weit unter dem Sprachpegel: eindeutig Sprache am Schnitt -> durchgefallen
SAFETY = 0.01         # Abstand zur gemessenen Lautgrenze (s)
EDGE = 0.08           # geprüft wird das Stück direkt am Schnitt: so lang (s)
VAD_HOP = VAD_FRAME / 16000       # Silero-VAD: 32 ms je Wert
VAD_SPEECH, VAD_MAYBE = 0.5, 0.2  # ab hier Sprache bzw. vielleicht Sprache (leise, geflüstert)
BED_MARGIN = 6.0                  # so weit über dem Musikbett gilt ein Laut als „anhören“
MIN_WINDOW = 0.02     # kürzere Ruhefenster lassen sich nicht sinnvoll prüfen (s)
EDGE_WORDS = 3        # so viele erkannte Wörter am Stückrand werden mit dem Randwort verglichen


def loudness(path: Path) -> float:
    res = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"])
    found = re.findall(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", res.stderr.decode("utf-8", "replace"))
    if not found:
        raise StepError(f"Lautheit von {path.name} nicht messbar.")
    return float(found[-1])


def piece_spans(clip: dict) -> list[tuple[float, float]]:
    """Stücke als (Anfang, Ende) in Sekunden des fertigen Shorts."""
    speed, t, out = float(clip.get("speed", 1.0)), 0.0, []
    for a, b in clip["cut"]:
        out.append((t, t + (b - a) / speed))
        t += (b - a) / speed
    return out


def require_metadata(clip: dict) -> None:
    """clips.scored.json muss zur aktuellen score.py passen, sonst wäre die Prüfung unvollständig."""
    n = len(clip.get("cut") or [])
    groups, air = clip.get("piece_sents"), clip.get("air")
    if (not n or not isinstance(groups, list) or len(groups) != n or any(not g for g in groups)
            or not isinstance(air, list) or len(air) != n):
        raise StepError(f"{clip.get('id')}: clips.scored.json ist unvollständig oder veraltet. "
                        "Bitte score.py und render.py neu laufen lassen.")


def edge_report(db: np.ndarray, hop: float, clip: dict, vad: np.ndarray | None = None) -> tuple[list[str], list[str]]:
    """Prüft das kurze Stück direkt am Schnitt (nach der Einblende bzw. vor der Ausblende). Gibt (Fehler, anhören).

    Fehler: Dort klingt Sprache (Pegel über Grundrauschen und Silero sagt Sprache; ohne Silero: Pegel fast so laut
    wie die Sprache). Anhören: leiser Laut, den Silero nicht eindeutig zuordnet, oder keine messbare Pause."""
    floor = float(np.percentile(db, 10))
    floor_thr = floor + OVER_FLOOR       # deutlich hörbar
    audible = floor + 3.0                # kaum über dem Grundpegel: reicht, wenn Silero dort Sprache hört
    strong = max(floor_thr, float(np.percentile(db, 90)) - SPEECH_NEAR)
    thr = floor_thr
    if vad is not None and len(vad):  # Musikbett: Pegel dort, wo Silero keine Sprache hört
        idx = np.minimum((np.arange(len(db)) * hop / VAD_HOP).astype(int), len(vad) - 1)
        bed = db[vad[idx] < VAD_MAYBE]
        if len(bed):
            thr = max(thr, float(np.median(bed)) + BED_MARGIN)
    speed = float(clip.get("speed", 1.0))
    spans = piece_spans(clip)
    air = clip["air"]
    problems, review = [], []

    def measure(t0: float, t1: float) -> tuple[float, float | None] | None:
        if t1 - t0 < MIN_WINDOW:
            return None
        seg = db[max(0, int(t0 / hop)):max(0, int(t1 / hop))]
        if not len(seg):
            return None
        prob = None
        if vad is not None:
            v = vad[max(0, int(t0 / VAD_HOP)):max(0, int(np.ceil(t1 / VAD_HOP)))]
            prob = float(v.max()) if len(v) else 0.0
        return float(seg.max()), prob

    for k, (o0, o1) in enumerate(spans):
        lead, tail = air[k][0] / speed, air[k][1] / speed
        fin = min(FIRST_FADE if k == 0 else PIECE_FADE, max(0.005, air[k][0] - 0.02)) / speed
        fout = min(FINAL_FADE if k == len(spans) - 1 else PIECE_FADE, max(0.005, air[k][1] - 0.02)) / speed
        windows = (("beginnt", o0, o0 + fin, min(o0 + fin + EDGE, o0 + lead - SAFETY)),
                   ("endet", o1, max(o1 - fout - EDGE, o1 - tail + SAFETY), o1 - fout))
        for where, at, t0, t1 in windows:
            found = measure(t0, t1)
            if found is None:
                review.append(f"Stück {k + 1} {where} bei {at:.2f} s ohne messbare Pause: anhören")
                continue
            level, prob = found
            if prob is None:  # nur Pegel: fast so laut wie Sprache = Fehler, sonst hörbar = anhören
                speech, unsure = level > strong, level > floor_thr
            else:  # Silero hört Sprache (auch unter Musik) = Fehler; leiser Laut über dem Musikbett = anhören
                speech, unsure = prob >= VAD_SPEECH and level > audible, level > thr
            if speech:
                detail = f"{level:.0f} dB" + (f", Sprache {prob:.0%}" if prob is not None else "")
                problems.append(f"Stück {k + 1} {where} bei {at:.2f} s in Sprache ({detail})")
            elif unsure:
                review.append(f"Stück {k + 1} {where} bei {at:.2f} s mit leisem Laut ({level:.0f} dB): anhören")
    return problems, review


def edge_word(tokens: list[str], last: bool) -> tuple[str, bool]:
    """Randwort eines Stücks und ob es nachsichtig zu prüfen ist (Zahl oder Abkürzung wie „BPI“: die schreibt die
    Erkennung oft anders, „eins Komma neun“ bzw. „B P I“; dann heißt eine Abweichung „anhören“, nicht „Fehler“)."""
    for t in (reversed(tokens) if last else tokens):
        w = norm_word(t)
        if w:
            return w, any(ch.isdigit() for ch in w) or is_acronym(t)
    return "", False


def is_acronym(token: str) -> bool:
    """Abkürzung aus 2–4 Großbuchstaben („BPI“, „DAX“), nicht ganze Wörter in Großschrift („NEIN“ zählt mit 4 schon)."""
    letters = re.sub(r"\W", "", token)
    return 1 < len(letters) <= 4 and letters.isupper() and letters.lower() not in COMMON_CAPS


COMMON_CAPS = {"nein", "ja", "und", "oder", "aber", "doch", "nie", "nur", "hier", "dort", "jetzt", "alle", "kein"}


def heard_near(word: str, heard: list[str], last: bool) -> bool:
    """Ist word am Rand der erkannten Wörter zu hören? Auch zusammengesetzt („Wirecard-Aktie“ = „Wirecard Aktie“)."""
    if not word:
        return True
    edge = heard[-EDGE_WORDS:] if last else heard[:EDGE_WORDS]
    joined = "".join(edge)
    if word in edge or (joined.endswith(word) if last else joined.startswith(word)):
        return True
    return any(difflib.SequenceMatcher(a=word, b=h).ratio() >= 0.95 for h in edge)


def word_report(path: Path, clip: dict, by_id: dict[str, dict], model: Any) -> tuple[list[str], list[str]]:
    """Erkennt den Short neu und vergleicht je Stück mit den dort geplanten Sätzen. Gibt (Fehler, anhören) zurück."""
    segs, _ = model.transcribe(load_audio(path), language=None, word_timestamps=True, vad_filter=False,
                               condition_on_previous_text=False)
    heard = [(norm_word(w.word), (w.start + w.end) / 2) for s in segs for w in (s.words or []) if norm_word(w.word)]
    problems, review = [], []
    for k, ((o0, o1), ids) in enumerate(zip(piece_spans(clip), clip["piece_sents"])):
        tokens = [t[0] for sid in ids for t in by_id[sid]["tokens"]]
        here = [w for w, mid in heard if o0 - 0.1 <= mid <= o1 + 0.1]
        if not here:
            problems.append(f"Stück {k + 1}: nichts zu hören")
            continue
        for last in (False, True):
            word, lenient = edge_word(tokens, last)
            if not heard_near(word, here, last):
                note = f"Stück {k + 1}: {'Ende' if last else 'Anfang'} „{word}“ nicht gehört"
                (review if lenient else problems).append(note + (" (Zahl/Abkürzung): anhören" if lenient else ""))
        expected = [(norm_word(t[0]), sid) for sid in ids for t in by_id[sid]["tokens"] if norm_word(t[0])]
        exp_words = [e[0] for e in expected]
        sm = difflib.SequenceMatcher(a=exp_words, b=here, autojunk=False)
        matched: set[int] = set()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            # gehört: gleiche Wörter oder dieselbe Stelle in anderer Schreibweise, nicht aber ein ganz anderes Wort
            if tag == "equal" or (tag == "replace" and same_spoken(exp_words[i1:i2], here[j1:j2])):
                matched.update(range(i1, i2))
            elif tag in ("delete", "replace") and i2 - i1 >= 2:
                problems.append(f"Stück {k + 1}: fehlt „{' '.join(exp_words[i1:i2])}“")
        for sid in ids:  # ganze Sätze, auch aus einem Wort („Nein.“): an ihrer Stelle muss etwas gehört worden sein
            own = [i for i, e in enumerate(expected) if e[1] == sid]
            if own and not matched.intersection(own):
                problems.append(f"Stück {k + 1}: Satz {sid} „{by_id[sid]['text'][:40]}“ nicht gehört")
    return problems, review


OK, REVIEW, FAIL = "ok", "prüfen", "durchgefallen"
UNSURE = ("enger Schnitt", "geschätzt", "nicht sicher")  # Hinweise aus score.py, die ein Anhören verlangen


def unsure_flags(clip: dict) -> list[str]:
    """Hinweise aus score.py, nach denen eine Grenze nicht sicher gemessen ist (verlangen Anhören)."""
    return [f"aus der Bewertung: {f}" for f in clip.get("flags", []) if any(u in f for u in UNSURE)]


def check_clip(clip: dict, d: Path, by_id: dict[str, dict], model: Any) -> tuple[str, list[str]]:
    """Gibt (ok | prüfen | durchgefallen, Befunde) zurück."""
    path = d / "out" / f"{clip['id']}.mp4"
    if not path.is_file():
        return FAIL, ["Datei fehlt (noch nicht gerendert?)"]
    require_metadata(clip)
    errors, review = [], []
    info = probe(path)
    planned = sum(b - a for a, b in clip["cut"]) / float(clip.get("speed", 1.0))
    if (info.width, info.height) != (1080, 1920):
        errors.append(f"Bildgröße {info.width}x{info.height}")
    if not info.has_audio:
        return FAIL, errors + ["kein Ton"]
    if abs(info.duration - planned) > DUR_TOL:
        errors.append(f"Länge {info.duration:.2f} s statt {planned:.2f} s")
    lufs = loudness(path)
    if abs(lufs - LUFS_TARGET) > LUFS_TOL:
        errors.append(f"Lautheit {lufs:.1f} LUFS statt {LUFS_TARGET:.0f} ±{LUFS_TOL}")
    db = level_db(load_audio(path, SR_LEVEL), HOP)
    if not len(db):
        return FAIL, errors + ["Tonspur leer"]
    edge_errors, unchecked = edge_report(db, HOP / SR_LEVEL, clip, speech_probability(load_audio(path)))
    errors += edge_errors
    review += unchecked
    review += unsure_flags(clip)
    if model is not None:
        word_errors, word_review = word_report(path, clip, by_id, model)
        errors += word_errors
        review += word_review
    if errors:
        return FAIL, errors + review
    if review:
        return REVIEW, review
    return OK, [f"{info.duration:.1f} s, {lufs:.1f} LUFS"]


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Arbeitsordner")
    ap.add_argument("--ids", help="Kommagetrennte Clip-IDs (Standard: alle gerenderten)")
    ap.add_argument("--words", action="store_true", help="Zusätzlich jeden Short neu erkennen (langsamer)")
    ap.add_argument("--model", default="small", help="Modell für --words (Standard small)")
    args = ap.parse_args()
    d = Path(args.dir).resolve()
    cj = read_json(d / "clips.scored.json")
    by_id = {s["id"]: s for s in read_json(d / "sentences.json")["sentences"]}
    want = {x.strip() for x in args.ids.split(",")} if args.ids else None
    clips = [c for c in cj["clips"]
             if (c["id"] in want if want is not None else (d / "out" / f"{c['id']}.mp4").is_file())]
    if not clips:
        raise StepError("Keine gerenderten Shorts gefunden. Erst render.py laufen lassen.")
    model = None
    if args.words:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise StepError("--words braucht faster-whisper: pip install -r requirements.txt") from exc
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    rows = ["| Clip | Ergebnis | Befund |", "|---|---|---|"]
    counts = {OK: 0, REVIEW: 0, FAIL: 0}
    for c in clips:
        status, notes = check_clip(c, d, by_id, model)
        counts[status] += 1
        shown = status if status == OK else f"**{status}**"
        rows.append(f"| {c['id']} | {shown} | {'; '.join(notes)} |")
        print(f"{c['id']}: {status.upper()}, {'; '.join(notes)}")
    words_note = "" if model is not None else "\n\nOhne --words: Wörter wurden nicht neu erkannt und abgeglichen.\n"
    (d / "CHECK.md").write_text("# Prüfung der Shorts\n\n" + "\n".join(rows) + "\n" + words_note, encoding="utf-8")
    summary = f"{counts[OK]} ok, {counts[REVIEW]} prüfen, {counts[FAIL]} durchgefallen -> {d / 'CHECK.md'}"
    print(summary)
    if counts[FAIL]:
        sys.exit(1)


if __name__ == "__main__":
    main()

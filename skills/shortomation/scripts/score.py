"""Schnittgrenzen und Viralitätswert je Clip: Inhaltsraster (Agent) + Messung (Bild, Ton, Text) (Schritt 5).

    python -I score.py --dir <ordner>

Liest <ordner>/sentences.json, analysis.json und clips.json; schreibt clips.scored.json und SCORES.md.

Schnittgrenzen: Ausgangspunkt sind die gemessenen Wortzeiten. Danach folgt der Schnitt dem Pegel, bis der letzte
Laut wirklich verklungen ist (auch leise oder geflüsterte Wortenden: alles, was hörbar über dem Grundrauschen liegt),
und setzt sich in die leiseste Stelle der Pause dahinter. Wo das nicht sicher geht (nächster Satz folgt direkt,
Ende nicht messbar, geschätzte Wortzeiten), steht eine Warnung in SCORES.md statt einer stillen Notlösung.
Gewichte und Schwellen: references/rubric.md.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import LAYOUTS, StepError, read_json, setup_console, write_json

RUBRIC_W = {"hook": .25, "standalone": .20, "payoff": .15, "stakes": .15, "surprise": .15, "comment": .10}
MEASURE_W = {"len": .2, "pace": .2, "gap": .15, "visual": .15, "still": .1, "start": .2}
SHARE_RUBRIC = 0.65
STILL_MOTION = 0.3  # Bildänderung darunter = nur Korn/Rauschen

# Schnitt am Ton (Sekunden, dB)
LEAD_MIN, LEAD_MAX = 0.12, 0.30    # Luft vor dem ersten Laut eines Stücks
TAIL_MIN, TAIL_MAX = 0.15, 0.45    # Luft nach dem letzten Laut eines Stücks
QUIET_RUN = 0.15                   # so lange muss es leise sein, damit ein Laut als beendet gilt
TAIL_SEARCH = 1.5                  # so weit hinter dem gemessenen Wortende wird noch nach Ausklang gesucht
OVER_FLOOR = 6.0                   # alles, was so weit über dem Grundrauschen liegt, zählt als Laut
FLOOR_SPAN = 8.0                   # Umgebung für das Grundrauschen (± s)
MAX_GAP, KEEP_GAP = 1.2, 0.6       # Pausen über MAX_GAP werden auf KEEP_GAP gekürzt
TIGHT = 0.05                       # Suchrand um die Wortgrenze, wenn zwei Sätze ohne Pause aufeinander folgen
P_SOUND = 0.3                      # Sprachwahrscheinlichkeit, ab der ein Fenster als Sprachlaut zählt
BED_PROB, BED_MARGIN = 0.2, 6.0    # Musikbett: Fenster ohne Sprache; ein Laut muss BED_MARGIN dB darüber liegen
PROB_WEIGHT = 30.0                 # beim Suchen der leisesten Stelle: Sprache wie +30 dB werten
SPEED_RANGE = (0.8, 1.3)
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}

CONTEXT_START = re.compile(r"^(Und|Aber|Denn|Doch|Also|Er|Sie|Es|Dort|Da|Dabei|Deren|Dessen|Dieser|Diese|Dieses)\b")
CONTEXT_ANY = re.compile(r"Erinnerst du dich|Kapitel \d|am Anfang|wie gesagt|vorhin|diesen Namen", re.I)
LEGAL = re.compile(r"Betrug|betrüg|gefälscht|fälsch|Haftbefehl|Spionage|Geheimdienst|Täter|angeklagt|Anklage|"
                   r"Geldwäsche|Marktmanipulation|verurteilt|festgenommen", re.I)
HEDGE = re.compile(r"mutmaßlich|laut |soll |sollen |Vorwurf|Verdacht|bestreitet|bestreiten|Unschuldsvermutung|"
                   r"nicht rechtskräftig|Rechtsmittel|Revision|ließ sich das nicht|nach Ansicht", re.I)


class Level:
    """Pegelkurve (dBFS je hop Sekunden; Fenster i deckt [i·hop, (i+1)·hop) ab) mit Hilfen für Schnittpunkte.

    Mit Sprachwahrscheinlichkeit (Silero-VAD) zählt Hintergrundmusik nicht als Laut, solange sie nicht so laut wie
    Sprache ist; ohne sie entscheidet allein der Pegel über dem Grundrauschen."""

    def __init__(self, rms_db: list[float], hop: float, duration: float | None = None,
                 prob: list[float] | None = None) -> None:
        self.db = np.asarray(rms_db, np.float32)
        if len(self.db) == 0:
            raise StepError("Die Tonspur ist leer oder zu kurz.")
        self.hop = hop
        self.duration = float(duration) if duration else len(self.db) * hop  # genaue Tonlänge, wenn bekannt
        self.prob = None
        if prob is not None and len(prob):
            p = np.zeros(len(self.db), np.float32)
            p[:min(len(p), len(prob))] = np.asarray(prob, np.float32)[:len(p)]
            self.prob = p

    def quiet(self, i: int, thr: float) -> bool:
        """Kein hörbarer Laut im Fenster i: unter der Schwelle und (wenn bekannt) keine Sprache laut Silero."""
        if self.prob is None:
            return bool(self.db[i] < thr)
        return bool(self.prob[i] < P_SOUND and self.db[i] < thr)

    def idx(self, t: float) -> int:
        return int(min(max(0, math.floor(t / self.hop)), max(0, len(self.db) - 1)))

    def threshold(self, t0: float, t1: float) -> float:
        """Ab hier gilt etwas als Laut: Grundrauschen der Umgebung plus OVER_FLOOR."""
        i0, i1 = self.idx(t0 - FLOOR_SPAN), self.idx(t1 + FLOOR_SPAN) + 1
        around = self.db[i0:i1]
        if len(around) == 0:
            return -60.0
        thr = float(np.percentile(around, 10)) + OVER_FLOOR
        if self.prob is not None:  # Musikbett = Pegel dort, wo Silero keine Sprache hört; erst darüber zählt ein Laut
            bed = around[self.prob[i0:i1] < BED_PROB]
            if len(bed):
                thr = max(thr, float(np.median(bed)) + BED_MARGIN)
        return thr

    def speech_end(self, t: float, limit: float, thr: float) -> tuple[float, bool]:
        """Erster Zeitpunkt ab t, nach dem der Pegel QUIET_RUN lang unter thr bleibt.

        Gibt (Zeit, sicher) zurück. sicher=False heißt: bis limit (oder TAIL_SEARCH) keine Ruhe gefunden."""
        need = max(1, round(QUIET_RUN / self.hop))
        stop = min(limit, t + TAIL_SEARCH, self.duration)
        i, end = self.idx(t), self.idx(stop)
        quiet = 0
        while i <= end:
            quiet = quiet + 1 if self.quiet(i, thr) else 0
            if quiet >= need:
                return max(t, (i - need + 1) * self.hop), True
            i += 1
        return max(t, stop), False

    def speech_start(self, t: float, limit: float, thr: float) -> tuple[float, bool]:
        """Erster Laut vor t: rückwärts, bis der Pegel QUIET_RUN lang unter thr liegt (nicht vor limit)."""
        need = max(1, round(QUIET_RUN / self.hop))
        stop = max(limit, t - TAIL_SEARCH, 0.0)
        i, end = self.idx(t), self.idx(stop)
        quiet = 0
        while i >= end:
            quiet = quiet + 1 if self.quiet(i, thr) else 0
            if quiet >= need:
                return min(t, (i + need) * self.hop), True
            i -= 1
        return min(t, stop), stop <= 0.0

    def quietest(self, target: float, lo: float, hi: float) -> float:
        """Leiseste Stelle in [lo, hi], bei Gleichstand die nächste zu target."""
        if hi <= lo:
            return round(lo, 3)
        i0, i1 = self.idx(lo), self.idx(hi)
        if i1 <= i0:
            return round((lo + hi) / 2, 3)
        seg = self.db[i0:i1 + 1].astype(np.float64)
        if self.prob is not None:  # Sprache wiegt schwerer als gleich laute Musik
            seg = seg + PROB_WEIGHT * self.prob[i0:i1 + 1]
        cand = [(float(v), abs((i0 + j + 0.5) * self.hop - target), (i0 + j + 0.5) * self.hop)
                for j, v in enumerate(seg)]
        return round(min(max(min(cand)[2], lo), hi), 3)


@dataclass
class Cut:
    pieces: list[list[float]] = field(default_factory=list)   # [[a, b]] in Sekunden des Originals
    air: list[list[float]] = field(default_factory=list)      # je Stück [Luft vor dem 1. Laut, nach dem letzten]
    piece_sents: list[list[str]] = field(default_factory=list)
    sents: list[dict] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def cut_part(s0: str, s1: str, sents: list[dict], idx: dict[str, int], level: Level, tighten: bool = True) -> Cut:
    """Schnitt für den Satzbereich s0..s1, in Stücken, wenn lange Pausen gekürzt werden."""
    if s0 not in idx or s1 not in idx:
        raise StepError(f"Unbekannte Satz-ID im Bereich {s0}–{s1}")
    i0, i1 = idx[s0], idx[s1]
    if i1 < i0:
        raise StepError(f"Bereich rückwärts: {s0}–{s1}")
    part = sents[i0:i1 + 1]
    thr = level.threshold(part[0]["t0"], part[-1]["t1"])
    res = Cut(sents=part)

    # Nachbarsätze, ebenfalls am Pegel gemessen: Der Schnitt darf nicht in ihren An- oder Ausklang fallen.
    # Endet der Satz davor ohne messbare Pause, gilt nur sein gemessenes Wortende als Grenze (sonst würde die Suche
    # nach seinem Ausklang den Anfang des gewählten Satzes verschlucken).
    prev_end = 0.0
    tight_start = False
    if i0 > 0:
        prev_end, sure = level.speech_end(sents[i0 - 1]["t1"], part[0]["t0"], thr)
        if not sure:
            prev_end = min(sents[i0 - 1]["t1"], part[0]["t0"])
            tight_start = True
            res.notes.append(f"enger Schnitt vor {part[0]['id']} (keine Pause zum Satz davor)")
    starts: list[float] = []
    ends: list[float] = []
    for k, s in enumerate(part):
        lo = prev_end if k == 0 else ends[-1]
        st, sure = level.speech_start(s["t0"], lo, thr)
        if not sure and k == 0 and i0 > 0 and not res.notes:
            res.notes.append(f"Anfang von {s['id']} nicht sicher gemessen")
        hi = part[k + 1]["t0"] if k + 1 < len(part) else (sents[i1 + 1]["t0"] if i1 + 1 < len(sents) else level.duration)
        en, sure = level.speech_end(s["t1"], hi, thr)
        to_eof = k == len(part) - 1 and i1 + 1 >= len(sents) and en >= level.duration - level.hop
        if not sure and not to_eof:  # keine messbare Ruhe (Musikbett, Sätze ohne Pause): gemessenes Wortende gilt
            en = s["t1"]
            if k == len(part) - 1:
                res.notes.append(f"Ende von {s['id']} nicht sicher gemessen")
        starts.append(st)
        ends.append(en)
    if i1 + 1 < len(sents):
        next_start = level.speech_start(sents[i1 + 1]["t0"], ends[-1], thr)[0]
    else:
        next_start = level.duration
    for s in (part[0], part[-1]):
        if s.get("est_words"):
            res.notes.append(f"Wortzeiten in {s['id']} geschätzt")

    if tight_start:  # leiseste Stelle zwischen dem letzten Wort davor und dem ersten Wort des Clips
        start = level.quietest(prev_end, prev_end - TIGHT, max(prev_end, part[0]["t0"]) + TIGHT)
    else:
        start = level.quietest(starts[0] - LEAD_MIN, max(0.0, prev_end, starts[0] - LEAD_MAX), starts[0] - LEAD_MIN)
    first = 0
    for k in range(len(part) - 1):
        gap = starts[k + 1] - ends[k]
        if tighten and gap > MAX_GAP:
            half = KEEP_GAP / 2
            end = level.quietest(ends[k] + half, ends[k] + TAIL_MIN, ends[k] + half + 0.1)
            res.pieces.append([round(start, 3), end])
            res.air.append([round(starts[first] - start, 3), round(end - ends[k], 3)])
            res.piece_sents.append([s["id"] for s in part[first:k + 1]])
            start = level.quietest(starts[k + 1] - half, starts[k + 1] - half - 0.1, starts[k + 1] - LEAD_MIN)
            first = k + 1
            res.gaps.append(KEEP_GAP)
        else:
            res.gaps.append(max(0.0, gap))
    room = next_start - ends[-1]
    if i1 + 1 >= len(sents) and ends[-1] >= level.duration - level.hop:
        end = level.duration  # Ton klingt bis zum Dateiende: alles mitnehmen, nichts abschneiden
        res.notes.append(f"enger Schnitt nach {part[-1]['id']} (Ton läuft bis zum Dateiende)")
    elif room >= TAIL_MIN + 0.02:
        end = level.quietest(ends[-1] + TAIL_MIN, ends[-1] + TAIL_MIN, min(next_start - 0.02, ends[-1] + TAIL_MAX))
    else:  # der nächste Satz folgt so dicht, dass die Mindestluft nicht passt: leiseste Stelle zwischen den Wörtern
        nxt_word_end = sents[i1 + 1]["tokens"][0][2] if i1 + 1 < len(sents) else level.duration
        hi = min(ends[-1] + 2 * TIGHT + 0.2, max(ends[-1] + TIGHT, nxt_word_end - 2 * TIGHT))
        end = level.quietest(ends[-1], ends[-1] - TIGHT, hi)
        res.notes.append(f"enger Schnitt nach {part[-1]['id']} ({max(0.0, room):.2f} s Pause)")
    end = min(end, level.duration)
    res.pieces.append([round(start, 3), round(end, 3)])
    res.air.append([round(starts[first] - start, 3), round(end - ends[-1], 3)])
    res.piece_sents.append([s["id"] for s in part[first:]])
    return res


def ramp(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """1 zwischen lo1 und hi1, linear auf 0 bei lo0 bzw. hi0."""
    if x < lo1:
        return max(0.0, (x - lo0) / (lo1 - lo0))
    if x > hi1:
        return max(0.0, (hi0 - x) / (hi0 - hi1))
    return 1.0


def measure(cut: list[list[float]], parts_sents: list[list[dict]], gaps: list[float], an: dict, speed: float
            ) -> tuple[dict, dict]:
    dur = sum(b - a for a, b in cut) / speed
    words = sum(s["n_words"] for p in parts_sents for s in p)
    speech = sum(s["t1"] - s["t0"] for p in parts_sents for s in p) / speed
    gaps = [g / speed for g in gaps]
    fps, motion = an.get("motion_fps", 10), an["motion"]
    n_cuts = sum(1 for c in an["cuts"] for a, b in cut if a + 0.3 < c < b)
    mot = [motion[i] for a, b in cut for i in range(int(a * fps), min(len(motion), int(b * fps)))]
    mot_mean = sum(mot) / max(1, len(mot))
    flat = run = 0  # längste Strecke ohne sichtbare Bewegung
    for v in mot:
        run = run + 1 if v < STILL_MOTION else 0
        flat = max(flat, run)
    flat_s = flat / fps / speed
    first = parts_sents[0][0]
    m = {
        "len": ramp(dur, 12, 25, 50, 90),
        "pace": ramp(words / max(0.1, speech), 1.3, 2.2, 3.2, 4.2),
        "gap": ramp(max(gaps, default=0), -1, 0, MAX_GAP, 3.0),
        "visual": min(1.0, 0.25 + 0.5 * min(1.0, n_cuts / max(1e-6, dur) * 10 / 2.5) + 0.25 * min(1.0, mot_mean / 3)),
        "still": ramp(flat_s, -1, 0, 1.5, 4.0),
        "start": 0.5 * ramp(first["n_words"], -1, 0, 12, 25)
                 + 0.5 * ramp((first["t0"] - cut[0][0]) / speed, -1, 0, 0.4, 1.5),
    }
    info = {"dur": round(dur, 1), "words_per_s": round(words / max(0.1, speech), 2),
            "max_gap": round(max(gaps, default=0), 2), "cuts": n_cuts, "motion": round(mot_mean, 2),
            "still_s": round(flat_s, 1)}
    return {k: round(v, 2) for k, v in m.items()}, info


def text_checks(parts_sents: list[list[dict]], cut: list[list[float]], claimed: list, attribution: list
                ) -> tuple[int, list[str]]:
    flat = [s for p in parts_sents for s in p]
    penalty, flags = 0, []
    for k, s in enumerate(flat[:2]):
        if (k == 0 and CONTEXT_START.match(s["text"])) or CONTEXT_ANY.search(s["text"]):
            penalty += 8
            flags.append(f"Kontext: „{s['text'][:40]}…“")
    for s in flat[2:]:
        if CONTEXT_ANY.search(s["text"]):
            flags.append(f"Verweis im Clip: „{s['text'][:40]}…“")
    if any(x < b and a < y for a, b in claimed for x, y in cut):
        flags.append("Content-ID-Musik")
    for a, b, credit in attribution:
        if any(x < b and a < y for x, y in cut):
            flags.append(f"Nachweis nötig: {credit}")
    text = " ".join(s["text"] for s in flat)
    if LEGAL.search(text):
        flags.append("⚖ prüfen" + ("" if HEDGE.search(text) else " (keine Einschränkung im Clip)"))
    return min(16, penalty), flags


def overlap(c1: list[list[float]], c2: list[list[float]]) -> float:
    inter = sum(max(0.0, min(b, d) - max(a, c)) for a, b in c1 for c, d in c2)
    return inter / max(1e-6, min(sum(b - a for a, b in c1), sum(d - c for c, d in c2)))


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ranges(cj: dict, key: str, width: int) -> None:
    items = cj.get(key, [])
    if not isinstance(items, list):
        raise StepError(f"„{key}“ muss eine Liste sein.")
    for r in items:
        if (not isinstance(r, list) or len(r) != width or not _number(r[0]) or not _number(r[1])
                or r[0] >= r[1] or (width == 3 and not isinstance(r[2], str))):
            form = "[Start, Ende, \"Nachweis\"]" if width == 3 else "[Start, Ende]"
            raise StepError(f"„{key}“ braucht Einträge wie {form} in Sekunden, gefunden: {r}")


def validate(cj: object) -> None:
    """Prüft clips.json, bevor gerechnet wird; nennt jedes Problem mit Clip-ID."""
    if not isinstance(cj, dict):
        raise StepError("clips.json muss ein Objekt { … } sein.")
    if not isinstance(cj.get("video"), str) or not cj["video"]:
        raise StepError("clips.json braucht „video“ (Pfad zum langen Video, relativ zum Arbeitsordner).")
    for key in ("channel", "cta"):
        if key in cj and not isinstance(cj[key], str):
            raise StepError(f"„{key}“ muss Text sein, gefunden: {cj[key]!r}")
    clips = cj.get("clips")
    if not isinstance(clips, list) or not clips:
        raise StepError("clips.json braucht eine Liste „clips“ mit mindestens einem Clip.")
    seen: set[str] = set()
    for c in clips:
        if not isinstance(c, dict):
            raise StepError(f"Jeder Clip muss ein Objekt sein, gefunden: {c!r}")
        cid = c.get("id")
        if not isinstance(cid, str) or not ID_PATTERN.match(cid) or cid.lower() in RESERVED:
            raise StepError(f"Clip-ID {cid!r} ungültig: nur Buchstaben, Ziffern, - und _, höchstens 32 Zeichen.")
        if cid.lower() in seen:
            raise StepError(f"Clip-ID doppelt (Groß/Klein zählt nicht): {cid}")
        seen.add(cid.lower())
        if not isinstance(c.get("hook"), str) or not c["hook"].strip():
            raise StepError(f"{cid}: „hook“ fehlt.")
        for key in ("label", "title", "why"):
            if key in c and not isinstance(c[key], str):
                raise StepError(f"{cid}: „{key}“ muss Text sein, gefunden: {c[key]!r}")
        if "layout" in c and c["layout"] not in LAYOUTS:
            raise StepError(f"{cid}: layout {c['layout']!r} unbekannt, erlaubt: {', '.join(LAYOUTS)}")
        if "tighten" in c and not isinstance(c["tighten"], bool):
            raise StepError(f"{cid}: „tighten“ muss true oder false sein.")
        parts = c.get("parts")
        if not isinstance(parts, list) or not parts or any(
                not isinstance(p, list) or len(p) != 2 or not all(isinstance(x, str) for x in p) for p in parts):
            raise StepError(f"{cid}: „parts“ muss wie [[\"S002\", \"S011\"]] aussehen.")
        rubric = c.get("rubric")
        if not isinstance(rubric, dict):
            raise StepError(f"{cid}: „rubric“ fehlt.")
        for k in RUBRIC_W:
            if not _number(rubric.get(k)) or not 0 <= rubric[k] <= 10:
                raise StepError(f"{cid}: Rasterwert „{k}“ fehlt oder liegt nicht zwischen 0 und 10.")
        speed = c.get("speed", cj.get("speed", 1.0))
        if not _number(speed) or not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise StepError(f"{cid}: speed {speed!r} muss eine Zahl zwischen {SPEED_RANGE[0]} und {SPEED_RANGE[1]} sein.")
    _ranges(cj, "claimed", 2)
    _ranges(cj, "attribution", 3)


def score_clips(cj: dict, sents: list[dict], an: dict) -> list[dict]:
    idx = {s["id"]: i for i, s in enumerate(sents)}
    level = Level(an["rms_db"], an["hop_s"], an.get("audio_duration"), an.get("speech_prob"))
    out = []
    for c in cj["clips"]:
        speed = float(c.get("speed", cj.get("speed", 1.0)))
        tighten = bool(c.get("tighten", cj.get("tighten", True)))
        cuts = [cut_part(a, b, sents, idx, level, tighten) for a, b in c["parts"]]
        cut = [p for r in cuts for p in r.pieces]
        ps = [r.sents for r in cuts]
        m, info = measure(cut, ps, [g for r in cuts for g in r.gaps], an, speed)
        penalty, flags = text_checks(ps, cut, cj.get("claimed", []), cj.get("attribution", []))
        flags += [n for r in cuts for n in r.notes]
        r = sum(RUBRIC_W[k] * float(c["rubric"][k]) for k in RUBRIC_W) * 10
        meas = sum(MEASURE_W[k] * m[k] for k in MEASURE_W) * 100
        score = max(0, min(100, round(SHARE_RUBRIC * r + (1 - SHARE_RUBRIC) * meas - penalty)))
        out.append({**c, "speed": speed, "cut": cut, "air": [a for r_ in cuts for a in r_.air],
                    "piece_sents": [p for r_ in cuts for p in r_.piece_sents], "rubric_score": round(r),
                    "measured_score": round(meas), "measured": m, "info": info, "penalty": penalty, "flags": flags,
                    "score": score, "text": " ".join(s["text"] for p in ps for s in p)})
    out.sort(key=lambda c: -c["score"])
    accepted: list[dict] = []  # Überlappung nur gegen Clips, die selbst gerendert würden
    for c in out:
        if any(f.startswith("Content-ID") for f in c["flags"]):
            continue
        hit = next((a for a in accepted if overlap(c["cut"], a["cut"]) > 0.3), None)
        if hit:
            c["flags"].append(f"überlappt {hit['id']}")
        else:
            accepted.append(c)
    return out


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Arbeitsordner")
    d = Path(ap.parse_args().dir)
    sents = read_json(d / "sentences.json")["sentences"]
    an = read_json(d / "analysis.json")
    cj = read_json(d / "clips.json")
    if sents and "tokens" not in sents[0]:
        raise StepError("sentences.json stammt aus einer alten Version. Bitte transcribe.py und windows.py neu laufen lassen.")
    validate(cj)
    out = score_clips(cj, sents, an)
    write_json(d / "clips.scored.json", {**cj, "clips": out})
    rows = ["| # | Clip | Wert | Inhalt | Messung | Länge | Hook | Hinweise |", "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(out, 1):
        rows.append(f"| {i} | {c['id']} | **{c['score']}** | {c['rubric_score']} | {c['measured_score']} | "
                    f"{c['info']['dur']} s | {c['hook']} | {'; '.join(c['flags']) or '–'} |")
    detail = [f"- **{c['id']}** {c.get('label', '')}: Tempo {c['info']['words_per_s']} W/s, längste Pause "
              f"{c['info']['max_gap']} s, {c['info']['cuts']} Schnitte, Messwerte {c['measured']}. {c.get('why', '')}"
              for c in out]
    (d / "SCORES.md").write_text("# Shorts-Rangliste\n\n" + "\n".join(rows) + "\n\n" + "\n".join(detail) + "\n",
                                 encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()

"""Wortgenaues Transkript mit Zeiten (Schritt 1).

    python -I transcribe.py <video> --out <ordner> [--srt <datei.srt>] [--model small] [--language de]

Erkennt die Sprache lokal mit faster-whisper und schreibt <ordner>/words.json mit Start und Ende jedes Wortes.
Mit --srt kommt der Wortlaut aus den Untertiteln (richtige Schreibweise von Namen und Zahlen), die Zeiten aus der
Erkennung. Stellen, die die Erkennung im ganzen Durchlauf überhört hat, werden einzeln nachgehört. Was danach noch
keine gemessene Zeit hat, ist in words.json mit "est": true markiert und wird in score.py gemeldet.
Nichts wird hochgeladen; beim ersten Lauf lädt faster-whisper das Modell einmal herunter.
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
from common import SR_SPEECH, StepError, load_audio, norm_word, parse_srt, setup_console, write_json

MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
# Sprachfilter großzügig: leise und geflüsterte Stellen sollen nicht als Stille wegfallen
VAD = {"threshold": 0.3, "min_silence_duration_ms": 800, "speech_pad_ms": 400}
REDO_PAD = 1.0  # so viel Ton vor und nach einer überhörten Stelle wird beim Nachhören mitgenommen (s)
SAME_SPOKEN = 0.6  # Ähnlichkeit der Buchstaben, ab der eine abweichende Stelle als dieselbe gilt
MIN_WORD = 0.12    # so viel Zeit braucht ein gesprochenes Wort mindestens (s); weniger heißt: falscher Anker
# Ein ganzes Wort aus Zahlbausteinen („dreiundvierzig“, „zweitausendzwanzig“, „eins“, „komma“), nicht „keine“/„Stein“
NUMBER_WORD = re.compile(
    r"(?:null|eins|eine[mnrs]?|ein|zwei|zwo|drei|vier|fünf|sechs|sieben|sieb|acht|neun|zehn|elf|zwölf|zwanzig|dreißig"
    r"|zig|ßig|hundert|tausend|und|millionen|million|milliarden|milliarde|komma|prozent|zero|one|two|three|four|five"
    r"|six|seven|eight|nine|ten|eleven|twelve|thirteen|fifteen|twenty|thirty|forty|fifty|teen|ty|hundred|thousand"
    r"|million|billion|percent|point|and)+")
WIDEN_MAX = 6      # höchstens so viele Nachbaranker je Seite werden gelöst


def load_model(name: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise StepError("faster-whisper fehlt: pip install -r requirements.txt") from exc
    return WhisperModel(name, device="cpu", compute_type="int8")


def recognize(model: Any, audio: np.ndarray, language: str | None, offset: float = 0.0, vad: bool = True,
              progress: bool = False) -> tuple[list[dict], str]:
    """Erkennung mit Wortzeiten. Gibt ([{w, t0, t1}], Sprache) zurück; Zeiten plus offset."""
    segments, info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=vad,
                                      vad_parameters=VAD if vad else None, condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"w": text, "t0": round(offset + float(w.start), 3),
                              "t1": round(offset + float(w.end), 3)})
        if progress:
            print(f"\r  {(offset + seg.end) / 60:5.1f} min erkannt", end="", flush=True)
    if progress:
        print()
    return words, info.language


def same_spoken(srt: list[str], heard: list[str]) -> bool:
    """Ist eine abweichende Stelle dasselbe Gesprochene in anderer Schreibweise?

    Zahlen („1,9“ gegen „eins Komma neun“) und ähnlich klingende Schreibweisen ja; ein ganz anderes Wort
    („Eine Milliarde“ gegen „Unsinn“) nein, dann gilt die Stelle als überhört."""
    if any(ch.isdigit() for w in [*srt, *heard] for ch in w):
        # Zahl gegen Zahl: Beide Seiten müssen eine Zahl enthalten (Ziffern oder Zahlwort), sonst überhört
        return all(any(ch.isdigit() for w in side for ch in w) or any(NUMBER_WORD.fullmatch(w) for w in side)
                   for side in (srt, heard))
    a, b = "".join(srt), "".join(heard)
    return bool(a and b) and difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= SAME_SPOKEN


def match_times(tokens: list[str], heard: list[dict]) -> list[tuple[float, float] | None]:
    """Zeiten für die Tokens aus den erkannten Wörtern: gleiche Wörter direkt, abweichende Schreibweisen derselben
    Stelle über die erkannten Wörter verteilt. Alles andere bleibt None (überhört)."""
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    if not tokens or not heard:
        return times
    a = [norm_word(t) for t in tokens]
    b = [norm_word(h["w"]) for h in heard]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k]["t0"], heard[j1 + k]["t1"])
        elif tag == "replace" and same_spoken(a[i1:i2], b[j1:j2]):
            spread(tokens, times, i1, i2, heard[j1]["t0"], heard[j2 - 1]["t1"])
    return times


def spread(tokens: list[str], times: list, i: int, j: int, lo: float, hi: float) -> None:
    """Verteilt [lo, hi] nach Zeichenanteil auf die Tokens i..j-1 (hi wird nie kleiner als lo)."""
    hi = max(hi, lo)
    weights = [len(tokens[k]) + 1 for k in range(i, j)]
    total = sum(weights)
    t = lo
    for k, wt in zip(range(i, j), weights):
        span = (hi - lo) * wt / total
        times[k] = (t, t + span)
        t += span


def gaps(times: list) -> list[tuple[int, int]]:
    """Läufe von Tokens ohne Zeit als (i, j)."""
    out, i = [], 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        out.append((i, j))
        i = j
    return out


def bounds(times: list, i: int, j: int, start: float, end: float) -> tuple[float, float]:
    """Zeitfenster zwischen den gemessenen Nachbarn eines Laufs."""
    lo = times[i - 1][1] if i > 0 else start
    hi = times[j][0] if j < len(times) else end
    return lo, max(lo, hi)


def release_weak_anchors(times: list, start: float, end: float) -> None:
    """Löst Anker, zwischen denen die überhörten Wörter keinen Platz hätten.

    Verhört sich die Erkennung zufällig zu einem Wort der SRT („Ende“ als „Eine“), entsteht ein falscher Anker, und
    für die Wörter dazwischen bleibt keine Zeit. Dann werden abwechselnd links und rechts Nachbaranker gelöst, bis
    jedes Wort mindestens MIN_WORD Zeit hat; die Stelle wird danach als Ganzes nachgehört."""
    done: set[tuple[int, int]] = set()
    while True:
        todo = [(i, j) for i, j in gaps(times) if (i, j) not in done]
        if not todo:
            return
        i, j = todo[0]
        lo, hi = bounds(times, i, j, start, end)
        for step in range(2 * WIDEN_MAX):
            if hi - lo >= MIN_WORD * (j - i):
                break
            if step % 2 == 0 and i > 0:
                i -= 1
            elif j < len(times):
                j += 1
            elif i > 0:
                i -= 1
            else:
                break
            times[i:j] = [None] * (j - i)
            while i > 0 and times[i - 1] is None:  # an benachbarte Lücken anschließen, bevor Grenzen gelesen werden
                i -= 1
            while j < len(times) and times[j] is None:
                j += 1
            lo, hi = bounds(times, i, j, start, end)
        done.add((i, j))  # nach dem Lösen ist der Lauf (i, j) erledigt; neu verschmolzene Läufe kommen danach dran


def align_srt(cues: list[dict], heard: list[dict], audio: np.ndarray | None = None, model: Any = None,
              language: str | None = None) -> list[dict]:
    """Wortlaut der SRT, Zeiten nur aus der Erkennung (SRT-Zeiten sind oft ungenau und zählen nicht).

    Überhörte Stellen werden, wenn audio und model da sind, einzeln ohne Sprachfilter nachgehört. Bleibt ein Wort
    ohne gemessene Zeit, bekommt es eine Zeit zwischen seinen Nachbarn und "est": True.
    """
    tokens = [w for c in cues for w in c["text"].split()]
    if not tokens or not heard:
        return []
    times = match_times(tokens, heard)
    # Ohne gemessenen Nachbarn reicht eine überhörte Stelle bis an den Anfang bzw. das Ende der Aufnahme
    start = 0.0 if audio is not None else heard[0]["t0"]
    end = len(audio) / SR_SPEECH if audio is not None else heard[-1]["t1"]
    release_weak_anchors(times, start, end)
    if audio is not None and model is not None:
        for i, j in gaps(times):
            lo, hi = bounds(times, i, j, start, end)
            a = max(0.0, lo - REDO_PAD)
            snippet = audio[int(a * SR_SPEECH):int((hi + REDO_PAD) * SR_SPEECH)]
            if len(snippet) < SR_SPEECH // 4:
                continue
            again, _ = recognize(model, snippet, language, offset=a, vad=False)
            again = [w for w in again if lo - 0.3 <= w["t0"] and w["t1"] <= hi + 0.3]
            for k, tm in enumerate(match_times(tokens[i:j], again)):
                if tm is not None:
                    times[i + k] = tm
    estimated = [False] * len(tokens)
    for i, j in gaps(times):
        lo, hi = bounds(times, i, j, start, end)
        spread(tokens, times, i, j, lo, hi)
        estimated[i:j] = [True] * (j - i)
    out = []
    last = 0.0
    for t, (t0, t1), est in zip(tokens, times, estimated):
        t0 = max(t0, last)  # Zeiten steigen nie rückwärts
        t1 = max(t1, t0)
        last = t1
        word = {"w": t, "t0": round(t0, 3), "t1": round(t1, 3)}
        if est:
            word["est"] = True
        out.append(word)
    return out


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="Arbeitsordner")
    ap.add_argument("--srt", help="Untertitel mit dem richtigen Wortlaut (optional)")
    ap.add_argument("--model", default="small", choices=MODELS,
                    help="Modellgröße (Standard small; large-v3-turbo ist genauer, aber langsamer)")
    ap.add_argument("--language", default=None, help="Sprachcode wie de oder en (Standard: automatisch)")
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise StepError(f"Video nicht gefunden: {video}")
    cues = None
    if args.srt:
        if not Path(args.srt).is_file():
            raise StepError(f"SRT nicht gefunden: {args.srt}")
        cues = parse_srt(args.srt)
        if not cues:
            raise StepError(f"Keine Untertitel in {args.srt}. Ist das eine SRT-Datei?")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Erkenne Sprache in {video.name} (Modell {args.model}) ...")
    model = load_model(args.model)
    audio = load_audio(video)
    heard, language = recognize(model, audio, args.language, progress=True)
    if not heard:
        raise StepError("Keine Sprache erkannt. Hat das Video eine Tonspur mit Sprache?")
    data: dict = {"video": str(video.resolve()), "model": args.model, "language": language, "source": "whisper",
                  "words": heard}
    if cues:
        words = align_srt(cues, heard, audio, model, args.language or language)
        n_est = sum(1 for w in words if w.get("est"))
        matched = difflib.SequenceMatcher(a=[norm_word(w["w"]) for w in words],
                                          b=[norm_word(h["w"]) for h in heard], autojunk=False).ratio()
        print(f"SRT und Erkennung stimmen zu {matched:.0%} überein; {n_est} Wörter ohne gemessene Zeit.")
        if matched < 0.6:
            print("Warnung: Geringe Übereinstimmung. Gehört die SRT zu diesem Video?", file=sys.stderr)
        data.update(source="srt+whisper", words=words, heard=heard)
    write_json(out / "words.json", data)
    print(f"{len(data['words'])} Wörter ({language}, {data['source']}) -> {out / 'words.json'}")


if __name__ == "__main__":
    main()

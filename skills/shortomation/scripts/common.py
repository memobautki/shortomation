"""Gemeinsame Helfer für alle Schritte: ffmpeg-Aufrufe, Ton laden, Video-Eckdaten, SRT lesen, Wörter vergleichen."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SR_SPEECH = 16000  # Abtastrate für die Spracherkennung
SR_LEVEL, HOP_LEVEL = 8000, 400  # Pegelmessung: 8 kHz, Fenster 0,05 s
FIRST_FADE, PIECE_FADE, FINAL_FADE = 0.05, 0.04, 0.06  # Tonblenden der Stücke (s); liegen in Pausen
# (score.py lässt vor dem ersten Laut mindestens 0,12 s und nach dem letzten mindestens 0,15 s Luft; zwischen
# Blende und Laut bleibt so ein Ruhefenster, das check.py nachmessen kann)
LAYOUTS = ("content", "full", "center", "face")  # Bild-Zuschnitte für render.py (Beschreibung dort)
TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


class StepError(SystemExit):
    """Abbruch mit verständlicher Meldung (Exit-Code 1)."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Fehler: {message}")


def setup_console() -> None:
    """Umlaute auch in der Windows-Konsole ausgeben."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    """Startet ein Programm; bei Fehler Abbruch mit dem Ende der Fehlermeldung."""
    try:
        res = subprocess.run(cmd, capture_output=True, **kw)
    except FileNotFoundError as exc:
        raise StepError(f"{cmd[0]} nicht gefunden. Ist es installiert und im PATH?") from exc
    if res.returncode:
        tail = res.stderr.decode("utf-8", "replace")[-800:] if res.stderr else ""
        raise StepError(f"{cmd[0]} ist fehlgeschlagen: {tail}")
    return res


def load_audio(path: str | Path, sr: int = SR_SPEECH) -> np.ndarray:
    """Ganze Tonspur als Mono-Float32 in [-1, 1], auf der Zeitachse des Videos.

    Beginnt die Tonspur später als das Video (Startversatz), wird vorn Stille eingefügt, damit Sekunde t hier
    dieselbe Stelle ist wie beim Schneiden in render.py."""
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0", "-af", "aresample=async=1:first_pts=0",
               "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]).stdout
    if not raw:
        raise StepError(f"Keine Tonspur in {path}")
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def level_db(audio: np.ndarray, hop: int = HOP_LEVEL) -> np.ndarray:
    """Pegel (dBFS) je Fenster von hop Abtastwerten; ein angebrochenes letztes Fenster zählt mit."""
    n_full = len(audio) // hop
    values = []
    if n_full:
        values.append(np.sqrt((audio[: n_full * hop].reshape(n_full, hop) ** 2).mean(axis=1) + 1e-12))
    rest = audio[n_full * hop:]
    if len(rest):
        values.append(np.array([np.sqrt((rest ** 2).mean() + 1e-12)]))
    if not values:
        return np.zeros(0, np.float32)
    return (20 * np.log10(np.concatenate(values))).astype(np.float32)


VAD_FRAME = 512  # Silero-VAD: 512 Abtastwerte bei 16 kHz = 32 ms


def speech_probability(audio16k: np.ndarray, hop_s: float | None = None) -> np.ndarray | None:
    """Sprachwahrscheinlichkeit (0–1) aus Silero-VAD, das faster-whisper mitbringt. Unterscheidet Sprache von Musik.

    Ohne hop_s je 32 ms; mit hop_s das Maximum je Fenster dieser Länge (passend zur Pegelkurve).
    None, wenn faster-whisper fehlt."""
    try:
        from faster_whisper.vad import get_vad_model
    except ImportError:
        return None
    usable = len(audio16k) // VAD_FRAME * VAD_FRAME
    try:
        probs = (np.asarray(get_vad_model()(audio16k[:usable]), np.float32).ravel() if usable
                 else np.zeros(0, np.float32))
    except (TypeError, ValueError, RuntimeError) as exc:  # andere faster-whisper-Version: nur Pegel verwenden
        print(f"Hinweis: Sprach-Erkennung (Silero-VAD) nicht nutzbar ({exc}); faster-whisper>=1.2 installieren. "
              "Es wird nur der Pegel verwendet.", file=sys.stderr)
        return None
    if hop_s is None:
        return probs
    frame_s = VAD_FRAME / SR_SPEECH
    n = int(np.ceil(len(audio16k) / SR_SPEECH / hop_s))
    out = np.zeros(n, np.float32)
    for i in range(n):
        seg = probs[int(i * hop_s / frame_s):max(int(i * hop_s / frame_s) + 1, int(np.ceil((i + 1) * hop_s / frame_s)))]
        out[i] = float(seg.max()) if len(seg) else 0.0
    return out


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    """Breite, Höhe (wie angezeigt, also nach Drehung), Bildrate, Dauer und ob es Ton gibt."""
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "stream=codec_type,width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate:"
               "stream_side_data=rotation:format=duration", "-of", "json", str(path)]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise StepError(f"Keine Videospur in {path}")
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(video.get(key) or "0/1").partition("/")
        if float(den or 1) and float(num):
            fps = float(num) / float(den or 1)
            break
    width, height = int(video["width"]), int(video["height"])
    rotation = video.get("tags", {}).get("rotate") or next(
        (d.get("rotation") for d in video.get("side_data_list", []) if "rotation" in d), 0)
    if int(float(rotation or 0)) % 180:  # Handyvideos: hochkant gespeichert als gedrehtes Querformat
        width, height = height, width
    return VideoInfo(width=width, height=height, fps=fps or 30.0,
                     duration=float(data.get("format", {}).get("duration", 0.0)),
                     has_audio=any(s.get("codec_type") == "audio" for s in streams))


def parse_srt(path: str | Path) -> list[dict]:
    """SRT -> [{t0, t1, text}] (Sekunden)."""
    cues = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        for i, line in enumerate(lines):
            m = TS.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append({"t0": t0, "t1": t1, "text": " ".join(lines[i + 1:]).strip()})
                break
    return cues


def norm_word(word: str) -> str:
    """Vergleichsform eines Wortes: klein, nur Buchstaben und Ziffern („1,9“ -> „19“, „Euro.“ -> „euro“)."""
    return re.sub(r"[\W_]", "", word.lower())


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StepError(f"{path.name} fehlt in {path.parent}. Ist der vorige Schritt gelaufen?") from exc
    except json.JSONDecodeError as exc:
        raise StepError(f"{path} ist kein gültiges JSON (Zeile {exc.lineno}): {exc.msg}") from exc


def write_json(path: Path, data: Any, indent: int | None = 1) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")

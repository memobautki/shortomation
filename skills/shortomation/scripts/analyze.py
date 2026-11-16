"""Misst das ganze Video einmal: Schnitte, Bildbewegung, Lautstärke (Schritt 3).

    python -I analyze.py <video> --out <ordner>

Schreibt <ordner>/analysis.json:
  width, height, fps, duration   Eckdaten des Videos
  cuts      Zeitpunkte harter Bildwechsel (s)
  motion    mittlere Bildänderung je 0,1 s (0-255, Graustufen 160x90)
  rms_db    Pegel des Tons je hop_s Sekunden (dBFS, Mono 8 kHz; das letzte Fenster kann kürzer sein)
  audio_duration  genaue Länge der Tonspur (s)
  speech_prob     Sprachwahrscheinlichkeit je hop_s (Silero-VAD; trennt Sprache von Hintergrundmusik)
Liest nur, verändert das Video nicht.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import (HOP_LEVEL, SR_LEVEL, StepError, level_db, load_audio, probe, setup_console, speech_probability,
                    write_json)

W, H, FPS = 160, 90, 10
CUT_MIN, CUT_FACTOR, CUT_GAP = 6.0, 4.0, 0.4  # harter Schnitt: Sprung > max(CUT_MIN, CUT_FACTOR × lokaler Median)


def video_motion(path: Path) -> np.ndarray:
    """Mittlere Bildänderung je Bild bei FPS, Bild für Bild gelesen (auch lange Videos brauchen kaum Speicher)."""
    size = W * H
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={W}:{H},format=gray",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    values: list[float] = []
    prev = None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).astype(np.int16)
            values.append(0.0 if prev is None else float(np.abs(frame - prev).mean()))
            prev = frame
    finally:
        proc.stdout.close()
        if proc.wait():
            raise StepError(f"ffmpeg konnte die Bilder von {path.name} nicht lesen.")
    return np.asarray(values, np.float32)


def find_cuts(motion: np.ndarray) -> list[float]:
    """Harter Schnitt: Sprung deutlich über dem lokalen Mittel und über einer festen Schwelle."""
    cuts: list[float] = []
    for i in range(1, len(motion)):
        local = float(np.median(motion[max(0, i - 15):i + 15]))
        if motion[i] > max(CUT_MIN, CUT_FACTOR * local) and (not cuts or i / FPS - cuts[-1] > CUT_GAP):
            cuts.append(round(i / FPS, 2))
    return cuts


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="Arbeitsordner")
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise StepError(f"Video nicht gefunden: {video}")
    info = probe(video)
    if not info.has_audio:
        raise StepError("Das Video hat keine Tonspur.")
    motion = video_motion(video)
    audio = load_audio(video, SR_LEVEL)
    rms = level_db(audio, HOP_LEVEL)
    prob = speech_probability(load_audio(video), HOP_LEVEL / SR_LEVEL)
    res = {"video": str(video.resolve()), "width": info.width, "height": info.height, "fps": round(info.fps, 3),
           "duration": round(info.duration, 3), "audio_duration": round(len(audio) / SR_LEVEL, 4),
           "motion_fps": FPS, "hop_s": HOP_LEVEL / SR_LEVEL,
           "cuts": find_cuts(motion), "motion": [round(float(v), 2) for v in motion],
           "rms_db": [round(float(v), 1) for v in rms]}
    if prob is not None:
        res["speech_prob"] = [round(float(v), 2) for v in prob[:len(rms)]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "analysis.json", res, indent=None)
    print(f"{info.width}x{info.height}, {info.fps:.2f} fps, {info.duration:.0f} s, {len(res['cuts'])} Schnitte "
          f"-> {out / 'analysis.json'}")


if __name__ == "__main__":
    main()

"""Gesichter finden und der Person folgen, die gerade spricht (Bild-Zuschnitt „face“).

Für Podcasts, Interviews, Vlogs und Vorträge. Erkennung mit YuNet (OpenCV Model Zoo, MIT-Lizenz, liegt in
assets/models) über onnxruntime, das faster-whisper ohnehin mitbringt. Alles lokal, kein weiteres Paket.

Je Einstellung (zwischen harten Schnitten), zehn Bilder je Sekunde:
  1. Gesichter finden und zu Spuren verbinden (eine Spur = eine Person).
  2. Nur Personen zählen, die oft im Bild und nicht viel kleiner als das größte Gesicht sind (kein Poster, kein
     Publikum im Hintergrund).
  3. Eine Person: ruhiger Ausschnitt um ihr Gesicht, Augen im oberen Teil.
     Mehrere Personen: Die Mundbewegung, während gesprochen wird, zeigt, wer spricht. Der Wechsel ist ein harter
     Schnitt, frühestens MIN_HOLD Sekunden nach dem letzten und möglichst an einem Satzanfang.
  4. Ohne Gesicht entscheidet der Aufrufer (render.py nimmt dann den Zuschnitt „content“).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import StepError, VideoInfo, run

MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
NET = 640                          # Modelleingang 640x640 (Bild links oben eingesetzt, Rest schwarz)
STRIDES = (8, 16, 32)
SCORE_MIN, NMS_IOU = 0.7, 0.3      # Mindestsicherheit je Gesicht; Überlappung, ab der zwei Treffer einer sind
SAMPLE_FPS = 10                    # Analysebilder je Sekunde
ANALYSIS_SIDE = 960                # längere Seite der Analysebilder (der Mund braucht etwas Auflösung)
MATCH_IOU, MAX_GAP = 0.25, 8       # dieselbe Person: Überlappung mit dem letzten Treffer, höchstens 0,8 s Lücke
PRESENT = 0.35                     # eine Person zählt ab 35 % der Bilder einer Einstellung
REL_SIZE, MIN_FACE = 0.45, 0.04    # mind. 45 % der Höhe des größten Gesichts und 4 % der Bildhöhe
FACE_SHARE, EYE_LINE, MAX_ZOOM = 0.30, 0.42, 2.0  # Gesicht ~30 % der Ausschnitthöhe, Mitte bei 42 %, höchstens 2x
OTHER_PAD = 0.35                   # andere Personen: so viel ihrer Gesichtsbreite je Seite zusätzlich aussparen
ACT_WINDOW = 1.0                   # Mundbewegung als Median über 1 s
RATIO, ACT_MIN = 1.3, 0.005        # Sprecher: 30 % mehr Mundbewegung als der Nächste und überhaupt Bewegung
                                   # (Median beim Sprechen ~0,01–0,03, still ~0,001)
CONFIRM, MIN_HOLD, SNAP = 0.6, 1.5, 0.7  # Wechsel nach 0,6 s klarer Lage, frühestens 1,5 s nach dem letzten,
                                         # bis 0,7 s verschoben auf einen Satzanfang
SPEECH = 0.5                       # Sprachwahrscheinlichkeit, ab der gesprochen wird
PATCH_W, PATCH_H = 16, 12          # Mund und Augen werden auf diese Größe gemittelt und verglichen
GRAY_EVERY = 5                     # für den Ersatz-Zuschnitt ohne Gesicht reicht jedes 5. Bild (2 je Sekunde)
CENSUS_SAMPLES = 12

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Face:
    """Gesicht in Pixeln: Kasten (x, y, w, h), Sicherheit und fünf Punkte (rechtes Auge, linkes Auge, Nase,
    rechter und linker Mundwinkel, aus Sicht der Person)."""

    x: float
    y: float
    w: float
    h: float
    score: float
    marks: tuple[tuple[float, float], ...] = ()

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], np.float64)

    def scaled(self, k: float) -> Face:
        return Face(self.x * k, self.y * k, self.w * k, self.h * k, self.score,
                    tuple((px * k, py * k) for px, py in self.marks))


# ---------- Erkennung ----------

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Überlappung (IoU) eines Kastens a = (x, y, w, h) mit jedem Kasten in b (n, 4)."""
    b = np.atleast_2d(b)
    w = np.clip(np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[1] + a[3], b[:, 1] + b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = w * h
    return inter / np.maximum(a[2] * a[3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = NMS_IOU) -> list[int]:
    """Doppelte Treffer entfernen: der sicherste bleibt, stark überlappende fallen weg."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        k = int(order[0])
        keep.append(k)
        rest = order[1:]
        order = rest[iou(boxes[k], boxes[rest]) <= threshold] if rest.size else rest
    return keep


def decode(outputs: Sequence[np.ndarray], size: int = NET, threshold: float = SCORE_MIN) -> list[Face]:
    """YuNet-Ausgaben (cls, obj, bbox, kps je Schrittweite 8/16/32) in Gesichter umrechnen, wie cv::FaceDetectorYN."""
    n = len(STRIDES)
    if len(outputs) != 4 * n:
        raise StepError(f"Gesichtsmodell liefert {len(outputs)} statt {4 * n} Ausgaben (falsche Modelldatei?)")
    boxes, scores, marks = [], [], []
    for i, stride in enumerate(STRIDES):
        cols = size // stride
        score = np.sqrt(np.clip(outputs[i].reshape(-1), 0, 1) * np.clip(outputs[i + n].reshape(-1), 0, 1))
        keep = np.flatnonzero(score >= threshold)
        if keep.size == 0:
            continue
        bbox = outputs[i + 2 * n].reshape(-1, 4)[keep].astype(np.float64)
        kps = outputs[i + 3 * n].reshape(-1, 5, 2)[keep].astype(np.float64)
        row, col = np.divmod(keep, cols)
        cell = np.stack([col, row], axis=1).astype(np.float64)
        centre = (cell + bbox[:, :2]) * stride
        size_wh = np.exp(np.minimum(bbox[:, 2:], 10.0)) * stride
        boxes.append(np.concatenate([centre - size_wh / 2, size_wh], axis=1))
        marks.append((kps + cell[:, None, :]) * stride)
        scores.append(score[keep])
    if not boxes:
        return []
    all_boxes, all_scores, all_marks = np.concatenate(boxes), np.concatenate(scores), np.concatenate(marks)
    return [Face(*(float(v) for v in all_boxes[k]), float(all_scores[k]),
                 tuple((float(px), float(py)) for px, py in all_marks[k]))
            for k in nms(all_boxes, all_scores)]


class FaceDetector:
    """YuNet über onnxruntime. detect(bgr) nimmt ein Bild (h, w, 3) uint8 in BGR und gibt Gesichter in dessen Pixeln."""

    def __init__(self, model: Path = MODEL) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise StepError("Für den Zuschnitt „face“ fehlt onnxruntime. Es kommt mit faster-whisper: "
                            "pip install -r requirements.txt") from exc
        if not model.is_file():
            raise StepError(f"Gesichtsmodell fehlt: {model}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # beschädigte oder unvollständige Modelldatei (onnxruntime hat eigene Fehlertypen)
            raise StepError(f"Gesichtsmodell nicht lesbar ({model.name}): {exc}") from exc
        self._input = self._session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        k = NET / max(h, w)
        sw, sh = min(NET, max(1, round(w * k))), min(NET, max(1, round(h * k)))
        small = bgr if (sw, sh) == (w, h) else np.asarray(Image.fromarray(bgr).resize((sw, sh), Image.BILINEAR))
        blob = np.zeros((1, 3, NET, NET), np.float32)
        blob[0, :, :sh, :sw] = small.transpose(2, 0, 1)
        try:
            outputs = self._session.run(None, {self._input: blob})
        except Exception as exc:
            raise StepError(f"Gesichtserkennung fehlgeschlagen: {exc}") from exc
        return [f.scaled(w / sw) for f in decode(outputs)]


@lru_cache(maxsize=1)
def detector() -> FaceDetector:
    return FaceDetector()


# ---------- Mundbewegung ----------

def region(img: Image.Image, cx: float, cy: float, half_w: float, up: float, down: float) -> np.ndarray | None:
    """Bildbereich um (cx, cy), auf PATCH_W x PATCH_H gemittelt und auf Helligkeit und Kontrast normiert.
    None, wenn er zu klein ist oder über den Bildrand ragt."""
    box = (cx - half_w, cy - up * half_w, cx + half_w, cy + down * half_w)
    if half_w < 2 or box[0] < 0 or box[1] < 0 or box[2] > img.width or box[3] > img.height:
        return None
    patch = np.asarray(img.resize((PATCH_W, PATCH_H), Image.BILINEAR, box=box), np.float32)
    return (patch - patch.mean()) / (patch.std() + 8.0)


def mouth_and_eyes(img: Image.Image, face: Face) -> tuple[np.ndarray, np.ndarray] | None:
    """Mundbereich (mit Kinn, damit das Öffnen zählt) und Augenbereich als Vergleich für Kopfbewegung."""
    if len(face.marks) != 5:
        return None
    (rex, rey), (lex, ley), _, (rmx, rmy), (lmx, lmy) = face.marks
    mouth = region(img, (rmx + lmx) / 2, (rmy + lmy) / 2, max(abs(lmx - rmx) * 0.8, face.w * 0.2), 0.6, 1.1)
    eyes = region(img, (rex + lex) / 2, (rey + ley) / 2, max(abs(lex - rex) * 0.9, face.w * 0.3), 0.5, 0.5)
    return None if mouth is None or eyes is None else (mouth, eyes)


def mouth_motion(prev: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray]) -> float:
    """Wie stark sich der Mund bewegt hat, abzüglich der Bewegung um die Augen (Kopf, Wackeln der Punkte)."""
    return max(0.0, float(np.abs(cur[0] - prev[0]).mean() - np.abs(cur[1] - prev[1]).mean()))


# ---------- Spuren ----------

@dataclass(eq=False)
class Track:
    """Eine Person innerhalb einer Einstellung; wird beim Durchlaufen der Bilder fortgeschrieben."""

    faces: dict[int, Face] = field(default_factory=dict)    # Bildnummer -> Gesicht (Pixel des Analysebilds)
    motion: dict[int, float] = field(default_factory=dict)  # Bildnummer -> Mundbewegung seit dem Bild davor
    last: int = -10 ** 9
    regions: tuple[np.ndarray, np.ndarray] | None = None   # Mund und Augen im letzten Bild

    def add(self, k: int, face: Face, regions: tuple[np.ndarray, np.ndarray] | None) -> None:
        if regions is not None and self.regions is not None and k - self.last == 1:
            self.motion[k] = mouth_motion(self.regions, regions)
        self.faces[k] = face
        self.last = k
        self.regions = regions

    def median_height(self) -> float:
        return float(np.median([f.h for f in self.faces.values()]))


def assign(tracks: list[Track], faces: list[Face], k: int) -> list[tuple[Track, Face]]:
    """Ordnet die Gesichter von Bild k Spuren zu (größte Überlappung mit dem letzten Treffer, höchstens MAX_GAP
    Bilder her). Gesichter ohne passende Spur bekommen eine neue. Ändert die Spuren nicht."""
    candidates = [t for t in tracks if 0 < k - t.last <= MAX_GAP]
    pairs: list[tuple[Track, Face]] = []
    for face in sorted(faces, key=lambda f: -f.w * f.h):
        overlap = [float(iou(face.box, t.faces[t.last].box)[0]) for t in candidates]
        if overlap and max(overlap) >= MATCH_IOU:
            pairs.append((candidates.pop(int(np.argmax(overlap))), face))
        else:
            pairs.append((Track(), face))
    return pairs


def relevant(tracks: list[Track], n_frames: int, frame_h: float) -> list[Track]:
    """Personen, die zählen: oft genug im Bild, nicht viel kleiner als das größte Gesicht und nicht winzig."""
    seen = [t for t in tracks if n_frames and len(t.faces) >= PRESENT * n_frames]
    if not seen:
        return []
    sizes = [t.median_height() for t in seen]
    floor = max(REL_SIZE * max(sizes), MIN_FACE * frame_h)
    return [t for t, s in zip(seen, sizes) if s >= floor]


# ---------- Wer spricht ----------

def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Gleitender Median je Zeile (ein kurzes Lachen oder Nicken zählt nicht); Lücken (NaN) zählen nicht, ein
    Fenster ganz ohne Wert bleibt NaN."""
    half = window // 2
    padded = np.pad(values, ((0, 0), (half, window - 1 - half)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Fenster nur aus NaN -> NaN
        return np.nanmedian(windows, axis=2)


def snap(t: float, starts: Sequence[float], lo: float, hi: float) -> float:
    """Nächster Satzanfang höchstens SNAP von t entfernt (zwischen lo und hi), sonst t."""
    near = [s for s in starts if abs(s - t) <= SNAP and lo <= s <= hi]
    return min(near, key=lambda s: abs(s - t)) if near else t


def speaker_turns(times: Sequence[float], motion: np.ndarray, speaking: Sequence[bool], starts: Sequence[float],
                  first: int) -> list[tuple[float, int]]:
    """Wer wann im Bild ist: [(ab Sekunde, Person)]. motion: (Personen, Bilder) Mundbewegung, NaN = nicht zu sehen.

    Ein Wechsel braucht CONFIRM Sekunden klare Lage (RATIO mehr Mundbewegung als der Nächste, während gesprochen
    wird), kommt frühestens MIN_HOLD nach dem letzten und rutscht auf einen Satzanfang, wenn einer in SNAP liegt.
    Wer am Anfang klar spricht, beginnt; sonst first (das größte Gesicht)."""
    ts = np.asarray(times, float)
    n = len(ts)
    if n == 0:
        return []
    filled = np.nan_to_num(smooth(np.asarray(motion, float), max(1, round(ACT_WINDOW * SAMPLE_FPS))), nan=0.0)
    order = np.argsort(-filled, axis=0, kind="stable")
    cols = np.arange(n)
    top = filled[order[0], cols]
    second = filled[order[1], cols] if filled.shape[0] > 1 else np.zeros(n)
    clear = np.asarray(speaking, bool) & (top >= ACT_MIN) & (top >= RATIO * second)
    cand = np.where(clear, order[0], -1)
    early = cand[(ts < ts[0] + 2 * MIN_HOLD) & (cand >= 0)]
    current = int(np.bincount(early).argmax()) if early.size else first
    turns = [(float(ts[0]), current)]
    need = max(1, round(CONFIRM * SAMPLE_FPS))
    who, start, count = -1, 0, 0
    for k in range(n):
        c = int(cand[k])
        if c == current:
            who, count = -1, 0
            continue
        if c < 0:
            continue  # Pause oder unklar: die laufende Beobachtung bleibt stehen
        if c != who:
            who, start, count = c, k, 0
        count += 1
        last_t = turns[-1][0]
        if count >= need and ts[k] - last_t >= MIN_HOLD:
            t = snap(max(float(ts[start]), last_t + MIN_HOLD), starts, lo=last_t + MIN_HOLD / 2, hi=float(ts[-1]))
            turns.append((t, c))
            current, who, count = c, -1, 0
    return turns


# ---------- Ausschnitt ----------

def face_box(faces: Sequence[Face], ww: float, wh: float, aspect: float, others: Sequence[Face] = ()) -> Box:
    """Ausschnitt (x, y, w, h) mit Seitenverhältnis aspect um die Gesichter einer Person (in Pixeln von ww x wh).

    Das Gesicht füllt etwa FACE_SHARE der Höhe (höchstens MAX_ZOOM-fach vergrößert), seine Mitte liegt bei
    EYE_LINE der Höhe. Bewegt sich die Person, umfasst der Ausschnitt den Bereich, in dem sie meistens ist.
    others: Gesichter der anderen Personen; der Ausschnitt spart sie samt Haaren aus, wenn es mit höchstens
    MAX_ZOOM geht (sonst kein halbes Gesicht am Rand, sondern die Person mittig)."""
    left = float(np.percentile([f.x for f in faces], 10))
    right = float(np.percentile([f.x + f.w for f in faces], 90))
    fh = float(np.median([f.h for f in faces]))
    fw = float(np.median([f.w for f in faces]))
    cy = float(np.median([f.y + f.h / 2 for f in faces]))
    centre = (left + right) / 2
    ch = min(wh, max(wh / MAX_ZOOM, fh / FACE_SHARE, (right - left + fw) / aspect))
    cw = min(ww, ch * aspect)
    # freier Bereich zwischen den Nachbarn (Gesicht plus OTHER_PAD seiner Breite für Haare und Ohren)
    lo = max([o.x + o.w * (1 + OTHER_PAD) for o in others if o.x + o.w / 2 < centre] + [0.0])
    hi = min([o.x - o.w * OTHER_PAD for o in others if o.x + o.w / 2 >= centre] + [float(ww)])
    min_w = min(ww, wh / MAX_ZOOM * aspect)
    if others and cw > hi - lo and hi - lo >= max(min_w, right - left + fw / 2):
        cw = hi - lo
    ch = cw / aspect
    x = centre - cw / 2
    if cw <= hi - lo:
        x = min(max(x, lo), hi - cw)
    x = min(max(0.0, x), ww - cw)
    y = min(max(0.0, cy - ch * EYE_LINE), wh - ch)
    return (x, y, cw, ch)


def analysis_size(ww: int, wh: int) -> tuple[int, int]:
    """Größe der Analysebilder: längere Seite höchstens ANALYSIS_SIDE, gerade Zahlen."""
    k = min(1.0, ANALYSIS_SIDE / max(ww, wh))
    return max(2, round(ww * k / 2) * 2), max(2, round(wh * k / 2) * 2)


# ---------- Ablauf ----------

@dataclass(eq=False)
class ShotScan:
    """Was in einer Einstellung gesehen wurde: Zeiten, Spuren und kleine Graubilder für den Ersatz-Zuschnitt."""

    t0: float
    t1: float
    times: list[float] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    gray: list[np.ndarray] = field(default_factory=list)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return (bgr.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)).astype(np.uint8)


def scan_piece(src: str, a: float, b: float, shots: list[tuple[float, float]], aw: int, ah: int,
               det: FaceDetector) -> list[ShotScan]:
    """Liest das Stück [a, b] mit SAMPLE_FPS Bildern je Sekunde und schreibt Gesichter und Mundbewegung je
    Einstellung mit. Die Bilder werden einzeln gelesen; für den Ersatz-Zuschnitt bleibt nur jedes GRAY_EVERY-te
    als kleines Graubild im Speicher."""
    scans = [ShotScan(s0, s1) for s0, s1 in shots]
    frame_bytes = aw * ah * 3
    i = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src, "-vf",
                                 f"fps={SAMPLE_FPS},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                t = round(a + i / SAMPLE_FPS, 6)  # gerundet: ein Bild genau auf einem Schnitt gehört nach dem Schnitt
                i += 1
                shot = next((s for s in scans if s.t0 <= t < s.t1), scans[-1])
                bgr = np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
                gray = to_gray(bgr)
                k = len(shot.times)
                shot.times.append(t)
                if k % GRAY_EVERY == 0:
                    shot.gray.append(gray[::2, ::2].copy())
                img = Image.fromarray(gray)
                for track, face in assign(shot.tracks, det.detect(bgr), k):
                    if not any(track is known for known in shot.tracks):
                        shot.tracks.append(track)
                    track.add(k, face, mouth_and_eyes(img, face))
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            code = proc.wait()
        if code or i == 0:
            err.seek(0)
            detail = err.read().decode("utf-8", "replace")[-400:].strip()
            raise StepError(f"ffmpeg konnte die Bilder bei {a:.1f}–{b:.1f} s nicht lesen. {detail}".strip())
    return scans


def plan_shot(shot: ShotScan, ww: int, wh: int, aw: int, ah: int, aspect: float, speaking: Callable[[float], bool],
              starts: Sequence[float], fallback: Callable[[np.ndarray, float], Box]
              ) -> list[tuple[float, float, Box]]:
    """Ausschnitte einer Einstellung: ohne Person der Ersatz, mit einer Person ein Ausschnitt, mit mehreren einer je
    Sprecherabschnitt."""
    people = relevant(shot.tracks, len(shot.times), ah)
    if not people:
        frames = np.stack(shot.gray) if shot.gray else np.zeros((0, ah // 2, aw // 2), np.uint8)
        return [(shot.t0, shot.t1, fallback(frames, ww / frames.shape[2]))]
    scale = ww / aw
    if len(people) == 1:
        return [(shot.t0, shot.t1, face_box([f.scaled(scale) for f in people[0].faces.values()], ww, wh, aspect))]
    motion = np.full((len(people), len(shot.times)), np.nan)
    for p, track in enumerate(people):
        for k, value in track.motion.items():
            motion[p, k] = value
    first = int(np.argmax([t.median_height() for t in people]))
    turns = speaker_turns(shot.times, motion, [speaking(t) for t in shot.times], starts, first)
    ends = [t for t, _ in turns[1:]] + [shot.t1]
    typical = [typical_face(t).scaled(scale) for t in people]
    crops = []
    for j, ((t0, who), t1) in enumerate(zip(turns, ends)):
        t0 = shot.t0 if j == 0 else t0
        track = people[who]
        seen = [f for k, f in track.faces.items() if t0 <= shot.times[k] < t1] or list(track.faces.values())
        others = [f for p, f in enumerate(typical) if p != who]
        crops.append((t0, t1, face_box([f.scaled(scale) for f in seen], ww, wh, aspect, others)))
    return crops


def typical_face(track: Track) -> Face:
    """Mittlere Lage und Größe einer Person über die Einstellung (Median je Wert)."""
    fs = list(track.faces.values())
    return Face(*(float(np.median([getattr(f, a) for f in fs])) for a in ("x", "y", "w", "h")), score=1.0)


def face_crops(src: str, pieces: Sequence[Sequence[float]], cuts: Sequence[float], ww: int, wh: int, aspect: float,
               speaking: Callable[[float], bool], starts: Sequence[float],
               fallback: Callable[[np.ndarray, float], Box]) -> list[tuple[float, float, Box]]:
    """Ausschnitte wie render.shot_crops: [(Start, Ende, (x, y, w, h))] in Pixeln der Arbeitsgröße ww x wh, je
    Einstellung und je Sprecherwechsel. Einstellungen ohne Person gehen an fallback(Graubilder, Maßstab)."""
    det = detector()
    aw, ah = analysis_size(ww, wh)
    crops: list[tuple[float, float, Box]] = []
    for a, b in pieces:
        bounds = [a] + [c for c in cuts if a + 0.05 < c < b - 0.05] + [b]
        shots = list(zip(bounds, bounds[1:]))
        for shot in scan_piece(src, a, b, shots, aw, ah, det):
            if not shot.times and crops:  # Einstellung kürzer als ein Analysebild: Ausschnitt davor weiter
                crops.append((shot.t0, shot.t1, crops[-1][2]))
            else:
                crops += plan_shot(shot, ww, wh, aw, ah, aspect, speaking, starts, fallback)
    return crops


def face_census(src: str, info: VideoInfo, samples: int = CENSUS_SAMPLES) -> dict:
    """Stichprobe für die Wahl des Zuschnitts: in wie vielen Bildern eine Person zu sehen ist, wie viele höchstens."""
    det = detector()
    aw, ah = analysis_size(info.width, info.height)
    counts = []
    for i in range(samples):
        t = info.duration * (i + 0.5) / samples
        raw = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", "-vf",
                   f"scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]).stdout
        if len(raw) < aw * ah * 3:
            continue
        faces = det.detect(np.frombuffer(raw[:aw * ah * 3], np.uint8).reshape(ah, aw, 3))
        counts.append(sum(f.h >= MIN_FACE * ah for f in faces))
    return {"samples": len(counts), "with_face": sum(c > 0 for c in counts), "max_faces": max(counts, default=0)}

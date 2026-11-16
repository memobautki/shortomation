"""Rendert Clips aus clips.scored.json als Hochkant-Short 1080x1920 (Schritt 6).

    python -I render.py --dir <ordner> [--ids k01,k04 | --top 5] [--layout content|full|center] [--keep]

Aufbau: oben Reihe und Hook, Mitte das Bild, unten große Untertitel im Takt der gemessenen Wörter.
Bild-Zuschnitt (--layout oder "layout" in clips.json):
  content  folgt dem hellen Inhalt auf dunklem Grund (Tafeln, Dokumente, Folien), je Einstellung neu
  full     ganzes Bild, nichts abgeschnitten
  center   füllt die Bildfläche, schneidet links und rechts ab (gut für eine Person in der Mitte)
Ton aus dem Original, Stücke mit kurzen Blenden in Pausen, Tempo nach "speed", Lautheit -14 LUFS.
Schreibt <ordner>/out/<id>.mp4 und <id>_cover.jpg. Braucht ffmpeg, numpy, Pillow.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I lässt den Skriptordner weg
from common import FINAL_FADE, FIRST_FADE, PIECE_FADE, StepError, VideoInfo, probe, read_json, run, setup_console

SKILL_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = SKILL_DIR / "assets" / "fonts"
OW, OH, FPS = 1080, 1920, 30
WORK_MAX_W, WORK_MAX_H = 1920, 1080   # Arbeitsgröße des Quellbilds (größere Quellen werden verkleinert)
LAYOUTS = ("content", "full", "center")

STYLE = {
    "bg": "#0B0C0E", "text": "#ECEAE4", "dim": "#8A8E93", "accent": "#FF4A2E",
    "accent_pattern": r"\d|€|\$|%|Milliard|Million",
    "fonts": {"hook": "stencil-extrabold.woff2", "label": "mono-semibold.woff2",
              "caption": "sans-bold.woff2", "foot": "mono-semibold.woff2"},
    "hook_size": 118, "hook_spacing": 3, "label_size": 30, "label_spacing": 4,
    "cap_size": 68, "foot_size": 26, "foot_spacing": 3,
    "panel_top": 520, "panel_bottom": 1400, "cap_y": 1545, "foot_y": 1690, "label_y": 200, "margin": 64,
    "ignore_edges": [0.09, 0.015, 0.09, 0.015],  # oben, rechts, unten, links: Randanteil, der nicht als Inhalt zählt
    "push_in": 0.06,  # Zoom-Zuwachs über eine Einstellung (0 = aus)
}
CONTENT_L, PAD = 70, 72            # Inhalt = heller als CONTENT_L; Rand um den Inhalt in px (Arbeitsgröße)
MASS_CUT, FULL_AREA, MAX_ZOOM = 0.004, 0.72, 1.7
CAP_MAX_CHARS, CAP_MAX_WORDS, CAP_HOLD = 20, 4, 0.25


COLOR_KEYS = ("bg", "text", "dim", "accent")
INT_KEYS = {"hook_size": (20, 300), "label_size": (10, 120), "cap_size": (20, 200), "foot_size": (10, 120),
            "panel_top": (0, OH), "panel_bottom": (0, OH), "cap_y": (130, OH - 70), "foot_y": (0, OH),
            "label_y": (0, OH), "margin": (0, 300)}
FLOAT_KEYS = {"hook_spacing": (0, 40), "label_spacing": (0, 40), "foot_spacing": (0, 40), "push_in": (0, 0.3)}


def validate_style(style: object) -> dict:
    """Prüft „style“ aus clips.json und gibt den vollständigen Stil zurück (Fehler mit Feldname)."""
    if not isinstance(style, dict):
        raise StepError("„style“ in clips.json muss ein Objekt sein (siehe SKILL.md, Stil anpassen).")
    unknown = set(style) - set(STYLE)
    if unknown:
        raise StepError(f"Unbekannte Stil-Felder: {', '.join(sorted(unknown))}. Erlaubt: {', '.join(STYLE)}")
    if not isinstance(style.get("fonts", {}), dict):
        raise StepError(f"style.fonts muss ein Objekt sein, z. B. {{\"hook\": \"MeineSchrift.ttf\"}}.")
    st = {**STYLE, **style, "fonts": {**STYLE["fonts"], **style.get("fonts", {})}}
    for k in COLOR_KEYS:
        if not isinstance(st[k], str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", st[k]):
            raise StepError(f"style.{k} muss eine Farbe wie \"#FF4A2E\" sein, gefunden: {st[k]!r}")
    for k, (lo, hi) in INT_KEYS.items():
        if isinstance(st[k], bool) or not isinstance(st[k], int) or not lo <= st[k] <= hi:
            raise StepError(f"style.{k} muss eine ganze Zahl zwischen {lo} und {hi} sein, gefunden: {st[k]!r}")
    for k, (lo, hi) in FLOAT_KEYS.items():
        if isinstance(st[k], bool) or not isinstance(st[k], (int, float)) or not lo <= st[k] <= hi:
            raise StepError(f"style.{k} muss eine Zahl zwischen {lo} und {hi} sein, gefunden: {st[k]!r}")
    if st["panel_bottom"] - st["panel_top"] < 200:
        raise StepError("style.panel_bottom muss mindestens 200 px unter style.panel_top liegen.")
    edges = st["ignore_edges"]
    if (not isinstance(edges, list) or len(edges) != 4
            or not all(isinstance(e, (int, float)) and not isinstance(e, bool) and 0 <= e <= 0.4 for e in edges)):
        raise StepError("style.ignore_edges muss vier Zahlen zwischen 0 und 0,4 enthalten (oben, rechts, unten, links).")
    try:
        re.compile(st["accent_pattern"])
    except (re.error, TypeError) as exc:
        raise StepError(f"style.accent_pattern ist kein gültiger regulärer Ausdruck: {exc}") from exc
    if set(st["fonts"]) != set(STYLE["fonts"]) or not all(isinstance(v, str) and v for v in st["fonts"].values()):
        raise StepError(f"style.fonts braucht Dateinamen (Text) für {', '.join(STYLE['fonts'])}.")
    return st


def hexrgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def work_size(info: VideoInfo) -> tuple[int, int]:
    """Arbeitsgröße: Seitenverhältnis bleibt, höchstens WORK_MAX_W x WORK_MAX_H, gerade Pixelzahlen."""
    s = min(1.0, WORK_MAX_W / info.width, WORK_MAX_H / info.height)
    return max(2, int(info.width * s) // 2 * 2), max(2, int(info.height * s) // 2 * 2)


# ---------- Schrift ----------

def resolve_font(name: str, style_dir: Path | None) -> Path:
    for base in [style_dir, FONT_DIR] if style_dir else [FONT_DIR]:
        p = Path(name) if Path(name).is_absolute() else Path(base) / name
        if p.is_file():
            return p
    raise StepError(f"Schrift nicht gefunden: {name} (gesucht in {style_dir or ''} und {FONT_DIR})")


@lru_cache(maxsize=64)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def preflight_fonts(st: dict, style_dir: Path | None) -> None:
    """Jede Schrift vor dem ersten Render finden und laden: Fehler sofort und mit Feldname statt nach Minuten."""
    for role, name in st["fonts"].items():
        path = resolve_font(name, style_dir)
        try:
            font(str(path), 40).getlength("ÄÖÜß€ Test 1,9")
        except OSError as exc:
            raise StepError(f"style.fonts.{role}: {path} lässt sich nicht als Schrift laden ({exc}).") from exc


def text_width(text: str, f: ImageFont.FreeTypeFont, spacing: float) -> float:
    if not spacing:
        return f.getlength(text)
    return sum(f.getlength(ch) for ch in text) + spacing * max(0, len(text) - 1)


def fit_size(text: str, path: str, size: int, spacing: float, maxw: float) -> int:
    while size > 8 and text_width(text, font(path, size), spacing) > maxw:
        size -= 2
    return size


def draw_text(draw: ImageDraw.ImageDraw, text: str, x: float, y: float, path: str, size: int, color: str,
              spacing: float = 0, align: str = "left", maxw: float | None = None) -> float:
    """Text auf Grundlinie y; x ist links oder Mitte. Gibt die Breite zurück."""
    if maxw:
        size = fit_size(text, path, size, spacing, maxw)
    f = font(path, size)
    w = text_width(text, f, spacing)
    x0 = x - w / 2 if align == "center" else x
    if not spacing:
        draw.text((x0, y), text, font=f, fill=color, anchor="ls")
        return w
    for ch in text:
        draw.text((x0, y), ch, font=f, fill=color, anchor="ls")
        x0 += f.getlength(ch) + spacing
    return w


def hook_lines(hook: str) -> list[str]:
    """Hook in Großbuchstaben, bei Bedarf in zwei etwa gleich lange Zeilen (bevorzugt nach Satzzeichen)."""
    hook = hook.upper().strip()
    words = hook.split()
    if len(hook) <= 16 or len(words) < 2:
        return [hook]
    best, cut = None, 1
    for k in range(1, len(words)):
        l1, l2 = " ".join(words[:k]), " ".join(words[k:])
        cost = abs(len(l1) - len(l2)) - (3 if re.search(r"[.:?!,–]$", l1) else 0)
        if best is None or cost < best:
            best, cut = cost, k
    return [" ".join(words[:cut]), " ".join(words[cut:])]


def base_layer(clip: dict, cj: dict, st: dict, fonts: dict[str, str]) -> np.ndarray:
    """Hintergrund mit Reihe, Hook, Akzentbalken und Fußzeile (RGB)."""
    img = Image.new("RGB", (OW, OH), hexrgb(st["bg"]))
    draw = ImageDraw.Draw(img)
    m, maxw = st["margin"], OW - 2 * st["margin"]
    draw_text(draw, clip.get("label", ""), m, st["label_y"], fonts["label"], st["label_size"], st["dim"],
              st["label_spacing"], maxw=maxw)
    lines = hook_lines(clip["hook"])
    size = min(fit_size(ln, fonts["hook"], st["hook_size"], st["hook_spacing"], maxw) for ln in lines)
    y = st["label_y"] + 30 + size
    for ln in lines:
        draw_text(draw, ln, m - 4, y, fonts["hook"], size, st["text"], st["hook_spacing"])
        y += size
    draw.rectangle([m, y - size + 34, m + 240, y - size + 43], fill=st["accent"])
    foot = f"{cj.get('channel', '')}  ·  {cj.get('cta', '')}".strip(" ·")
    if foot:
        draw_text(draw, foot, OW / 2, st["foot_y"], fonts["foot"], st["foot_size"], st["dim"], st["foot_spacing"],
                  align="center", maxw=maxw)
    return np.asarray(img).copy()


def caption_layer(words: list[str], st: dict, fonts: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Untertitelzeile als (RGB, Alpha) im Streifen 1080x200, Grundlinie bei y=130, mit weichem Schatten."""
    accent = re.compile(st["accent_pattern"], re.I)
    path, maxw = fonts["caption"], OW - 2 * st["margin"]
    text = " ".join(words)
    size = fit_size(text, path, st["cap_size"], 0, maxw)
    f = font(path, size)
    space = f.getlength(" ")
    total = sum(f.getlength(w) for w in words) + space * (len(words) - 1)
    layer = Image.new("RGBA", (OW, 200), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (OW, 200), (0, 0, 0, 0))
    dl, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    x = (OW - total) / 2
    for w in words:
        color = st["accent"] if accent.search(w) else st["text"]
        ds.text((x + 3, 134), w, font=f, fill=(0, 0, 0, 220), anchor="ls")
        dl.text((x, 130), w, font=f, fill=color, anchor="ls")
        x += f.getlength(w) + space
    out = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(4)), layer)
    arr = np.asarray(out).astype(np.float32) / 255
    return arr[..., :3] * 255, arr[..., 3:4]


def caption_chunks(sents: list[dict]) -> list[dict]:
    """Gemessene Wörter -> kurze Einblendungen (höchstens 4 Wörter / 20 Zeichen), Zeiten aus den Wortzeiten."""
    chunks: list[dict] = []
    for s in sents:
        cur: list[list] = []
        for tok in s["tokens"]:
            if cur and (len(" ".join(t[0] for t in cur + [tok])) > CAP_MAX_CHARS or len(cur) >= CAP_MAX_WORDS):
                chunks.append({"t0": cur[0][1], "t1": cur[-1][2], "words": [t[0] for t in cur]})
                cur = []
            cur.append(tok)
            if re.search(r"[.:?!]$", tok[0]) and len(" ".join(t[0] for t in cur)) > 8:
                chunks.append({"t0": cur[0][1], "t1": cur[-1][2], "words": [t[0] for t in cur]})
                cur = []
        if cur:
            chunks.append({"t0": cur[0][1], "t1": cur[-1][2], "words": [t[0] for t in cur]})
    for a, b in zip(chunks, chunks[1:]):  # Einblendung steht bis zur nächsten, aber nicht ewig in Pausen
        a["t1"] = max(a["t1"], min(b["t0"], a["t1"] + CAP_HOLD + 0.5))
    if chunks:
        chunks[-1]["t1"] += CAP_HOLD
    return chunks


# ---------- Zuschnitt ----------

def shot_crops(src: str, pieces: list[list[float]], cuts: list[float], st: dict, layout: str, ww: int, wh: int
               ) -> list[tuple[float, float, tuple[float, float, float, float]]]:
    """Je Einstellung (zwischen harten Schnitten) ein Ausschnitt (x, y, w, h) in Pixeln der Arbeitsgröße."""
    panel_h = st["panel_bottom"] - st["panel_top"]
    crops = []
    for a, b in pieces:
        bounds = [a] + [c for c in cuts if a + 0.05 < c < b - 0.05] + [b]
        if layout != "content":
            box = (0.0, 0.0, float(ww), float(wh)) if layout == "full" else center_box(ww, wh, OW / panel_h)
            crops += [(s0, s1, box) for s0, s1 in zip(bounds, bounds[1:])]
            continue
        sw, sh = max(2, ww // 4 // 2 * 2), max(2, wh // 4 // 2 * 2)
        raw = run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src, "-vf",
                   f"fps=10,scale={sw}:{sh},format=gray", "-f", "rawvideo", "-"]).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, sh, sw)
        for s0, s1 in zip(bounds, bounds[1:]):
            i0 = int((s0 - a) * 10)
            i1 = max(i0 + 1, int((s1 - a) * 10))
            crops.append((s0, s1, content_box(frames[i0:i1], ww / sw, ww, wh, OW / panel_h, st)))
    return crops


def center_box(ww: int, wh: int, aspect: float) -> tuple[float, float, float, float]:
    """Größter Ausschnitt mit Seitenverhältnis aspect in der Bildmitte."""
    w = min(ww, wh * aspect)
    h = w / aspect
    return ((ww - w) / 2, (wh - h) / 2, w, h)


def content_box(frames: np.ndarray, scale: float, ww: int, wh: int, a_min: float, st: dict
                ) -> tuple[float, float, float, float]:
    if len(frames) == 0:
        return (0.0, 0.0, float(ww), float(wh))
    mask = (frames > CONTENT_L).astype(np.float32)
    t, r, b, l = st["ignore_edges"]  # Randstreifen mit Senderleiste/Kopfzeile zählen nicht als Inhalt
    hh, w_ = mask.shape[1:]
    mask[:, :int(t * hh)] = 0
    mask[:, hh - int(b * hh):] = 0
    mask[:, :, :int(l * w_)] = 0
    mask[:, :, w_ - int(r * w_):] = 0
    if mask.sum() < 40:  # (fast) leeres Bild
        return (0.0, 0.0, float(ww), float(wh))

    def extent(profile: np.ndarray) -> tuple[int, int]:
        """Bereich mit 99,2 % der Inhaltsmasse: kleine Randelemente (Kopfzeile, Seitenzahl) fallen weg."""
        cum = np.cumsum(profile) / profile.sum()
        return int(np.searchsorted(cum, MASS_CUT)), int(np.searchsorted(cum, 1 - MASS_CUT)) + 1

    xa, xb = extent(mask.sum(axis=(0, 1)))
    ya, yb = extent(mask.sum(axis=(0, 2)))
    x0, x1 = xa * scale - PAD, xb * scale + PAD
    y0, y1 = ya * scale - PAD, yb * scale + PAD
    bw, bh = x1 - x0, y1 - y0
    if bw * bh > FULL_AREA * ww * wh:  # Foto oder Film über das ganze Bild: nicht beschneiden
        return (0.0, 0.0, float(ww), float(wh))
    asp = min(ww / wh, max(a_min, bw / max(1.0, bh)))
    w = max(ww / MAX_ZOOM, bw, bh * asp)
    w = min(ww, w, wh * asp)
    h = w / asp
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (min(max(0.0, cx - w / 2), ww - w), min(max(0.0, cy - h / 2), wh - h), w, h)


# ---------- Bild ----------

def compose_video(clip: dict, src: str, crops: list, chunks: list[dict], base: np.ndarray, caps: list,
                  st: dict, ww: int, wh: int, tmp: Path) -> tuple[Path, int]:
    """Setzt alle Bilder zusammen (30 fps Quellzeit). Gibt (Datei, Bildzahl) zurück.

    Die Bildzahl je Stück folgt der gemeinsamen Zeitachse (gerundete Summe der Stücklängen), damit Bild und Ton
    auch nach vielen Stücken auf weniger als ein halbes Bild genau zusammenpassen."""
    out_path = tmp / "video.mp4"
    log = open(tmp / "encoder.log", "wb")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                            "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE, stderr=log)
    top, panel_h, cap_y = st["panel_top"], st["panel_bottom"] - st["panel_top"], st["cap_y"]
    frame_bytes = ww * wh * 3
    written, timeline, code = 0, 0.0, 1
    dec = None
    try:
        for a, b in clip["cut"]:
            timeline += b - a
            n = max(1, round(timeline * FPS) - written)
            dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-i", src, "-t", f"{n / FPS + 0.1:.3f}",
                                    "-vf", f"fps={FPS},scale={ww}:{wh}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            last = None
            for i in range(n):
                buf = dec.stdout.read(frame_bytes)
                if len(buf) == frame_bytes:
                    last = np.frombuffer(buf, np.uint8).reshape(wh, ww, 3)
                if last is None:
                    raise StepError(f"Kein Bild bei {a:.2f} s im Video.")
                enc.stdin.write(frame_at(last, a + i / FPS, crops, base, chunks, caps, st, top, panel_h, cap_y).tobytes())
                written += 1
            dec.stdout.close()
            dec.wait()
            dec = None
        enc.stdin.close()
        code = enc.wait()
    except BrokenPipeError as exc:
        raise StepError(f"Video-Encoder abgebrochen: {_tail(tmp / 'encoder.log')}") from exc
    finally:
        if dec is not None:  # Abbruch mitten im Stück: Decoder beenden, sonst bleibt er hängen
            dec.kill()
            dec.wait()
        if enc.poll() is None:
            enc.kill()
            enc.wait()
        log.close()
    if code:
        raise StepError(f"Video-Encoder fehlgeschlagen: {_tail(tmp / 'encoder.log')}")
    return out_path, written


def _tail(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", "replace")[-400:]
    except OSError:
        return ""


def frame_at(frame: np.ndarray, t: float, crops: list, base: np.ndarray, chunks: list[dict], caps: list, st: dict,
             top: int, panel_h: int, cap_y: int) -> np.ndarray:
    s0, s1, (x, y, w, h) = next((c for c in crops if c[0] <= t < c[1]), crops[-1])
    # langsamer Push-in je Einstellung: stehende Tafeln bleiben lebendig
    z = 1 + st["push_in"] * min(1.0, max(0.0, t - s0) / max(0.5, s1 - s0))
    dw, dh = w * (1 - 1 / z) / 2, h * (1 - 1 / z) / 2
    scale = min(OW / w, panel_h / h)
    pw, ph = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    panel = Image.fromarray(frame).resize((pw, ph), Image.BILINEAR, box=(x + dw, y + dh, x + w - dw, y + h - dh))
    out = base.copy()
    px, py = (OW - pw) // 2, top + (panel_h - ph) // 2
    out[py:py + ph, px:px + pw] = np.asarray(panel)
    k = next((j for j, c in enumerate(chunks) if c["t0"] <= t < c["t1"]), None)
    if k is not None:
        rgb, al = caps[k]
        y0 = cap_y - 130
        region = out[y0:y0 + 200].astype(np.float32)
        out[y0:y0 + 200] = (rgb * al + region * (1 - al)).astype(np.uint8)
    return out


# ---------- Ton und Ausgabe ----------

def build_audio(clip: dict, src: str, tmp: Path) -> tuple[Path, float]:
    """Tonstücke mit kurzen Blenden (liegen in Pausen), Tempo, Lautheit. Gibt (Datei, Dauer) zurück."""
    parts, labels = [], []
    air = clip.get("air") or [[1.0, 1.0]] * len(clip["cut"])
    for i, (a, b) in enumerate(clip["cut"]):
        d = b - a
        # Blenden nie länger als die gemessene Luft vor dem ersten bzw. nach dem letzten Laut (minus Sicherheit)
        fin = min(FIRST_FADE if i == 0 else PIECE_FADE, max(0.005, air[i][0] - 0.02))
        fout = min(FINAL_FADE if i == len(clip["cut"]) - 1 else PIECE_FADE, max(0.005, air[i][1] - 0.02))
        # aresample first_pts=0: dieselbe Zeitachse wie load_audio (Startversatz der Tonspur wird zu Stille);
        # apad whole_dur: jedes Stück hat genau seine Länge, auch am Dateiende, damit Bild und Ton synchron bleiben
        parts.append(f"[0:a]aresample=async=1:first_pts=0,atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,"
                     f"apad=whole_dur={d:.3f},afade=t=in:d={fin},afade=t=out:st={max(0.0, d - fout):.3f}:d={fout}[a{i}]")
        labels.append(f"[a{i}]")
    speed = float(clip.get("speed", 1.0))
    total = sum(b - a for a, b in clip["cut"]) / speed
    chain = (";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1,atempo={speed},"
             f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[out]")
    out = tmp / "audio.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", chain, "-map", "[out]", "-ac", "2", str(out)])
    return out, total


def render_clip(clip: dict, cj: dict, sents: list[dict], an: dict, d: Path, args: argparse.Namespace) -> Path:
    st = validate_style(cj.get("style") or {})
    style_dir = Path(args.fonts) if args.fonts else None
    fonts = {k: str(resolve_font(v, style_dir)) for k, v in st["fonts"].items()}
    layout = clip.get("layout", args.layout or cj.get("layout", "content"))
    if layout not in LAYOUTS:
        raise StepError(f"{clip['id']}: layout „{layout}“ unbekannt, erlaubt: {', '.join(LAYOUTS)}")
    if layout == "full":
        st["push_in"] = 0.0  # „full“ verspricht das ganze Bild: kein Zoom, der Ränder abschneidet
    src = str((d / cj["video"]).resolve()) if not Path(cj["video"]).is_absolute() else cj["video"]
    if not Path(src).is_file():
        raise StepError(f"Video nicht gefunden: {src} (Feld „video“ in clips.json, relativ zum Arbeitsordner)")
    info = probe(src)
    ww, wh = work_size(info)
    in_clip = [s for s in sents if any(a - 0.05 <= s["t0"] and s["t1"] <= b + 0.05 for a, b in clip["cut"])]
    chunks = caption_chunks(in_clip)
    outdir = d / "out"
    outdir.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"tmp_{clip['id']}_", dir=outdir))
    try:
        base = base_layer(clip, cj, st, fonts)
        caps = [caption_layer(c["words"], st, fonts) for c in chunks]
        crops = shot_crops(src, clip["cut"], an["cuts"], st, layout, ww, wh)
        video, _ = compose_video(clip, src, crops, chunks, base, caps, st, ww, wh, tmp)
        audio, total = build_audio(clip, src, tmp)
        speed = float(clip.get("speed", 1.0))
        final = outdir / f"{clip['id']}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-filter:v",
             f"setpts=PTS/{speed},fps={FPS}", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow",
             "-crf", "19", "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k",
             "-t", f"{total:.3f}", "-movflags", "+faststart", str(final)])
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{min(1.0, total / 2):.2f}", "-i", str(final), "-frames:v", "1",
             "-q:v", "3", str(outdir / f"{clip['id']}_cover.jpg")])
        print(f"{clip['id']}: {final} ({total:.1f} s, {len(crops)} Einstellungen, {len(chunks)} Untertitel, {layout})")
        return final
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Arbeitsordner")
    ap.add_argument("--ids", help="Kommagetrennte Clip-IDs (Standard: die besten --top ohne Content-ID-Warnung)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--layout", choices=LAYOUTS, help="Bild-Zuschnitt für alle Clips (Standard: content)")
    ap.add_argument("--fonts", help="Ordner mit eigenen Schriften (Dateinamen in style.fonts)")
    ap.add_argument("--keep", action="store_true", help="Zwischendateien behalten")
    args = ap.parse_args()
    d = Path(args.dir).resolve()
    cj = read_json(d / "clips.scored.json")
    sents = read_json(d / "sentences.json")["sentences"]
    an = read_json(d / "analysis.json")
    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
        unknown = set(want) - {c["id"] for c in cj["clips"]}
        if unknown:
            raise StepError(f"Unbekannte Clip-IDs: {', '.join(sorted(unknown))}")
        clips = [c for c in cj["clips"] if c["id"] in want]
    else:
        clips = [c for c in cj["clips"] if not any(f.startswith(("Content-ID", "überlappt")) for f in c["flags"])]
        clips = clips[:max(0, args.top)]
    if not clips:
        raise StepError("Kein Clip zum Rendern (alle mit Content-ID- oder Überlappungs-Warnung?). Nutze --ids.")
    preflight_fonts(validate_style(cj.get("style") or {}), Path(args.fonts) if args.fonts else None)
    for c in clips:
        render_clip(c, cj, sents, an, d, args)


if __name__ == "__main__":
    main()

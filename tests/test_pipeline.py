"""Ende-zu-Ende-Test mit einem künstlichen Video (braucht ffmpeg, keine Spracherkennung, nichts aus dem Netz).

Das Video ist 1280x720 bei 25 fps (also nicht 1920x1080/30), der „Satz“ endet laut Wortzeiten bei 2,0 s, klingt
aber leise bis 2,5 s nach. Geprüft wird: Der fertige Short hat 1080x1920, die richtige Länge, und der leise
Ausklang ist noch drin (Regression „letztes Wort fehlt“).

    python -m unittest discover -s tests -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "shortomation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import HOP_LEVEL, SR_LEVEL, level_db, load_audio, probe  # noqa: E402

SR = 48000


def tone(seconds, db, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds, db, seed=1):
    rng = np.random.default_rng(seed)
    return (10 ** (db / 20) * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg fehlt")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shorts-test-"))
        d = cls.tmp
        # Ton: Satz 1 laut 1,0–2,0 s, geflüstertes Wortende 2,0–2,5 s, Pause, Satz 2 laut 4,0–5,0 s
        audio = np.concatenate([silence(1.0), tone(1.0, -14), noise(0.5, -36), silence(1.5), tone(1.0, -14, 330),
                                silence(1.0)])
        with wave.open(str(d / "ton.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25",
                        "-i", str(d / "ton.wav"), "-t", "6", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "film.mp4")], check=True)
        # Wortzeiten wie aus der Erkennung, mit zu frühem Ende des letzten Wortes (2,0 statt 2,5 s)
        words = [{"w": "Ist", "t0": 1.0, "t1": 1.3}, {"w": "das", "t0": 1.3, "t1": 1.6},
                 {"w": "pleite.", "t0": 1.6, "t1": 2.0}, {"w": "Weiter", "t0": 4.0, "t1": 4.5},
                 {"w": "geht's.", "t0": 4.5, "t1": 5.0}]
        (d / "words.json").write_text(json.dumps({"video": str(d / "film.mp4"), "source": "srt+whisper",
                                                  "words": words}), encoding="utf-8")
        cls.py(SCRIPTS / "windows.py", "--dir", d)
        cls.py(SCRIPTS / "analyze.py", d / "film.mp4", "--out", d)
        clips = {"video": "film.mp4", "channel": "TESTKANAL", "cta": "Ganzes Video: Link",
                 "clips": [{"id": "k01", "label": "TEST 01", "hook": "Ist das pleite? Äußerst €-teuer",
                            "parts": [["S000", "S000"]],
                            "rubric": {"hook": 8, "standalone": 8, "payoff": 7, "stakes": 6, "surprise": 7,
                                       "comment": 6}}]}
        (d / "clips.json").write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
        cls.py(SCRIPTS / "score.py", "--dir", d)
        cls.py(SCRIPTS / "render.py", "--dir", d, "--ids", "k01", "--layout", "full")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def py(*args):
        res = subprocess.run([sys.executable, "-I", *map(str, args)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if res.returncode:
            raise AssertionError(f"{args[0]} fehlgeschlagen:\n{res.stdout}\n{res.stderr}")
        return res.stdout

    def test_cut_keeps_whispered_tail(self):
        clip = json.loads((self.tmp / "clips.scored.json").read_text(encoding="utf-8"))["clips"][0]
        start, end = clip["cut"][0][0], clip["cut"][-1][1]
        self.assertLessEqual(start, 1.0)
        self.assertGreaterEqual(end, 2.6)   # 2,5 s Ausklang + Luft
        self.assertLess(end, 4.0)           # Satz 2 bleibt draußen

    def test_rendered_short(self):
        out = self.tmp / "out" / "k01.mp4"
        info = probe(out)
        self.assertEqual((info.width, info.height), (1080, 1920))
        self.assertTrue(info.has_audio)
        clip = json.loads((self.tmp / "clips.scored.json").read_text(encoding="utf-8"))["clips"][0]
        planned = sum(b - a for a, b in clip["cut"])
        self.assertAlmostEqual(info.duration, planned, delta=0.1)
        db = level_db(load_audio(out, SR_LEVEL), HOP_LEVEL)
        hop = HOP_LEVEL / SR_LEVEL
        offset = clip["cut"][0][0]
        whisper = db[int((2.1 - offset) / hop):int((2.4 - offset) / hop)]
        self.assertGreater(float(whisper.mean()), -50.0, "leiser Ausklang fehlt im fertigen Short")
        self.assertTrue((self.tmp / "out" / "k01_cover.jpg").is_file())

    def test_check_passes(self):
        stdout = self.py(SCRIPTS / "check.py", "--dir", self.tmp)
        self.assertIn("0 prüfen, 0 durchgefallen", stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg fehlt")
class SyncTest(unittest.TestCase):
    """Bild und Ton bleiben über viele Stücke synchron (Pausen gekürzt, Tempo 1,1, Quelle 25 fps).

    Im Quellvideo ist das Bild genau dann weiß, wenn ein Ton klingt. Im fertigen Short muss jeder Toneinsatz mit
    dem Aufhellen des Bildes zusammenfallen, höchstens ein Bild versetzt."""

    W, H, SRC_FPS, BURSTS = 320, 180, 25, 6

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shorts-sync-"))
        d = cls.tmp
        on = [(1.0 + k * 2.4, 1.0 + k * 2.4 + 0.8) for k in range(cls.BURSTS)]  # 0,8 s Ton, 1,6 s Pause
        total = on[-1][1] + 1.5
        audio = silence(total)
        for k, (a, b) in enumerate(on):
            i0, i1 = int(a * SR), int(b * SR)
            audio[i0:i1] = tone((i1 - i0 + 1) / SR, -14, 200 + 40 * k)[:i1 - i0]
        with wave.open(str(d / "ton.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        n = int(total * cls.SRC_FPS)
        frames = np.zeros((n, cls.H, cls.W, 3), np.uint8)
        for i in range(n):
            t = (i + 0.5) / cls.SRC_FPS
            if any(a <= t < b for a, b in on):
                frames[i] = 255
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cls.W}x{cls.H}",
                        "-r", str(cls.SRC_FPS), "-i", "-", "-i", str(d / "ton.wav"), "-c:v", "libx264", "-preset",
                        "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "film.mp4")],
                       input=frames.tobytes(), check=True)
        words = [{"w": f"Ton{k}.", "t0": a, "t1": b} for k, (a, b) in enumerate(on)]
        (d / "words.json").write_text(json.dumps({"video": "film.mp4", "source": "srt+whisper", "words": words}),
                                      encoding="utf-8")
        PipelineTest.py(SCRIPTS / "windows.py", "--dir", d)
        PipelineTest.py(SCRIPTS / "analyze.py", d / "film.mp4", "--out", d)
        clips = {"video": "film.mp4", "speed": 1.1,
                 "clips": [{"id": "s01", "hook": "Synchron", "parts": [["S000", f"S{cls.BURSTS - 1:03d}"]],
                            "rubric": {"hook": 8, "standalone": 8, "payoff": 7, "stakes": 6, "surprise": 7,
                                       "comment": 6}}]}
        (d / "clips.json").write_text(json.dumps(clips), encoding="utf-8")
        PipelineTest.py(SCRIPTS / "score.py", "--dir", d)
        PipelineTest.py(SCRIPTS / "render.py", "--dir", d, "--ids", "s01", "--layout", "full")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_many_pieces(self):
        clip = json.loads((self.tmp / "clips.scored.json").read_text(encoding="utf-8"))["clips"][0]
        self.assertEqual(len(clip["cut"]), self.BURSTS)

    def test_picture_and_sound_stay_in_sync(self):
        out = self.tmp / "out" / "s01.mp4"
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vf", "crop=200:200:440:860,scale=1:1",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        bright = np.frombuffer(raw, np.uint8).astype(np.float32)
        audio = load_audio(out, 30 * 100)  # 100 Abtastwerte je Bild bei 30 fps
        n = min(len(bright), len(audio) // 100)
        loud = np.sqrt((audio[:n * 100].reshape(n, 100) ** 2).mean(axis=1)) > 0.05
        lit = bright[:n] > 128
        v_on = [i for i in range(1, n) if lit[i] and not lit[i - 1]]
        a_on = [i for i in range(1, n) if loud[i] and not loud[i - 1]]
        self.assertEqual(len(v_on), self.BURSTS, f"Bild-Einsätze {v_on}")
        self.assertEqual(len(a_on), self.BURSTS, f"Ton-Einsätze {a_on}")
        for v, a in zip(v_on, a_on):
            self.assertLessEqual(abs(v - a), 1, f"Bild bei Bild {v}, Ton bei Bild {a}")

    def test_check_passes(self):
        stdout = PipelineTest.py(SCRIPTS / "check.py", "--dir", self.tmp)
        self.assertIn("0 prüfen, 0 durchgefallen", stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg fehlt")
class DelayedAudioTest(unittest.TestCase):
    """Codex R06 (Runde 3): Beginnt die Tonspur später als das Bild, muss Sekunde t überall dieselbe Stelle sein."""

    def test_audio_is_on_video_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with wave.open(str(d / "ton.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((tone(2.0, -14) * 32767).astype(np.int16).tobytes())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=5",
                            "-itsoffset", "2", "-i", str(d / "ton.wav"), "-map", "0:v", "-map", "1:a", "-c:v",
                            "libx264", "-c:a", "aac", str(d / "spaet.mp4")], check=True)
            audio = load_audio(d / "spaet.mp4", 8000)
            db = level_db(audio, 400)
            self.assertLess(float(db[5:35].max()), -50.0, "vor Sekunde 2 muss Stille sein")
            self.assertGreater(float(db[45:75].mean()), -25.0, "ab Sekunde 2 muss der Ton liegen")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg fehlt")
class DelayedAudioRenderTest(unittest.TestCase):
    """R04 (Runde 4): Auch beim Rendern liegt der Ton auf der Zeitachse des Videos, jedes Stück mit genauer Länge."""

    def test_render_audio_keeps_leading_silence(self):
        from render import build_audio
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with wave.open(str(d / "ton.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((tone(2.0, -14) * 32767).astype(np.int16).tobytes())
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=5",
                            "-itsoffset", "2", "-i", str(d / "ton.wav"), "-map", "0:v", "-map", "1:a", "-c:v",
                            "libx264", "-c:a", "aac", str(d / "spaet.mp4")], check=True)
            clip = {"cut": [[1.5, 3.0]], "air": [[0.4, 0.4]], "speed": 1.0}
            out, total = build_audio(clip, str(d / "spaet.mp4"), d)
            self.assertAlmostEqual(total, 1.5)
            db = level_db(load_audio(out, 8000), 400)
            self.assertLess(float(db[1:9].max()), -40.0, "0,5 s Stille vor dem Toneinsatz fehlt")
            self.assertGreater(float(db[12:28].mean()), -30.0)
            self.assertAlmostEqual(len(load_audio(out, 8000)) / 8000, 1.5, delta=0.05)

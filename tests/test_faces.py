"""Zuschnitt „face“: Modellausgabe dekodieren, Personen verfolgen, Ausschnitt setzen, Sprecher wechseln.

Die Logik wird mit erfundenen Zahlen geprüft (kein Video, kein Foto einer echten Person). Ein Rauchtest lädt das
mitgelieferte Modell, wenn onnxruntime installiert ist.

    python -m unittest discover -s tests -v
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "skills" / "shortomation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from faces import (MAX_ZOOM, MIN_HOLD, NET, SAMPLE_FPS, STRIDES, Face, FaceDetector, Track,  # noqa: E402
                   assign, decode, face_box, nms, relevant, speaker_turns)

ASPECT = 1080 / 880  # Bildfläche des Shorts (zwischen Hook und Untertiteln)


def blank_outputs():
    """Ausgaben wie vom Modell (cls, obj, bbox, kps je Schrittweite), alles null."""
    return [np.zeros((1, (NET // s) ** 2, depth), np.float32)
            for depth in (1, 1, 4, 10) for s in STRIDES]


def face(x, y, w=100.0, h=130.0):
    return Face(x, y, w, h, 0.9)


class DecodeTest(unittest.TestCase):
    def test_one_cell_becomes_one_face(self):
        outs = blank_outputs()
        i, stride, row, col = 1, 16, 10, 20          # Schrittweite 16, Zelle (Zeile 10, Spalte 20)
        idx = row * (NET // stride) + col
        outs[i][0, idx, 0] = 0.81                     # cls
        outs[i + 3][0, idx, 0] = 1.0                  # obj -> Sicherheit sqrt(0.81) = 0.9
        outs[i + 6][0, idx] = [0.5, 0.25, math.log(4), math.log(5)]
        outs[i + 9][0, idx] = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0]
        (f,) = decode(outs)
        self.assertAlmostEqual(f.score, 0.9, places=5)
        self.assertAlmostEqual(f.x, (col + 0.5) * stride - 32, places=4)   # Mitte 328, Breite 64
        self.assertAlmostEqual(f.y, (row + 0.25) * stride - 40, places=4)  # Mitte 164, Höhe 80
        self.assertAlmostEqual(f.w, 64, places=4)
        self.assertAlmostEqual(f.h, 80, places=4)
        self.assertEqual(f.marks[0], (col * stride, row * stride))
        self.assertEqual(f.marks[2], ((col + 0.5) * stride, (row + 0.5) * stride))

    def test_unsure_cells_are_ignored(self):
        outs = blank_outputs()
        outs[0][0, 5, 0] = 0.3
        outs[3][0, 5, 0] = 0.3
        self.assertEqual(decode(outs), [])

    def test_wrong_model_is_reported(self):
        with self.assertRaises(SystemExit):
            decode(blank_outputs()[:6])

    def test_nms_keeps_the_surest_of_overlapping_boxes(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [300, 0, 100, 100]], float)
        self.assertEqual(nms(boxes, np.array([0.8, 0.95, 0.7])), [1, 2])


class TrackTest(unittest.TestCase):
    def follow(self, frames):
        tracks = []
        for k, faces in enumerate(frames):
            for track, f in assign(tracks, faces, k):
                if not any(track is t for t in tracks):
                    tracks.append(track)
                track.add(k, f, None)
        return tracks

    def test_same_person_stays_one_track(self):
        tracks = self.follow([[face(100 + 4 * k, 100), face(900, 120)] for k in range(20)])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(sorted(len(t.faces) for t in tracks), [20, 20])

    def test_short_gap_keeps_the_track(self):
        frames = [[face(100, 100)] for _ in range(10)] + [[] for _ in range(5)] + [[face(102, 100)] for _ in range(5)]
        self.assertEqual(len(self.follow(frames)), 1)

    def test_motion_only_between_neighbouring_frames(self):
        t = Track()
        patch = (np.zeros((12, 16), np.float32), np.zeros((12, 16), np.float32))
        moved = (np.ones((12, 16), np.float32), np.zeros((12, 16), np.float32))
        t.add(0, face(0, 0), patch)
        t.add(1, face(0, 0), moved)
        t.add(3, face(0, 0), patch)   # Lücke: keine Bewegung messbar
        self.assertEqual(t.motion, {1: 1.0})

    def test_poster_and_passers_by_do_not_count(self):
        host = Track(faces={k: face(500, 300, 200, 260) for k in range(50)})
        poster = Track(faces={k: face(1500, 100, 60, 80) for k in range(50)})       # klein: Bild an der Wand
        passer = Track(faces={k: face(900, 300, 200, 260) for k in range(10)})      # nur kurz im Bild
        self.assertEqual(relevant([host, poster, passer], 50, 1080), [host])


class BoxTest(unittest.TestCase):
    def test_close_up_keeps_full_height_and_centres_the_face(self):
        x, y, w, h = face_box([face(860, 300, 200, 360)], 1920, 1080, ASPECT)
        self.assertAlmostEqual(w / h, ASPECT, places=6)
        self.assertAlmostEqual(h, 1080)
        self.assertAlmostEqual(x + w / 2, 960, delta=1)

    def test_small_face_is_zoomed_at_most_max_zoom(self):
        x, y, w, h = face_box([face(900, 500, 40, 50)], 1920, 1080, ASPECT)
        self.assertAlmostEqual(h, 1080 / MAX_ZOOM)
        self.assertTrue(x <= 900 and x + w >= 940 and y <= 500 and y + h >= 550)

    def test_box_stays_inside_the_frame(self):
        x, y, w, h = face_box([face(1850, 10, 60, 80)], 1920, 1080, ASPECT)
        self.assertTrue(0 <= x and x + w <= 1920 + 1e-6 and 0 <= y and y + h <= 1080 + 1e-6)

    def test_other_person_is_left_out(self):
        speaker, other = face(450, 350, 220, 280), face(1250, 350, 220, 280)
        x, y, w, h = face_box([speaker], 1920, 1080, ASPECT, [other])
        self.assertLessEqual(x + w, other.x - 0.3 * other.w)       # kein Stück vom Nachbarn samt Haaren
        self.assertTrue(x <= speaker.x and x + w >= speaker.x + speaker.w)


class SpeakerTest(unittest.TestCase):
    N = 10 * SAMPLE_FPS  # 10 s

    def times(self):
        return [k / SAMPLE_FPS for k in range(self.N)]

    def test_switch_lands_on_the_sentence_start(self):
        a = np.where(np.arange(self.N) < 50, 0.06, 0.004)  # links spricht bis 5 s
        b = np.where(np.arange(self.N) < 50, 0.004, 0.05)  # rechts ab 5 s
        turns = speaker_turns(self.times(), np.stack([a, b]), [True] * self.N, [2.0, 5.2, 8.0], first=1)
        self.assertEqual([who for _, who in turns], [0, 1])
        self.assertAlmostEqual(turns[1][0], 5.2)

    def test_short_blip_is_no_switch(self):
        a = np.full(self.N, 0.05)
        b = np.full(self.N, 0.004)
        b[40:43] = 0.3          # 0,3 s Lachen oder Nicken
        a[40:43] = 0.004
        turns = speaker_turns(self.times(), np.stack([a, b]), [True] * self.N, [], first=0)
        self.assertEqual(turns, [(0.0, 0)])

    def test_without_speech_the_picture_stays(self):
        a = np.where(np.arange(self.N) < 50, 0.06, 0.004)
        b = np.where(np.arange(self.N) < 50, 0.004, 0.06)  # Kauen, Nicken: Bewegung ohne Sprache
        turns = speaker_turns(self.times(), np.stack([a, b]), [k < 50 for k in range(self.N)], [], first=0)
        self.assertEqual([who for _, who in turns], [0])

    def test_turns_keep_the_minimum_hold(self):
        flip = (np.arange(self.N) // 8) % 2 == 0   # Wechsel alle 0,8 s
        a, b = np.where(flip, 0.06, 0.004), np.where(flip, 0.004, 0.06)
        turns = speaker_turns(self.times(), np.stack([a, b]), [True] * self.N, [], first=0)
        gaps = np.diff([t for t, _ in turns])
        self.assertTrue(all(g >= MIN_HOLD - 1e-9 for g in gaps), f"Wechsel zu dicht: {turns}")

    def test_hidden_person_cannot_win(self):
        a = np.full(self.N, 0.05)
        b = np.full(self.N, np.nan)   # nicht zu sehen
        turns = speaker_turns(self.times(), np.stack([a, b]), [True] * self.N, [], first=1)
        self.assertEqual(turns[-1][1], 0)


@unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime fehlt")
class ModelSmokeTest(unittest.TestCase):
    def test_model_loads_and_finds_nobody_in_an_empty_picture(self):
        det = FaceDetector()
        self.assertEqual(det.detect(np.zeros((360, 640, 3), np.uint8)), [])
        self.assertEqual(det.detect(np.full((1080, 608, 3), 200, np.uint8)), [])


if __name__ == "__main__":
    unittest.main()

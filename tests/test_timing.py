"""Unit-Tests für Satzbildung, SRT-Abgleich, Schnittgrenzen und Prüfung (ohne ffmpeg, ohne Spracherkennung).

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "shortomation" / "scripts"))

from types import SimpleNamespace  # noqa: E402

from check import edge_report, edge_word, heard_near, require_metadata, unsure_flags, word_report  # noqa: E402
from common import StepError, level_db, norm_word  # noqa: E402
from render import preflight_fonts, validate_style  # noqa: E402
from score import KEEP_GAP, LEAD_MIN, TAIL_MIN, Level, cut_part, score_clips, validate  # noqa: E402
from transcribe import align_srt, match_times, release_weak_anchors  # noqa: E402
from windows import build_sentences, build_windows, ends_sentence, merge_punctuation  # noqa: E402

HOP = 0.05
RUBRIC = {"hook": 8, "standalone": 7, "payoff": 6, "stakes": 5, "surprise": 7, "comment": 6}


def words(spec):
    """[(wort, t0, t1)] -> words.json-Form."""
    return [{"w": w, "t0": a, "t1": b} for w, a, b in spec]


def level_curve(segments, total, floor=-70.0):
    """[(t0, t1, dB)] -> Pegelkurve in 50-ms-Schritten, sonst Grundrauschen floor."""
    db = np.full(int(round(total / HOP)), floor)
    for a, b, v in segments:
        db[int(round(a / HOP)):int(round(b / HOP))] = v
    return list(db)


def sentences(spec):
    s = build_sentences(words(spec), split_on_pause=False)
    return s, {x["id"]: i for i, x in enumerate(s)}


class SentenceTests(unittest.TestCase):
    def test_sentence_end_rules(self):
        self.assertTrue(ends_sentence("Euro."))
        self.assertTrue(ends_sentence("gemerkt?"))
        self.assertTrue(ends_sentence("„Gefälscht.“"))
        self.assertFalse(ends_sentence("18."))  # Ordinalzahl „18. Juni“
        self.assertFalse(ends_sentence("z.B."))
        self.assertFalse(ends_sentence("Konzern"))

    def test_times_come_from_first_and_last_word(self):
        s = build_sentences(words([("Es", 1.0, 1.2), ("geht", 1.2, 1.5), ("um", 1.5, 1.6), ("Geld.", 1.6, 2.4),
                                   ("Viel", 3.0, 3.3), ("Geld.", 3.3, 3.9)]))
        self.assertEqual([x["text"] for x in s], ["Es geht um Geld.", "Viel Geld."])
        self.assertEqual((s[0]["t0"], s[0]["t1"]), (1.0, 2.4))
        self.assertEqual(s[1]["tokens"][-1], ["Geld.", 3.3, 3.9])
        self.assertEqual(s[0]["n_words"], 4)

    def test_long_pause_splits_only_without_srt(self):
        spec = words([("Da", 0.0, 0.2), ("ist", 0.2, 0.4), ("nichts", 2.0, 2.5)])
        self.assertEqual(len(build_sentences(spec)), 2)
        self.assertEqual(len(build_sentences(spec, split_on_pause=False)), 1)

    def test_punctuation_tokens_join_their_neighbour(self):
        merged = merge_punctuation(words([("–", 0.0, 0.1), ("Das", 0.1, 0.3), ("war", 0.3, 0.5), ("…", 0.5, 0.6)]))
        self.assertEqual([w["w"] for w in merged], ["– Das", "war …"])
        self.assertEqual(len(build_sentences(words([("Gut.", 0, 1), ("…", 1, 1.1), ("Weiter.", 2, 3)]))), 2)

    def test_estimated_words_are_counted(self):
        s = build_sentences([{"w": "Eine", "t0": 1, "t1": 2, "est": True}, {"w": "Milliarde.", "t0": 2, "t1": 3}])
        self.assertEqual(s[0]["est_words"], 1)

    def test_windows_cover_all_sentences(self):
        sents = build_sentences(words([(f"W{i}.", i * 10.0, i * 10.0 + 2) for i in range(20)]))
        wins = build_windows(sents, win=75, step=35)
        self.assertEqual(wins[0]["s0"], "S000")
        self.assertEqual(wins[-1]["s1"], sents[-1]["id"])

    def test_windows_never_skip_a_long_sentence(self):
        """Satz 10–90 s darf nicht zwischen zwei Fenstern verloren gehen."""
        sents = build_sentences(words([("Kurz.", 0, 1), ("Sehr", 10, 50), ("lang.", 50, 90), ("Ende.", 95, 100)]))
        wins = build_windows(sents, win=75, step=35)
        covered = set()
        idx = {s["id"]: i for i, s in enumerate(sents)}
        for w in wins:
            covered.update(range(idx[w["s0"]], idx[w["s1"]] + 1))
        self.assertEqual(covered, set(range(len(sents))))


class SrtAlignTests(unittest.TestCase):
    def test_srt_spelling_with_recognized_times(self):
        cues = [{"t0": 10.0, "t1": 14.0, "text": "Es geht um 1,9 Milliarden Euro."}]
        heard = words([("Es", 10.1, 10.3), ("geht", 10.3, 10.6), ("um", 10.6, 10.8),
                       ("eins", 10.9, 11.2), ("Komma", 11.2, 11.5), ("neun", 11.5, 11.8),
                       ("Milliarden", 11.9, 12.6), ("Euro.", 12.6, 13.4)])
        out = align_srt(cues, heard)
        self.assertEqual([w["w"] for w in out], ["Es", "geht", "um", "1,9", "Milliarden", "Euro."])
        self.assertAlmostEqual(out[-1]["t1"], 13.4)          # gemessenes Ende, nicht das SRT-Ende 14.0
        self.assertTrue(10.8 <= out[3]["t0"] < out[3]["t1"] <= 11.9)
        self.assertFalse(any(w.get("est") for w in out))

    def test_unheard_words_are_marked_and_monotone(self):
        """Überhörte Wörter bekommen keine erfundene „gemessene“ Zeit, sondern est=True."""
        cues = [{"t0": 0.0, "t1": 9.0, "text": "Bank. Eine Milliarde. Ende."}]
        heard = words([("Bank.", 1.0, 1.5), ("Ende.", 8.0, 8.5)])
        out = align_srt(cues, heard)
        self.assertEqual([w.get("est", False) for w in out], [False, True, True, False])
        times = [t for w in out for t in (w["t0"], w["t1"])]
        self.assertEqual(times, sorted(times))

    def test_different_words_are_not_treated_as_spelling(self):
        """„Eine Milliarde“ gegen „Unsinn“ ist keine andere Schreibweise."""
        times = match_times(["Bank.", "Eine", "Milliarde.", "Ende."],
                            words([("Bank.", 1.0, 1.5), ("Unsinn.", 3.0, 3.5), ("Ende.", 8.0, 8.5)]))
        self.assertIsNone(times[1])
        self.assertIsNone(times[2])
        cues = [{"t0": 0, "t1": 9, "text": "Bank. Eine Milliarde. Ende."}]
        out = align_srt(cues, words([("Bank.", 1.0, 1.5), ("Unsinn.", 3.0, 3.5), ("Ende.", 8.0, 8.5)]))
        self.assertEqual([w.get("est", False) for w in out], [False, True, True, False])

    def test_false_anchor_is_released(self):
        """Echter Fall: „Ende“ wurde als „Eine“ verhört und hat den Satz davor auf 0 s gequetscht."""
        cues = [{"t0": 580, "t1": 593, "text": "Bank. Eine Milliarde. Belegt durch Briefe. Ende 2019 wechselt."}]
        heard = words([("Bank.", 584.1, 584.46), ("Eine", 589.95, 590.65), ("2019", 590.65, 591.47),
                       ("wechselt.", 591.5, 592.0)])
        out = {w["w"]: w for w in align_srt(cues, heard)}
        self.assertTrue(out["Eine"]["est"])
        self.assertGreater(out["Milliarde."]["t1"] - out["Milliarde."]["t0"], 0.12)
        self.assertLess(out["Eine"]["t0"], 586.0)

    def test_releasing_anchor_between_gaps_does_not_crash(self):
        """Lösen eines Ankers verbindet zwei Lücken."""
        times = [None, (0.05, 0.10), None, (1.0, 1.2)]
        release_weak_anchors(times, 0.0, 2.0)
        self.assertEqual(times[:3], [None, None, None])

    def test_number_against_unrelated_word_is_unheard(self):
        """„1 Milliarde“ gegen „Unsinn“ ist keine andere Schreibweise."""
        times = match_times(["Bank.", "1", "Milliarde.", "Ende."],
                            words([("Bank.", 1.0, 1.5), ("Unsinn.", 3.0, 3.5), ("Ende.", 8.0, 8.5)]))
        self.assertIsNone(times[1])
        ok = match_times(["um", "1,9", "Milliarden."], words([("um", 1.0, 1.2), ("eins", 1.3, 1.5),
                                                              ("Komma", 1.5, 1.8), ("neun", 1.8, 2.0),
                                                              ("Milliarden.", 2.0, 2.6)]))
        self.assertIsNotNone(ok[1])

    def test_unheard_ending_spans_to_end_of_recording(self):
        """Ein überhörter letzter Satz liegt zwischen letztem Wort und Aufnahmeende."""
        cues = [{"t0": 0, "t1": 6, "text": "Anfang. Schluss hier."}]
        out = align_srt(cues, words([("Anfang.", 1.0, 1.5)]), audio=np.zeros(6 * 16000, np.float32))
        self.assertTrue(out[1]["est"] and out[2]["est"])
        self.assertGreater(out[2]["t1"], 5.0)
        self.assertGreater(out[1]["t1"] - out[1]["t0"], 0.5)

    def test_norm_word(self):
        self.assertEqual(norm_word("„Euro.“"), "euro")
        self.assertEqual(norm_word("1,9"), "19")
        self.assertEqual(norm_word("Größe!"), "größe")
        self.assertEqual(norm_word("…"), "")


class CutTests(unittest.TestCase):
    def test_whispered_last_word_is_kept(self):
        """Regression „pleite“: Erkennung meldet das Wortende zu früh, das Flüstern klingt noch 0,5 s nach."""
        sents, idx = sentences([("Da", 1.0, 1.2), ("ist", 1.2, 1.4), ("es", 1.4, 1.6), ("pleite.", 1.6, 2.0),
                                ("Weiter.", 4.0, 4.6)])
        level = Level(level_curve([(1.0, 2.0, -16), (2.0, 2.5, -38), (4.0, 4.6, -16)], 6.0), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertGreaterEqual(cut.pieces[-1][1], 2.5 + TAIL_MIN - 1e-6)
        self.assertLess(cut.pieces[-1][1], 4.0)
        self.assertGreaterEqual(cut.air[-1][1], TAIL_MIN - 1e-6)

    def test_whisper_over_realistic_background(self):
        """Raumton -45 dB, Sprache -16 dB, Flüstern -38 dB muss als Laut zählen."""
        sents, idx = sentences([("Es", 1.0, 1.5), ("pleite.", 1.5, 2.0), ("Weiter.", 4.5, 5.0)])
        level = Level(level_curve([(1.0, 2.0, -16), (2.0, 2.5, -38), (4.5, 5.0, -16)], 8.0, floor=-45), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertGreaterEqual(cut.pieces[-1][1], 2.5 + TAIL_MIN - 1e-6)

    def test_music_bed_is_not_speech_with_vad(self):
        """Musikbett −22 dB direkt nach dem Satz: mit Sprachwahrscheinlichkeit endet der Satz am Wortende."""
        sents, idx = sentences([("Satz.", 1.0, 2.0), ("Weiter.", 4.0, 4.6)])
        db = level_curve([(0.0, 6.0, -22), (1.0, 2.0, -12), (4.0, 4.6, -12)], 6.0, floor=-40)
        prob = [0.95 if 1.0 <= i * HOP < 2.0 or 4.0 <= i * HOP < 4.6 else 0.02 for i in range(len(db))]
        with_vad = cut_part("S000", "S000", sents, idx, Level(db, HOP, prob=prob))
        self.assertTrue(2.1 <= with_vad.pieces[-1][1] <= 2.5)
        self.assertFalse(any("nicht sicher" in n for n in with_vad.notes))

    def test_whisper_with_speech_probability_is_kept(self):
        sents, idx = sentences([("Es", 1.0, 1.5), ("pleite.", 1.5, 2.0), ("Weiter.", 4.5, 5.0)])
        db = level_curve([(1.0, 2.0, -16), (2.0, 2.5, -38), (4.5, 5.0, -16)], 8.0, floor=-45)
        prob = [0.9 if 1.0 <= i * HOP < 2.5 or 4.5 <= i * HOP < 5.0 else 0.0 for i in range(len(db))]
        cut = cut_part("S000", "S000", sents, idx, Level(db, HOP, prob=prob))
        self.assertGreaterEqual(cut.pieces[-1][1], 2.5 + TAIL_MIN - 1e-6)

    def test_start_includes_quiet_onset(self):
        sents, idx = sentences([("Hallo.", 1.0, 1.5)])
        level = Level(level_curve([(0.7, 1.0, -36), (1.0, 1.5, -15)], 3.0), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertLessEqual(cut.pieces[0][0], 0.7)

    def test_long_pause_is_shortened_without_cutting_words(self):
        sents, idx = sentences([("Eins.", 1.0, 1.6), ("Zwei.", 4.0, 4.6)])
        level = Level(level_curve([(1.0, 1.6, -15), (4.0, 4.6, -15)], 6.0), HOP)
        cut = cut_part("S000", "S001", sents, idx, level)
        self.assertEqual(len(cut.pieces), 2)
        self.assertEqual(cut.gaps, [KEEP_GAP])
        self.assertEqual(cut.piece_sents, [["S000"], ["S001"]])
        self.assertGreaterEqual(cut.pieces[0][1], 1.6 + TAIL_MIN - 1e-6)
        self.assertLessEqual(cut.pieces[1][0], 4.0)

    def test_next_sentence_onset_is_not_cut(self):
        """Prüfer: Der nächste Satz beginnt hörbar früher, als die Erkennung meldet."""
        sents, idx = sentences([("Eins.", 1.0, 1.6), ("Zwei.", 2.3, 2.9)])
        level = Level(level_curve([(1.0, 1.6, -15), (2.0, 2.9, -15)], 4.0), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertLessEqual(cut.pieces[-1][1], 2.0)

    def test_close_next_sentence_is_reported(self):
        sents, idx = sentences([("Eins.", 1.0, 1.6), ("Zwei.", 1.65, 2.3)])
        level = Level(level_curve([(1.0, 1.6, -15), (1.65, 2.3, -15)], 4.0), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertTrue(1.6 <= cut.pieces[-1][1] <= 1.65)
        self.assertTrue(any("enger Schnitt" in n for n in cut.notes))

    def test_onset_without_pause_is_kept_and_reported(self):
        """Satz davor endet 1,60, gewählter Satz klingt ab 1,65, Erkennung sagt 1,90."""
        sents, idx = sentences([("Davor.", 1.0, 1.6), ("Jetzt", 1.9, 2.3), ("los.", 2.3, 2.8), ("Weiter.", 5.0, 5.5)])
        level = Level(level_curve([(1.0, 1.6, -15), (1.65, 2.8, -15), (5.0, 5.5, -15)], 7.0), HOP)
        cut = cut_part("S001", "S001", sents, idx, level)
        self.assertLessEqual(cut.pieces[0][0], 1.65)
        self.assertTrue(any("enger Schnitt vor S001" in n for n in cut.notes))

    def test_lead_leaves_room_for_check(self):
        sents, idx = sentences([("Hallo.", 2.0, 2.5)])
        level = Level(level_curve([(2.0, 2.5, -15)], 4.0), HOP)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertGreaterEqual(cut.air[0][0], LEAD_MIN - 1e-6)

    def test_end_never_beyond_media(self):
        """Ton läuft bis zum Dateiende; nichts davon geht verloren."""
        sents, idx = sentences([("Schluss.", 1.0, 1.9)])
        level = Level(level_curve([(1.0, 2.05, -15)], 2.05), HOP, duration=2.049)
        cut = cut_part("S000", "S000", sents, idx, level)
        self.assertLessEqual(cut.pieces[-1][1], 2.049)
        self.assertAlmostEqual(cut.pieces[-1][1], 2.049, places=3)

    def test_unknown_or_reversed_range(self):
        sents, idx = sentences([("Eins.", 1.0, 1.6), ("Zwei.", 3.0, 3.6)])
        level = Level(level_curve([], 5.0), HOP)
        with self.assertRaises(StepError):
            cut_part("S000", "S009", sents, idx, level)
        with self.assertRaises(StepError):
            cut_part("S001", "S000", sents, idx, level)


class ValidateTests(unittest.TestCase):
    good = {"id": "k01", "hook": "Test", "parts": [["S000", "S001"]], "rubric": RUBRIC}

    def cj(self, *clips, **extra):
        return {"video": "film.mp4", "clips": list(clips), **extra}

    def test_accepts_valid(self):
        validate(self.cj(dict(self.good)))

    def test_rejects_bad_input(self):
        bad = [[], {"clips": [self.good]}, self.cj(), self.cj({**self.good, "rubric": None}),
               self.cj({**self.good, "rubric": {**RUBRIC, "hook": 12}}), self.cj({**self.good, "parts": [["S000"]]}),
               self.cj(self.good, dict(self.good)), self.cj(self.good, {**self.good, "id": "K01"}),
               self.cj({**self.good, "id": "../x"}), self.cj({**self.good, "id": "con"}),
               self.cj({**self.good, "speed": "fast"}), self.cj({**self.good, "speed": 2.0}),
               self.cj(self.good, claimed=[[10, 5]]), self.cj(self.good, attribution=[[1, 2]]), self.cj("k01")]
        for cj in bad:
            with self.subTest(cj=cj), self.assertRaises(StepError):
                validate(cj)


class OverlapTests(unittest.TestCase):
    def test_overlap_only_against_renderable_clips(self):
        """Ein Clip mit Content-ID darf einen sauberen Nachbarn nicht ausschließen."""
        sents, _ = sentences([(f"Satz{i}.", i * 2.0 + 1, i * 2.0 + 2) for i in range(6)])
        an = {"rms_db": level_curve([(i * 2.0 + 1, i * 2.0 + 2, -15) for i in range(6)], 14.0), "hop_s": HOP,
              "motion": [1.0] * 140, "motion_fps": 10, "cuts": []}
        cj = {"video": "x.mp4", "claimed": [[0.5, 1.5]],
              "clips": [{"id": "a", "hook": "A", "parts": [["S000", "S005"]], "rubric": {**RUBRIC, "hook": 10}},
                        {"id": "b", "hook": "B", "parts": [["S001", "S005"]], "rubric": RUBRIC}]}
        out = {c["id"]: c for c in score_clips(cj, sents, an)}
        self.assertIn("Content-ID-Musik", out["a"]["flags"])
        self.assertFalse(any(f.startswith("überlappt") for f in out["b"]["flags"]))


class LevelTests(unittest.TestCase):
    def test_partial_last_window_counts(self):
        """Die letzten 49 ms dürfen nicht verschwinden."""
        audio = np.full(16000 + 392, 0.1, np.float32)  # 2,049 s bei 8 kHz mit Fenster 400
        self.assertEqual(len(level_db(audio, 400)), 41)
        level = Level([-20.0] * 41, HOP, duration=2.049)
        self.assertAlmostEqual(level.duration, 2.049)

    def test_empty_audio_is_rejected(self):
        with self.assertRaises(StepError):
            Level([], HOP)


class CheckEdgeTests(unittest.TestCase):
    clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.0], [5.0, 7.0]], "air": [[0.0, 0.0], [0.2, 0.3]],
            "piece_sents": [["S000"], ["S001"]]}

    def test_unmeasurable_boundary_is_reported(self):
        """Ohne messbare Pause darf eine Grenze nicht still als sauber gelten."""
        db = np.full(400, -60.0)
        errors, unchecked = edge_report(db, 0.01, self.clip)
        self.assertEqual(errors, [])
        self.assertEqual(len(unchecked), 2)

    def test_loud_vs_quiet_sound_at_boundary(self):
        """Sprache am Schnitt -> Fehler; leiser Laut (Ausklang, Musik) -> anhören."""
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.0]], "air": [[0.2, 0.4]], "piece_sents": [["S000"]]}
        db = np.full(200, -60.0)
        db[30:150] = -14.0                      # Sprache
        loud, quiet = db.copy(), db.copy()
        loud[170:195] = -16.0                    # Sprache im Ruhefenster vor dem Schnitt
        quiet[170:195] = -45.0                   # leiser Laut im Ruhefenster
        self.assertEqual(len(edge_report(loud, 0.01, clip)[0]), 1)
        errors, review = edge_report(quiet, 0.01, clip)
        self.assertEqual(errors, [])
        self.assertEqual(len(review), 1)

    def test_music_at_boundary_is_not_speech(self):
        """Musikbett am Schnitt (laut, aber Silero sagt keine Sprache) ist kein Fehler; Sprache schon."""
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.0]], "air": [[0.2, 0.4]], "piece_sents": [["S000"]]}
        db = np.full(200, -60.0)
        db[30:150] = -14.0
        db[150:200] = -18.0                       # Musik nach dem letzten Wort, fast so laut wie Sprache
        vad = np.zeros(63, np.float32)
        vad[10:46] = 0.95                         # Sprache 0,3–1,5 s
        self.assertEqual(edge_report(db, 0.01, clip, vad), ([], []))
        vad[55:63] = 0.9                          # Sprache bis an den Schnitt
        self.assertEqual(len(edge_report(db, 0.01, clip, vad)[0]), 1)

    def test_all_uncertainty_notes_are_recognized(self):
        """jede Unsicherheit aus score.py führt zu „prüfen“."""
        flags = ["Wortzeiten in S001 geschätzt", "Anfang von S002 nicht sicher gemessen",
                 "Ende von S003 nicht sicher gemessen", "enger Schnitt nach S004 (0.00 s Pause)", "⚖ prüfen"]
        self.assertEqual(len(unsure_flags({"flags": flags})), 4)

    def test_old_metadata_is_rejected(self):
        """Metadaten im alten Format werden mit klarer Meldung abgelehnt."""
        with self.assertRaises(StepError):
            require_metadata({"id": "k01", "cut": [[0, 1], [2, 3]], "air": [[0.2, 0.2]] * 2, "piece_sents": [["S0"]]})
        require_metadata(self.clip)


class FakeModel:
    """Spielt die Spracherkennung nach: liefert feste Wörter mit Zeiten."""

    def __init__(self, spec):
        self.words = [SimpleNamespace(word=w, start=a, end=b) for w, a, b in spec]

    def transcribe(self, *args, **kwargs):
        return [SimpleNamespace(words=self.words)], None


class CheckWordTests(unittest.TestCase):
    def test_missing_one_word_sentence_is_detected(self):
        """Ein fehlender Ein-Wort-Satz („Nein.“) muss auffallen."""
        by_id = {s["id"]: s for s in build_sentences(words([("Einleitung.", 0.2, 0.8), ("Nein.", 1.0, 1.3),
                                                             ("Schluss.", 1.5, 2.0)]))}
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.3]], "air": [[0.2, 0.3]],
                "piece_sents": [["S000", "S001", "S002"]]}
        import check
        original = check.load_audio
        check.load_audio = lambda *a, **k: np.zeros(16000, np.float32)
        try:
            missing, _ = word_report(Path("x.mp4"), clip, by_id, FakeModel([(" Einleitung.", 0.2, 0.8),
                                                                         (" Schluss.", 1.5, 2.0)]))
            complete, _ = word_report(Path("x.mp4"), clip, by_id, FakeModel([(" Einleitung.", 0.2, 0.8),
                                                                          (" Nein.", 1.0, 1.3),
                                                                          (" Schluss.", 1.5, 2.0)]))
        finally:
            check.load_audio = original
        self.assertTrue(any("S001" in p for p in missing))
        self.assertEqual(complete, [])

    def test_missing_sentence_not_covered_by_neighbour(self):
        """„nein“ aus dem Satz davor darf den fehlenden Satz „Nein.“ nicht abdecken."""
        by_id = {s["id"]: s for s in build_sentences(words([("Er", 0.1, 0.3), ("sagt", 0.3, 0.5), ("nein.", 0.5, 0.8),
                                                             ("Nein.", 1.0, 1.3), ("Schluss.", 1.5, 2.0)]))}
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.3]], "air": [[0.1, 0.3]],
                "piece_sents": [["S000", "S001", "S002"]]}
        import check
        original = check.load_audio
        check.load_audio = lambda *a, **k: np.zeros(16000, np.float32)
        try:
            found, _ = word_report(Path("x.mp4"), clip, by_id, FakeModel([(" Er", 0.1, 0.3), (" sagt", 0.3, 0.5),
                                                                       (" nein.", 0.5, 0.8),
                                                                       (" Schluss.", 1.5, 2.0)]))
        finally:
            check.load_audio = original
        self.assertTrue(any("S001" in p for p in found))

    def test_acronyms_are_not_edge_words(self):
        self.assertEqual(edge_word(["BDO", "und", "BPI."], last=True), ("bpi", True))
        self.assertEqual(edge_word(["Er", "sagt", "NEIN."], last=True), ("nein", False))

    def test_cut_last_word_is_detected(self):
        self.assertFalse(heard_near("pleite", ["ist", "firma", "schon", "pleit"], last=True))
        self.assertTrue(heard_near("pleite", ["ist", "firma", "schon", "pleite"], last=True))

    def test_compound_and_numbers(self):
        self.assertTrue(heard_near("firmenaktie", ["der", "firmen", "aktie"], last=True))
        self.assertEqual(edge_word(["Am", "3.", "September", "2020."], last=True), ("2020", True))
        self.assertEqual(edge_word(["…"], last=True), ("", False))


class StyleTests(unittest.TestCase):
    def test_valid_and_invalid_styles(self):
        """kaputte Stilangaben -> klare Meldung statt Traceback."""
        self.assertEqual(validate_style({"accent": "#FFC83D"})["accent"], "#FFC83D")
        for bad in ({"fonts": {"hook": None}}, {"panel_top": "520"}, {"accent": "gelb"}, {"fonts": "x.ttf"},
                    {"unbekannt": 1}, {"ignore_edges": [0.1]}, {"accent_pattern": "("}, [], {"cap_size": True}):
            with self.subTest(style=bad), self.assertRaises(StepError):
                validate_style(bad)

    def test_broken_font_file_is_reported(self):
        """Eine kaputte Schriftdatei wird vor dem Rendern gemeldet."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "kaputt.ttf").write_bytes(b"keine Schrift")
            st = validate_style({"fonts": {"hook": "kaputt.ttf"}})
            with self.assertRaises(StepError):
                preflight_fonts(st, Path(tmp))
        preflight_fonts(validate_style({}), None)


class ClipTextTests(unittest.TestCase):
    def test_label_must_be_text(self):
        """Das Feld label muss Text sein."""
        with self.assertRaises(StepError):
            validate({"video": "x.mp4", "clips": [{"id": "k01", "hook": "H", "label": None, "parts": [["S000", "S000"]],
                                                   "rubric": RUBRIC}]})


if __name__ == "__main__":
    unittest.main()


class EdgeCaseTests(unittest.TestCase):
    """Flüstern, Musikbett, Zahlwörter und Abkürzungen an den Schnitten."""

    def test_whisper_with_low_vad_confidence_is_kept(self):
        """Raumton −45, Flüstern −38 bis 2,5 s, Silero nur 0,1: Energie zählt, das Ende bleibt drin."""
        sents, idx = sentences([("Es", 1.0, 1.5), ("pleite.", 1.5, 2.0), ("Weiter.", 4.5, 5.0)])
        db = level_curve([(1.0, 2.0, -16), (2.0, 2.5, -38), (4.5, 5.0, -16)], 8.0, floor=-45)
        prob = [0.9 if 1.0 <= i * HOP < 2.0 or 4.5 <= i * HOP < 5.0 else 0.1 for i in range(len(db))]
        cut = cut_part("S000", "S000", sents, idx, Level(db, HOP, prob=prob))
        self.assertGreaterEqual(cut.pieces[-1][1], 2.5 + TAIL_MIN - 1e-6)

    def test_speech_at_cut_fails_even_under_loud_music(self):
        """Musikbett −22 dB, Sprache −18 dB mit Silero 0,95 direkt am Schnitt = Fehler."""
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.0]], "air": [[0.2, 0.4]], "piece_sents": [["S000"]]}
        db = np.full(200, -22.0)
        db[30:150] = -12.0
        db[180:200] = -18.0
        vad = np.zeros(63, np.float32)
        vad[10:46] = 0.95
        vad[56:63] = 0.95
        self.assertEqual(len(edge_report(db, 0.01, clip, vad)[0]), 1)

    def test_unrelated_words_are_not_numbers(self):
        """„keine“ ist kein Zahlwort."""
        times = match_times(["Bank.", "1", "Milliarde.", "Ende."],
                            words([("Bank.", 1.0, 1.5), ("Keine.", 3.0, 3.5), ("Ende.", 8.0, 8.5)]))
        self.assertIsNone(times[1])

    def test_unrelated_replacement_is_a_missing_sentence(self):
        """„Eine Milliarde“ gegen gehörtes „Unsinn“ = Satz fehlt."""
        by_id = {s["id"]: s for s in build_sentences(words([("Start.", 0.1, 0.5), ("Eine", 0.8, 1.0),
                                                             ("Milliarde.", 1.0, 1.5), ("Ende.", 1.8, 2.2)]))}
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 2.4]], "air": [[0.1, 0.2]],
                "piece_sents": [["S000", "S001", "S002"]]}
        import check
        original = check.load_audio
        check.load_audio = lambda *a, **k: np.zeros(16000, np.float32)
        try:
            errors, _ = word_report(Path("x.mp4"), clip, by_id, FakeModel([(" Start.", 0.1, 0.5), (" Unsinn.", 0.9, 1.4),
                                                                           (" Ende.", 1.8, 2.2)]))
        finally:
            check.load_audio = original
        self.assertTrue(any("S001" in e for e in errors))

    def test_acronym_at_edge_is_reviewed(self):
        """fehlt die Abkürzung am Ende, ist das „anhören“ statt still bestanden."""
        by_id = {s["id"]: s for s in build_sentences(words([("Das", 0.1, 0.3), ("sagt", 0.3, 0.6), ("BPI.", 0.6, 1.0)]))}
        clip = {"id": "k01", "speed": 1.0, "cut": [[0.0, 1.2]], "air": [[0.1, 0.2]], "piece_sents": [["S000"]]}
        import check
        original = check.load_audio
        check.load_audio = lambda *a, **k: np.zeros(16000, np.float32)
        try:
            _, review = word_report(Path("x.mp4"), clip, by_id, FakeModel([(" Das", 0.1, 0.3), (" sagt", 0.3, 0.6)]))
        finally:
            check.load_audio = original
        self.assertTrue(any("bpi" in r for r in review))

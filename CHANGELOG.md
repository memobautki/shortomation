# Änderungen

## 2.1.0 (17.11.2026)

- **Neu: Zuschnitt `face`** für Podcasts, Interviews und Vlogs. Das Bild folgt dem Gesicht, bei mehreren Personen
  der, die gerade spricht (Mundbewegung während Sprache, harter Schnitt am Satzanfang, frühestens nach 1,5 s). Die
  anderen Personen bleiben aus dem Bild. Einstellungen ohne Gesicht werden wie `content` geschnitten.
- Gesichter findet YuNet (OpenCV Model Zoo, MIT, 230 KB, liegt im Skill) über onnxruntime, das mit faster-whisper
  kommt. Kein neues Paket, alles lokal, keine Identifizierung von Personen.
- `analyze.py` schreibt eine Stichprobe der Gesichter (`faces` in `analysis.json`); Claude wählt damit den
  Zuschnitt von selbst.
- Lautheit: Leise Aufnahmen mit lauten Spitzen (typisch Podcast) bekommen vor dem Normieren eine leichte
  Kompression und erreichen so −14 LUFS. Bei allen anderen bleibt der Ton unverändert.
- 77 Tests (vorher 58).

## 2.0.0 (16.11.2026)

- Erste öffentliche Fassung: Transkript mit Wortzeiten (faster-whisper), Viralitätswert aus Inhaltsraster und
  Messung, Schnitt bis zum letzten Laut mit Silero-VAD gegen Hintergrundmusik, Rendern ohne Chrome und Node,
  Endkontrolle (ok, prüfen, durchgefallen) mit optionalem Wortabgleich.

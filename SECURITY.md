# Sicherheit

## Lücke melden

Bitte keine öffentlichen Issues für Sicherheitslücken. Nutze auf GitHub den Reiter **Security** und dort
**Report a vulnerability**. Die Meldung sieht nur der Betreiber des Repos.

Please do not open public issues for security problems. Use **Security → Report a vulnerability** on GitHub.

## Was der Skill tut und was nicht

- Er liest das Video, das du angibst, und schreibt nur in den Arbeitsordner, den du angibst.
- Er ruft ffmpeg und ffprobe mit festen Argumentlisten auf, nie über eine Shell.
- Netzwerk: Nur faster-whisper lädt beim ersten Lauf das Sprachmodell von Hugging Face. Sonst gibt es keine
  Verbindungen, keine Telemetrie und keine Uploads.
- Text aus dem Video (Transkript) behandelt Claude als Material, nicht als Anweisung.
- Gesichter werden nur gefunden, nicht erkannt. Es werden keine Gesichtsmerkmale gespeichert.

## Versionen

Unterstützt wird die jeweils neueste Fassung auf `main`.

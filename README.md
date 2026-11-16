# Shortomation

Ein Claude-Skill auf Deutsch, der aus einem langen Video mit Sprache fertige Hochkant-Shorts macht. Er findet die
stärksten Momente, gibt jedem einen Viralitätswert, schneidet wortgenau, setzt Hook und Untertitel im Takt der Wörter
und prüft jeden Short, bevor du ihn siehst. Alles läuft auf deinem Rechner, das Video wird nirgends hochgeladen.

Gebaut für Dokus und Erklärvideos (auch ohne Gesicht), funktioniert genauso mit Podcasts, Vorträgen und Interviews.

## Was du bekommst

- **3 bis 10 Shorts** je Video: 1080x1920, 30 fps, H.264, Lautheit −14 LUFS, dazu ein Vorschaubild.
- **Eine Rangliste** (`SCORES.md`) mit Viralitätswert: 65 % Inhalt (Hook, Verständlichkeit ohne Vorwissen,
  Auflösung, Fallhöhe, Überraschung, Diskussionsstoff), 35 % Messung (Länge, Sprechtempo, Pausen, Bildwechsel).
- **Einen Prüfbericht** (`CHECK.md`): Bildgröße, Länge, Lautheit, kein angeschnittenes Wort an den Schnitten
  (auch nicht geflüstert), auf Wunsch jedes Stück neu erkannt und mit dem geplanten Text abgeglichen.
- **Warnungen** bei Musik mit Content-ID-Anspruch, Material mit Namensnennungspflicht (CC BY) und Clips mit
  Vorwürfen gegen benannte Personen.

Der Viralitätswert ist eine Rangfolge, keine Vorhersage von Aufrufen.

## Voraussetzungen

| Was | Wofür | Pflicht |
| --- | --- | --- |
| Claude Code | Skill ausführen, Momente bewerten | ja |
| Python 3.10 oder neuer | Schnitt, Messung, Rendern | ja |
| ffmpeg und ffprobe | Video lesen und schreiben | ja |
| numpy, Pillow, faster-whisper | aus `requirements.txt` | ja |

**Windows (PowerShell):**

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

**Mac (Terminal):**

```bash
brew install python ffmpeg
```

Danach ein neues Terminal öffnen und prüfen: `python --version` und `ffmpeg -version`.

Beim ersten Lauf lädt faster-whisper das Sprachmodell einmal herunter (Modell „small“, rund 500 MB). Danach läuft
die Erkennung ohne Internet.

## Einrichten

**Als Plugin in Claude Code** (zwei Befehle in einem Claude-Code-Fenster):

```
/plugin marketplace add memobautki/shortomation
/plugin install shortomation@shortomation
```

**Oder von Hand:** Den Ordner `skills/shortomation` nach `~/.claude/skills/shortomation` kopieren.

**Python-Pakete:** Beim ersten Lauf legt Claude im Ordner des Skills eine eigene Python-Umgebung (`.venv`) an und
installiert numpy, Pillow und faster-whisper daraus. Es fragt vorher. Von Hand geht es so (im Ordner
`skills/shortomation`, bzw. dort, wohin du ihn kopiert hast):

```powershell
# Windows
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

```bash
# Mac / Linux
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Benutzen

Sag Claude zum Beispiel:

```text
Mach mir Shorts aus C:\Videos\folge12.mp4. Die Untertitel liegen daneben als folge12.srt.
```

Claude führt dann diese Schritte aus (Details in `skills/shortomation/SKILL.md`):

1. `transcribe.py`: Transkript mit Wortzeiten (lokal, faster-whisper; mit SRT die richtige Schreibweise)
2. `windows.py`: Sätze und Fenster von etwa 75 s
3. `analyze.py`: Schnitte, Bildbewegung, Pegel
4. Claude bewertet jedes Fenster und wählt die Clips (`clips.json`)
5. `score.py`: Schnittgrenzen und Viralitätswert
6. `render.py`: die fertigen Shorts
7. `check.py`: Prüfung, erst danach bekommst du die Shorts

Alles landet in einem Arbeitsordner, zum Beispiel `folge12/shorts/`:

```
shorts/
├── words.json · sentences.json · transcript.md · analysis.json
├── clips.json · clips.scored.json · SCORES.md · CHECK.md
└── out/
    ├── k01.mp4 · k01_cover.jpg
    └── k02.mp4 · k02_cover.jpg
```

### Bildzuschnitt

| `layout` | Wann |
| --- | --- |
| `content` (Standard) | Tafeln, Dokumente, Folien auf dunklem Grund: der Ausschnitt folgt dem Inhalt |
| `full` | Das ganze Bild bleibt sichtbar |
| `center` | Eine Person in der Bildmitte füllt die Fläche |

### Eigener Look

Farben, Größen und Schriften lassen sich in `clips.json` unter `style` ändern. Beispiel und alle Werte in
`SKILL.md`, Abschnitt „Stil anpassen“.

## Grenzen

- Der Skill braucht **Sprache** im Video. Reine Musik- oder Gameplay-Videos ohne Kommentar kann er nicht bewerten.
- Er verfolgt **keine Gesichter**. Für Interviews mit zwei Personen nebeneinander ist `full` meist die beste Wahl.
- **Sätze ohne Pause:** Gehen zwei Sätze nahtlos ineinander über, schneidet er an der leisesten Stelle zwischen den
  Wörtern und markiert den Short mit „prüfen“. Dann einmal anhören oder den Clip einen Satz länger wählen.
- **Flüstern unter lauter Musik** kann die Sprach-Erkennung übersehen. Auch dann steht „prüfen“ im Bericht, wenn am
  Schnitt ein leiser Laut liegt.
- **Content-ID** erkennt er nicht selbst. Die beanstandeten Bereiche trägst du aus YouTube Studio ein, dann schneidet
  er drumherum und warnt.
- Die **Erkennung auf der CPU** dauert etwa ein Viertel der Videolänge (Modell „small“).
- Der **Viralitätswert** ist eine begründete Rangfolge. Ob ein Short läuft, entscheiden die Zuschauer.
- Veröffentliche nur Videos, an denen du die Rechte hast oder für die du eine Erlaubnis hast.

## Datenschutz

Das Video und der Ton bleiben auf deinem Rechner. Die Spracherkennung läuft lokal. Claude liest den Text des
Transkripts, um die Momente zu bewerten; das geht wie jede Unterhaltung an deinen Claude-Zugang. Es gibt keine
Telemetrie und keine weiteren Dienste.

## Prüfen

```bash
python -m unittest discover -s tests -v
```

Die Unit-Tests prüfen Satzbildung, SRT-Abgleich, Schnittgrenzen (auch Flüstern über Raumton) und die
Eingabeprüfung. Zwei Ende-zu-Ende-Tests bauen künstliche Videos: eines mit leise ausklingendem letztem Wort
(1280x720, 25 fps), eines mit sechs Tonstößen, bei denen das Bild genau mit dem Ton aufleuchtet. Geprüft werden
Format, Länge, der erhaltene Ausklang und dass Bild und Ton nach sechs Schnitten und Tempo 1,1 höchstens ein Bild
auseinanderliegen. Die Tests brauchen ffmpeg, aber kein Sprachmodell und kein Internet.

## Lizenz

Code: MIT, siehe `LICENSE`. Schriften in `skills/shortomation/assets/fonts`: SIL Open Font License 1.1
(Big Shoulders Stencil Display, IBM Plex Sans, IBM Plex Mono), Lizenztexte liegen daneben.

Von memo.baut.ki

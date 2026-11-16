# Bewertungsraster für Shorts-Kandidaten

Der Viralitätswert (0–100) setzt sich aus zwei Teilen zusammen:

- **65 % Inhalt:** Das bewertest du als Agent nach diesem Raster.
- **35 % Messung:** Das rechnet `score.py` aus Bild, Ton und Transkript.

Dazu kommen Abzüge für Sätze, die ohne Vorwissen nicht verständlich sind. Der Wert ist eine Rangfolge, keine Vorhersage von
Aufrufen. Nach der Veröffentlichung gehören die echten Zahlen in `results.json` (siehe SKILL.md, Abschnitt Kalibrieren).

## Stufe 1: jedes Fenster grob bewerten (0–10)

Lies `transcript.md` ganz und gib **jedem** Fenster einen Wert für sein Short-Potenzial. Bewerte jedes Fenster,
statt nur Favoriten herauszupicken. Erst danach suchst du aus. Wer nur die ersten Treffer nimmt, übersieht die Mitte
des Videos.

- 9–10: Ein Moment, den man weitererzählt (Enthüllung, Widerspruch, absurde Wendung, starke These) und der für sich
  verständlich ist.
- 6–8: Starker Inhalt, braucht aber einen Satz Kontext oder hat kein klares Ende.
- 3–5: Erklärung, Überleitung, Aufzählung von Daten.
- 0–2: Intro, Werbung, Quellenhinweis, Zusammenfassung, Abspann.

## Stufe 2: Clips schneiden (aus den besten Fenstern)

Wähle aus den Fenstern ab Wert 7 die Clips. Ihre Zahl ergibt sich aus der Videolänge: etwa einer pro 4 Minuten,
mindestens 3 und höchstens 10. Fülle nicht mit schwachen Clips auf, eine kurze Liste ist besser als Füller.

**Grenzen:** Ein Clip ist eine Folge von Satzbereichen (`parts`, Satz-IDs). Normalerweise ist es ein Bereich. Zwei
Bereiche nur dann, wenn ein Kontextsatz von weiter vorn den Clip verständlich macht. Ein Clip darf sich höchstens zu
30 % mit einem anderen überschneiden.

- **Erster Satz:** Er muss für sich stehen. Er beginnt nicht mit „Und“, „Aber“, „Denn“ oder mit einem Pronomen ohne
  Bezug („Er“, „Dort“). Er verweist nicht auf andere Stellen („Erinnerst du dich?“, „wie am Anfang“, „Kapitel 3“).
  Wenn der stärkste Satz so beginnt, nimm einen Satz früher dazu oder schneide den Bezug weg.
- **Letzter Satz:** Er löst etwas auf oder lässt eine scharfe Frage stehen. Er endet nie mitten in einer Aufzählung.
- **Länge:** Ideal sind 25–50 s. Unter 15 s fehlt die Geschichte, über 60 s nur mit sehr starkem Inhalt.

## Stufe 2: Inhaltsraster je Clip (0–10)

| Feld | Gewicht | 10 heißt | 3 heißt |
|---|---|---|---|
| `hook` | 25 % | In den ersten 2 s eine Zahl, ein Widerspruch oder eine offene Frage, die man klären will | Der Anfang erklärt erst einmal etwas |
| `standalone` | 20 % | Jemand ohne das ganze Video versteht alles | Namen und Begriffe fallen ohne Einführung |
| `payoff` | 15 % | Es gibt eine Auflösung oder einen klaren Knall | Der Clip hört einfach auf |
| `stakes` | 15 % | Große, greifbare Fallhöhe (Geld, Gesundheit, Gefängnis, Existenz) | Abstrakte Abläufe |
| `surprise` | 15 % | Etwas Unerwartetes oder Empörendes | Erwartbar |
| `comment` | 10 % | Man will widersprechen, raten oder eine Meinung schreiben | Nichts zu sagen |

## `clips.json`

```json
{
  "video": "../mein-video.mp4",
  "channel": "MEIN KANAL",
  "cta": "Ganzes Video: auf dem Kanal",
  "speed": 1.0,
  "layout": "content",
  "claimed": [[1101.0, 1217.0]],
  "attribution": [[69.4, 72.3, "Foto: Name, CC BY 4.0"]],
  "windows": {"W00": 9, "W01": 6},
  "clips": [
    {
      "id": "k01",
      "label": "FOLGE 12 · CLIP 1",
      "hook": "1,9 Milliarden. Einfach weg?",
      "title": "Wie verschwinden 1,9 Milliarden Euro? #wirtschaft",
      "parts": [["S002", "S012"]],
      "rubric": {"hook": 9, "standalone": 8, "payoff": 7, "stakes": 9, "surprise": 8, "comment": 7},
      "why": "Kalter Einstieg mit Zahl, offene Frage am Ende"
    }
  ]
}
```

- **`video`:** Pfad zum langen Video, relativ zum Arbeitsordner (oder absolut).
- **`hook`:** Der Text oben im Bild, höchstens 6 Wörter und etwa 32 Zeichen. Er muss stimmen und darf nicht mehr
  versprechen als der Clip zeigt.
- **`label`:** Reihe und Nummer, klein über dem Hook.
- **`title`:** Der Titel für YouTube, höchstens 70 Zeichen.
- **`windows`:** Deine Werte aus Stufe 1 (zur Nachvollziehbarkeit).
- Je Clip lassen sich `speed`, `tighten` und `layout` einzeln überschreiben.

## Was `score.py` misst (35 %, je 0–1; Gewichte: Länge 20, Tempo 20, Pausen 15, Bildwechsel 15, tote Strecke 10, Einstieg 20)

| Messwert | 1,0 bei | 0 bei |
|---|---|---|
| Länge | 25–50 s | unter 12 s oder über 90 s |
| Sprechtempo (Wörter pro Sprechsekunde) | 2,2–3,2 | unter 1,3 oder über 4,2 |
| Längste Pause zwischen Sätzen | höchstens 1,2 s | ab 3,0 s |
| Bildwechsel (Schnitte je 10 s, Bewegung) | ab 2,5 Schnitten je 10 s | Standbild |
| Längste tote Strecke (nur Korn, keine Bewegung) | höchstens 1,5 s | ab 4 s (render.py mildert mit Push-in) |
| Schneller Einstieg (erster Satz kurz, Stimme sofort) | höchstens 12 Wörter, Stimme ab 0,4 s | über 25 Wörter |

## Abzüge und Warnungen

- **−8 je Kontextbezug** im ersten oder zweiten Satz, höchstens −16. Dazu zählen ein Satzanfang mit Und/Aber/Denn,
  ein Pronomen ohne Bezug, „Erinnerst du dich“, „Kapitel“, „am Anfang“ und „wie gesagt“.
- **Warnung `Content-ID-Musik`:** Der Clip überschneidet einen beanstandeten Musikbereich (`claimed`). Nicht
  veröffentlichen oder vorher die Musik austauschen.
- **Warnung `Nachweis nötig`:** Der Clip enthält Material mit Namensnennungspflicht (`attribution`).
- **Warnung `⚖ prüfen`:** Vorwürfe gegen benannte Personen (Betrug, gefälscht, Haftbefehl, Spionage …). Kein Abzug,
  aber der Clip zeigt den Vorwurf ohne die Einordnung aus dem Video. Vor dem Veröffentlichen rechtlich prüfen, ob die
  Einschränkung („mutmaßlich“, „laut Anklage“, Unschuldsvermutung) im Clip mit drin ist.

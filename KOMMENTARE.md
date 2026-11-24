# Kommentare zum Einfügen

Jeder Block kommt **direkt über** die angegebene Zeile, mit derselben Einrückung.
Die Liste läuft in jeder Datei von unten nach oben. Arbeite sie der Reihe nach ab, dann stimmen die Zeilennummern.
Ändere die Formulierungen ruhig und lass weg, was du selbst klar findest.

## daten_vorbereiten.py

**Zeile 52:** `zaehler = Counter(p.parent.name for p in ziel.glob("*/*.jpg"))`
```python
    # Kurzer Check, ob alle Bilder da sind
```

**Zeile 46:** `meta = {"datensatz": "resisc45", "klassennamen": {k: k.replace("_", " ") for k in klassen},`
```python
    # Klassennamen und Prompts braucht später CLIP in Station 3
```

**Zeile 39:** `if ziel.exists():`
```python
    # Achtung: löscht den alten Ordner, danach muss neu bereinigt werden
```

**Zeile 28:** `def verlinken(quelle, ziel):`
```python
# Hardlink spart Platz, die Bilder liegen sonst doppelt auf der Platte.
# Geht das nicht (z. B. anderes Laufwerk), wird normal kopiert.
```

**Zeile 22:** `sha = hashlib.sha256(zip_pfad.read_bytes()).hexdigest()`
```python
        # Prüfen, ob die ZIP vollständig geladen wurde
```

**Zeile 10:** `RESISC45_URL = ("https://hf.co/datasets/isaaccorley/resisc45/resolve/"`
```python
# Feste Version vom Datensatz, damit bei allen dasselbe ankommt
```

## daten.py

**Zeile 46:** `def anzahl_bilder_pro_klasse(ds):`
```python
# Zum Nachschauen, wie viele Bilder jede Klasse im Split hat
```

**Zeile 29:** `train_idx, val_idx = train_test_split(rest, test_size=anteil_val / (1 - anteil_test),`
```python
    # 15 % vom Ganzen sind hier mehr als 15 % vom Rest, deshalb das Umrechnen
```

**Zeile 28:** `rest, test_idx = train_test_split(alle, test_size=anteil_test, stratify=labels, random_state=42)`
```python
    # Erst 15 % Test abtrennen, stratify hält die Klassen gleich verteilt.
    # Seed ist fest, also haben alle Modelle genau dieselben Testbilder.
```

**Zeile 24:** `mit_aug = datasets.ImageFolder(ordner, transform=train_tf)`
```python
    # Zweimal derselbe Ordner, einmal mit und einmal ohne Augmentierung
```

**Zeile 15:** `eval_tf = transforms.Compose([`
```python
    # Val und Test immer gleich, sonst sind die Werte nicht vergleichbar
```

**Zeile 9:** `train_tf = transforms.Compose([`
```python
    # Nur im Training wird zufällig zugeschnitten und gespiegelt
```

## datenqualitaet.py

**Zeile 87:** `for p in sorted(weg):`
```python
        # Nur verschieben, nicht löschen, falls man etwas zurückholen will
```

**Zeile 86:** `weg.update(g[1:])`
```python
                # Von Duplikaten bleibt das erste Bild drin
```

**Zeile 82:** `for g in widerspruch:`
```python
        # Bei widersprüchlichen Labels fliegen alle Bilder der Gruppe raus
```

**Zeile 49:** `widerspruch = {frozenset(g) for g in exakt + beinahe if len({p.parent.name for p in g}) > 1}`
```python
    # Gleiches Bild in zwei Klassen, da kann mindestens ein Label nicht stimmen
```

**Zeile 47:** `flach = sum(len(g) for h, g in nach_dhash.items() if bin(h).count("1") < 4 or bin(h).count("1") > 60)`
```python
    # Sehr einfarbige Bilder haben fast alle denselben Hash, deshalb extra zählen
```

**Zeile 46:** `exakt = [g for g in nach_md5.values() if len(g) > 1]`
```python
    # Gleicher MD5 heißt: Datei ist Byte für Byte identisch
```

**Zeile 35:** `for p in dateien:`
```python
    # Jedes Bild einmal öffnen, dabei fallen kaputte Dateien auf
```

**Zeile 18:** `def dhash(bild, groesse=8):`
```python
# Vergleicht Nachbarpixel (heller oder dunkler) und macht daraus 64 Bit.
# Bilder, die fast gleich aussehen, bekommen denselben Hash.
```

## lora.py

**Zeile 49:** `if __name__ == "__main__":`
```python
# Kleiner Test: vor und nach dem Zusammenführen muss dasselbe rauskommen
```

**Zeile 44:** `kind.original.weight += kind.skalierung * (kind.B @ kind.A)`
```python
                    # Korrektur fest auf W addieren, das Ergebnis bleibt gleich
```

**Zeile 34:** `setattr(eltern, name, LoRALinear(getattr(eltern, name), r=r, alpha=alpha))`
```python
            # Alte Schicht durch die LoRA-Variante austauschen
```

**Zeile 24:** `ZIELE = {`
```python
# Wo die Schichten in einem timm-ViT-Block liegen
```

**Zeile 21:** `return self.original(x) + (x @ self.A.T @ self.B.T) * self.skalierung`
```python
        # Normale Ausgabe plus die gelernte Korrektur
```

**Zeile 17:** `self.B = nn.Parameter(torch.zeros(original.out_features, r))`
```python
        # B startet mit 0, am Anfang rechnet das Modell also wie das vortrainierte
```

**Zeile 16:** `self.A = nn.Parameter(torch.empty(r, original.in_features))`
```python
        # A und B sind klein, zusammen deutlich weniger Parameter als W
```

**Zeile 15:** `self.skalierung = alpha / r`
```python
        # Damit die Stärke der Korrektur nicht direkt vom Rang abhängt
```

**Zeile 7:** `class LoRALinear(nn.Module):`
```python
# Ersetzt eine normale Linear-Schicht: W bleibt fest, gelernt werden nur A und B
```

## modelle.py

**Zeile 30:** `modell.load_state_dict(daten["zustand"], strict=not nur_teilweise)`
```python
    # Fehlt ein Teil absichtlich, darf load_state_dict nicht meckern
```

**Zeile 19:** `zustand.update({n: b.detach().cpu() for n, b in modell.named_buffers()})`
```python
        # Buffer mit speichern, z. B. die BatchNorm-Werte vom ResNet
```

**Zeile 17:** `if konfig["methode"] in ("lp", "lora") and konfig["vortrainiert"]:`
```python
    # Bei lp und lora reicht das Trainierte, der Rest kommt wieder von timm
```

**Zeile 12:** `for p in modell.get_classifier().parameters():`
```python
        # Der Kopf ist neu und muss immer trainiert werden
```

**Zeile 11:** `lora_einbauen(modell, r=rang, alpha=2 * rang, ziele=ziele)`
```python
            # alpha = 2 * Rang, dadurch ist die Skalierung bei jedem Rang 2
```

**Zeile 8:** `for p in modell.parameters():`
```python
        # Erstmal alles einfrieren
```

**Zeile 6:** `modell = timm.create_model(name, pretrained=vortrainiert, num_classes=anzahl_klassen)`
```python
    # timm setzt direkt einen neuen Kopf mit unserer Klassenzahl drauf
```

## werkzeuge.py

**Zeile 84:** `return max(zeilen, key=lambda z: float(z["val_acc"]))["checkpoint"]`
```python
    # Auswahl nach Validierung, der Testsatz bleibt außen vor
```

**Zeile 63:** `return sorted(zeiten)[len(zeiten) // 2]`
```python
    # Median statt Mittelwert, Ausreißer stören dann nicht so
```

**Zeile 61:** `if i >= 10:`
```python
            # Die ersten 10 Durchläufe sind nur zum Aufwärmen
```

**Zeile 48:** `def warten():`
```python
    # Die GPU rechnet im Hintergrund weiter, ohne Warten wäre die Zeit zu kurz
```

**Zeile 38:** `neu = not os.path.exists(pfad)`
```python
    # Kopfzeile nur schreiben, wenn es die Datei noch nicht gibt
```

**Zeile 22:** `def setze_seed(seed):`
```python
# Gleicher Seed, damit man Läufe wiederholen kann
```

**Zeile 14:** `def waehle_geraet():`
```python
# Mac-GPU, NVIDIA oder CPU, je nachdem was der Rechner hat
```

## train.py

**Zeile 130:** `modell = lora_zusammenfuehren(modell)`
```python
    # Vor der Latenzmessung zusammenführen, so würde man es auch einsetzen
```

**Zeile 121:** `if val_acc > beste_val_acc:`
```python
        # Nur das beste Modell nach Val-Accuracy behalten
```

**Zeile 100:** `optimizer = torch.optim.AdamW([p for p in modell.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)`
```python
    # Nur die Parameter übergeben, die wirklich trainiert werden
```

**Zeile 88:** `cfg = modell.pretrained_cfg`
```python
    # Jedes Modell erwartet eine eigene Normierung, die steht in der timm-Konfig
```

**Zeile 68:** `scheduler.step()`
```python
        # Lernrate sinkt nach jedem Batch, nicht erst nach der Epoche
```

**Zeile 66:** `verlust.backward()`
```python
        # Gradienten berechnen und Gewichte anpassen
```

**Zeile 47:** `def auswerten(modell, loader, verlust_fn, geraet):`
```python
# Für Val und Test: Labels, Vorhersagen und mittlerer Verlust
```

**Zeile 38:** `def lauf_name(args):`
```python
# Alle wichtigen Einstellungen in den Namen, sonst überschreiben sich Läufe
```

## distill.py

**Zeile 118:** `student.load_state_dict(torch.load(checkpoint, map_location="cpu")["zustand"])`
```python
    # Für den Test wieder das beste Modell laden, nicht das letzte
```

**Zeile 99:** `logits_t = teacher(fuer_teacher(bilder)) if args.alpha > 0 else None`
```python
                # Teacher sieht genau dieselben Bilder, ohne KD wird er nicht gebraucht
```

**Zeile 71:** `assert klassen == t_konfig["klassen"], "Teacher wurde auf anderen Klassen trainiert"`
```python
    # Sonst würden die Ausgaben von Teacher und Student nicht zusammenpassen
```

**Zeile 60:** `p.requires_grad = False`
```python
        # Der Teacher lernt nicht mehr mit
```

**Zeile 56:** `teacher_pfad = args.teacher or bester_checkpoint(ordner, "lora")`
```python
    # Ohne Angabe nehmen wir den besten LoRA-Lauf aus Station 1
```

**Zeile 43:** `def normierung_umrechnen(cfg_s, cfg_t, geraet):`
```python
# Teacher und Student können anders normiert sein, hier wird umgerechnet
```

**Zeile 40:** `kd = F.kl_div(F.log_softmax(logits_s / T, dim=1), F.softmax(logits_t / T, dim=1), reduction="batchmean") * T * T`
```python
    # Hohe Temperatur macht die Verteilung weicher, so sieht der Student auch,
    # welche Klassen der Teacher ähnlich findet. T * T gleicht die kleineren Gradienten aus.
```

**Zeile 36:** `def kd_verlust(logits_s, logits_t, y, alpha, T):`
```python
# Mischung aus normalem Verlust und Abstand zum Teacher.
# alpha sagt, wie stark der Teacher zählt, bei 0 ist es normales Training.
```

## clip_zeroshot.py

**Zeile 73:** `text = text_merkmale(namen, vorlagen).T`
```python
            # Es gewinnt die Klasse, deren Text am besten zum Bild passt
```

**Zeile 58:** `def text_merkmale(namen, vorlagen):`
```python
    # Pro Klasse ein Vektor, bei mehreren Vorlagen der Mittelwert
```

**Zeile 51:** `merkmale.append(m / m.norm(dim=-1, keepdim=True))`
```python
                # Auf Länge 1 bringen, dann ist das Skalarprodukt die Cosinus-Ähnlichkeit
```

**Zeile 39:** `prompts = {`
```python
    # Vier Varianten, um zu sehen wie viel der Text ausmacht
```

**Zeile 32:** `_, val_ds, test_ds, klassen = lade_daten(args.daten, vorverarbeitung, vorverarbeitung)`
```python
    # Gleicher Split wie in Station 1 und 2, aber mit der Vorverarbeitung von CLIP
```

**Zeile 27:** `modell, _, vorverarbeitung = open_clip.create_model_and_transforms(args.modell, pretrained=args.pretrained)`
```python
    # CLIP wird hier nicht trainiert, nur ausgewertet
```

## zusammenfassung.py

**Zeile 74:** `luecke = ""`
```python
            # Wie viel vom Abstand zum Teacher holt der Student durch KD auf?
```

**Zeile 57:** `if k[4] == "ja":`
```python
            # Ohne Vortraining ist nur ein Kontrollversuch und kommt nicht in den Vergleich
```

**Zeile 22:** `def mittel(zeilen, spalte):`
```python
# Bei nur einem Lauf gibt es keine Streuung, dann 0
```

**Zeile 15:** `def gruppieren(zeilen, schluessel):`
```python
# Läufe mit gleichen Einstellungen zusammenfassen, übrig bleiben die Seeds
```

## benchmark.py

**Zeile 194:** `subprocess.run([sys.executable, __file__, "--nur", name] + weiter, check=True)`
```python
            # Jedes Modell in einem neuen Prozess, sonst bleibt Speicher vom vorigen belegt
```

**Zeile 145:** `ziel = Path(pfad).with_name(Path(pfad).stem + "_median.csv")`
```python
    # Eigene Datei, damit die Rohwerte erhalten bleiben
```

**Zeile 109:** `if i >= args.aufwaermen:`
```python
        # Erste Schritte sind langsamer und zählen nicht mit
```

**Zeile 94:** `y = torch.randint(0, 45, (args.batch,), device=geraet)`
```python
    # Zufallsdaten reichen, es geht nur um Zeit und Speicher
```

**Zeile 62:** `def speicher_mb(geraet):`
```python
# Auf der CPU gibt es hier keinen sinnvollen Wert, deshalb nan
```

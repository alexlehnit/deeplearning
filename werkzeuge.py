import random
import numpy as np
import torch
import os
from pathlib import Path
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Mac-GPU, NVIDIA oder CPU, je nachdem was der Rechner hat
def waehle_geraet():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Gleicher Seed, damit man Läufe wiederholen kann
def setze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ausgabe_ordner(daten):
    ordner = Path("ausgaben") / Path(daten).name
    for unter in ("modelle", "konfusion", "runs"):
        (ordner / unter).mkdir(parents=True, exist_ok=True)
    return ordner


def anzahl_klassen(daten):
    return len([d for d in os.listdir(daten) if os.path.isdir(os.path.join(daten, d))])

def csv_anhaengen(pfad, zeile):
    # Kopfzeile nur schreiben, wenn es die Datei noch nicht gibt
    neu = not os.path.exists(pfad)
    with open(pfad, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(zeile.keys()))
        if neu:
            w.writeheader()
        w.writerow(zeile)

def latenz_ms(vorwaerts, geraet, groesse=224, wiederholungen=50):
    bild = torch.randn(1, 3, groesse, groesse, device=geraet)

    # Die GPU rechnet im Hintergrund weiter, ohne Warten wäre die Zeit zu kurz
    def warten():
        if geraet.type == "mps":
            torch.mps.synchronize()
        elif geraet.type == "cuda":
            torch.cuda.synchronize()

    zeiten = []
    with torch.no_grad():
        for i in range(wiederholungen + 10):
            warten()
            t0 = time.perf_counter()
            vorwaerts(bild)
            warten()
            # Die ersten 10 Durchläufe sind nur zum Aufwärmen
            if i >= 10:
                zeiten.append((time.perf_counter() - t0) * 1000)
    # Median statt Mittelwert, Ausreißer stören dann nicht so
    return sorted(zeiten)[len(zeiten) // 2]

def konfusionsmatrix(wahr, vorhersage, klassen, pfad, titel=""):
    cm = confusion_matrix(wahr, vorhersage, labels=list(range(len(klassen))))
    groesse = max(6, 0.28 * len(klassen))
    fig = plt.figure(figsize=(groesse + 1, groesse))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(klassen)), klassen, rotation=90, fontsize=5)
    plt.yticks(range(len(klassen)), klassen, fontsize=5)
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahr")
    plt.title(titel, fontsize=9)
    plt.tight_layout()
    fig.savefig(pfad, dpi=130)
    return fig

def bester_checkpoint(ordner, methode="lora"):
    with open(Path(ordner) / "ergebnisse.csv") as f:
        # wiederholte Läufe überschreiben den Checkpoint, also zählt nur ihre letzte Zeile
        zeilen = [z for z in {z["checkpoint"]: z for z in csv.DictReader(f)}.values()
                  if z["methode"] == methode and z["vortrainiert"] == "ja" and os.path.exists(z["checkpoint"])]
    # Auswahl nach Validierung, der Testsatz bleibt außen vor
    return max(zeilen, key=lambda z: float(z["val_acc"]))["checkpoint"]

def vorhersagen(modell, loader, geraet, vorverarbeitung=None):
    """Gibt (wahre Labels, vorhergesagte Labels) zurück; vorverarbeitung z. B. Normierung für den Teacher."""
    modell.eval()
    alle_y, alle_p = [], []
    with torch.no_grad():
        for bilder, y in loader:
            bilder = bilder.to(geraet)
            if vorverarbeitung is not None:
                bilder = vorverarbeitung(bilder)
            alle_y += y.tolist()
            alle_p += modell(bilder).argmax(1).cpu().tolist()
    return alle_y, alle_p

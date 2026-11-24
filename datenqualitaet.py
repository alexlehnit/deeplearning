import numpy as np
from PIL import Image

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil




BILDENDUNGEN = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff", ".webp"}

# Vergleicht Nachbarpixel (heller oder dunkler) und macht daraus 64 Bit.
# Bilder, die fast gleich aussehen, bekommen denselben Hash.
def dhash(bild, groesse=8):
    grau = np.asarray(bild.convert("L").resize((groesse + 1, groesse), Image.BILINEAR), dtype=np.int16)
    bits = (grau[:, 1:] > grau[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daten", default="daten/resisc45")
    parser.add_argument("--bereinigen", action="store_true")
    parser.add_argument("--beinahe_entfernen", action="store_true")
    args = parser.parse_args()

    ordner = Path(args.daten)
    dateien = sorted(p for p in ordner.glob("*/*") if p.suffix.lower() in BILDENDUNGEN)
    pro_klasse = Counter(p.parent.name for p in dateien)
    groessen, modi, defekt = Counter(), Counter(), []
    nach_md5, nach_dhash = defaultdict(list), defaultdict(list)
    # Jedes Bild einmal öffnen, dabei fallen kaputte Dateien auf
    for p in dateien:
        try:
            with Image.open(p) as bild:
                bild.load()
                groessen[bild.size] += 1
                modi[bild.mode] += 1
                nach_dhash[dhash(bild)].append(p)
            nach_md5[hashlib.md5(p.read_bytes()).hexdigest()].append(p)
        except Exception as fehler:
            defekt.append((p, str(fehler)))

    # Gleicher MD5 heißt: Datei ist Byte für Byte identisch
    exakt = [g for g in nach_md5.values() if len(g) > 1]
    # Sehr einfarbige Bilder haben fast alle denselben Hash, deshalb extra zählen
    flach = sum(len(g) for h, g in nach_dhash.items() if bin(h).count("1") < 4 or bin(h).count("1") > 60)
    beinahe = [g for h, g in nach_dhash.items() if len(g) > 1 and 4 <= bin(h).count("1") <= 60]
    # Gleiches Bild in zwei Klassen, da kann mindestens ein Label nicht stimmen
    widerspruch = {frozenset(g) for g in exakt + beinahe if len({p.parent.name for p in g}) > 1}

    ziel = ordner.parent / f"{ordner.name}_aussortiert"
    zeilen = [
        f"Bilder: {len(dateien):,} in {len(pro_klasse)} Klassen",
        f"Bilder pro Klasse: min {min(pro_klasse.values())}, max {max(pro_klasse.values())}",
        f"Bildgrößen: {dict(groessen)}",
        f"Farbmodi: {dict(modi)}",
        f"Defekte Dateien: {len(defekt)}",
        f"Exakte Duplikate: {len(exakt)} Gruppen, {sum(len(g) - 1 for g in exakt)} überzählige Dateien",
        f"Beinahe-Duplikate: {len(beinahe)} Gruppen",
        f"Widersprüchliche Labels: {len(widerspruch)} Gruppen",
        f"Sehr kontrastarme Bilder: {flach}",
        f"Vor dieser Prüfung aussortiert: {len(list(ziel.glob('*/*')))} Dateien",
    ]
    if beinahe:
        zeilen += ["", "Beinahe-Duplikate (erste 30):"] + [
            "  " + " | ".join(str(p.relative_to(ordner)) for p in g) for g in beinahe[:30]]

    ausgabe = Path("ausgaben") / ordner.name
    ausgabe.mkdir(parents=True, exist_ok=True)
    (ausgabe / "datenqualitaet.txt").write_text("\n".join(zeilen) + "\n")
    print("\n".join(zeilen[:10]))

    klassen = sorted(pro_klasse)
    plt.figure(figsize=(max(6, 0.22 * len(klassen)), 4))
    plt.bar(range(len(klassen)), [pro_klasse[k] for k in klassen])
    plt.xticks(range(len(klassen)), [k[:20] for k in klassen], rotation=90, fontsize=6)
    plt.ylabel("Bilder")
    plt.tight_layout()
    plt.savefig(ausgabe / "klassenverteilung.png", dpi=130)
    if args.bereinigen:
        weg = {p for p, _ in defekt}
        # Bei widersprüchlichen Labels fliegen alle Bilder der Gruppe raus
        for g in widerspruch:
            weg.update(g)
        for g in exakt + (beinahe if args.beinahe_entfernen else []):
            if frozenset(g) not in widerspruch:
                # Von Duplikaten bleibt das erste Bild drin
                weg.update(g[1:])
        # Nur verschieben, nicht löschen, falls man etwas zurückholen will
        for p in sorted(weg):
            (ziel / p.parent.name).mkdir(parents=True, exist_ok=True)
            # sonst lässt rename die Quelle liegen, wenn beide Hardlinks auf dieselbe Datei sind
            (ziel / p.parent.name / p.name).unlink(missing_ok=True)
            shutil.move(str(p), ziel / p.parent.name / p.name)
        print(f"Bereinigt: {len(weg)} Dateien nach {ziel}/ verschoben.")


if __name__ == "__main__":
    main()

import hashlib
from pathlib import Path
from torchvision.datasets.utils import download_url, extract_archive
import os
import shutil
import json
from collections import Counter
from PIL import Image

RESISC45_URL = ("https://hf.co/datasets/isaaccorley/resisc45/resolve/"
                "883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip")
RESISC45_SHA256 = "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"



def herunterladen(roh):
    ziel = Path(roh) / "resisc45"
    ordner = ziel / "NWPU-RESISC45"
    if not ordner.is_dir():
        download_url(RESISC45_URL, str(ziel), filename="NWPU-RESISC45.zip")
        zip_pfad = ziel / "NWPU-RESISC45.zip"
        sha = hashlib.sha256(zip_pfad.read_bytes()).hexdigest()
        if sha != RESISC45_SHA256:
            print("Warnung: Prüfsumme weicht ab. Download unvollständig?")
        extract_archive(str(zip_pfad), str(ziel))
    return ordner

def verlinken(quelle, ziel):
    try:
        os.link(quelle, ziel)
    except OSError:
        shutil.copy2(quelle, ziel)

def main():
    quelle = herunterladen("daten/roh")
    klassen = sorted(d.name for d in quelle.iterdir() if d.is_dir())

    ziel = Path("daten") / "resisc45"
    if ziel.exists():
        shutil.rmtree(ziel)
    for klasse in klassen:
        (ziel / klasse).mkdir(parents=True)
        for p in sorted((quelle / klasse).glob("*.jpg")):
            verlinken(p, ziel / klasse / p.name)

    meta = {"datensatz": "resisc45", "klassennamen": {k: k.replace("_", " ") for k in klassen},
            "prompts": ["a satellite photo of {}.", "a centered satellite photo of {}.", "an aerial photo of {}.",
                        "satellite imagery of {}.", "an overhead view of {}."]}
    with open(ziel / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    zaehler = Counter(p.parent.name for p in ziel.glob("*/*.jpg"))
    print(f"{sum(zaehler.values()):,} Bilder in {len(zaehler)} Klassen "
          f"(min {min(zaehler.values())}, max {max(zaehler.values())})")
    print("Bildgröße:", Image.open(next(ziel.glob("*/*.jpg"))).size)


if __name__ == "__main__":
    main()
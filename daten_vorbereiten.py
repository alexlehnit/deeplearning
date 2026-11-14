import hashlib
from pathlib import Path
from torchvision.datasets.utils import download_url, extract_archive
import os
import shutil

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
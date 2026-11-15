#!/usr/bin/env bash
set -e
DATEN=daten/resisc45

python clip_zeroshot.py --daten "$DATEN" --modell ViT-B-32
python clip_zeroshot.py --daten "$DATEN" --modell ViT-B-16
# optional, groß und langsam:
# python clip_zeroshot.py --daten "$DATEN" --modell ViT-L-14

python zusammenfassung.py --daten "$DATEN"
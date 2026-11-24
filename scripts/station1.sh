#!/usr/bin/env bash
set -e
DATEN=daten/resisc45
EPOCHEN=${EPOCHEN:-10}
BATCH=${BATCH:-32}

[ -d "$DATEN" ] || python daten_vorbereiten.py
python datenqualitaet.py --daten "$DATEN" --bereinigen

T="python train.py --daten $DATEN --epochen $EPOCHEN --batch $BATCH"

for SEED in 0 1 2; do $T --methode lp --seed $SEED; done

for RANG in 4 16 32; do $T --methode lora --rang $RANG --lr 5e-4; done
for SEED in 0 1 2; do $T --methode lora --rang 8 --lr 5e-4 --seed $SEED; done
$T --methode lora --rang 8 --lr 5e-4 --lora_ziele qkv,proj,fc1,fc2

$T --modell resnet50.a1_in1k --methode lp
$T --modell resnet50.tv_in1k --methode lp
for SEED in 0 1 2; do $T --modell resnet50.a1_in1k --methode full --lr 3e-4 --seed $SEED; done
$T --modell resnet50.tv_in1k --methode full --lr 3e-4

for SEED in 0 1 2; do $T --methode full --lr 3e-5 --seed $SEED; done

$T --methode full --lr 1e-4 --ohne_vortraining
$T --modell resnet50.a1_in1k --methode full --lr 1e-3 --ohne_vortraining

python zusammenfassung.py --daten "$DATEN"
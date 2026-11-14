#!/usr/bin/env bash
set -e
DATEN=daten/resisc45
EPOCHEN=${EPOCHEN:-10}
BATCH=${BATCH:-32}
TINY=vit_tiny_patch16_224.augreg_in21k_ft_in1k
SMALL=vit_small_patch16_224.augreg_in21k_ft_in1k

D="python distill.py --daten $DATEN --epochen $EPOCHEN --batch $BATCH"

for SEED in 0 1 2; do
  $D --student $TINY --alpha 0 --seed $SEED
  $D --student $TINY --alpha 0.7 --temperatur 4 --seed $SEED
done
$D --student $SMALL --alpha 0
$D --student $SMALL --alpha 0.7 --temperatur 4

for T in 2 8; do $D --student $TINY --alpha 0.7 --temperatur $T; done

python zusammenfassung.py --daten "$DATEN"
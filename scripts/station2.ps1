# PowerShell Entsprechung des Distillations-Skripts
$ErrorActionPreference = "Stop"

$DATEN = "daten/resisc45"
$EPOCHEN = if ($env:EPOCHEN) { $env:EPOCHEN } else { 10 }
$BATCH = if ($env:BATCH) { $env:BATCH } else { 32 }

$TINY = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
$SMALL = "vit_small_patch16_224.augreg_in21k_ft_in1k"

# Hilfsfunktion als Ersatz für den Bash-Befehl $D
function Invoke-Distill {
    param([string[]]$ExtraArgs)
    python distill.py --daten $DATEN --epochen $EPOCHEN --batch $BATCH @ExtraArgs
}

# --- Distillationsläufe ---

foreach ($seed in 0, 1, 2) {
    Invoke-Distill "--student", $TINY, "--alpha", "0", "--seed", "$seed"
    Invoke-Distill "--student", $TINY, "--alpha", "0.7", "--temperatur", "4", "--seed", "$seed"
}

Invoke-Distill "--student", $SMALL, "--alpha", "0"
Invoke-Distill "--student", $SMALL, "--alpha", "0.7", "--temperatur", "4"

foreach ($t in 2, 8) {
    Invoke-Distill "--student", $TINY, "--alpha", "0.7", "--temperatur", "$t"
}

python zusammenfassung.py --daten $DATEN
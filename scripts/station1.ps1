# PowerShell Entsprechung des Bash-Skripts
$ErrorActionPreference = "Stop"

$DATEN = "daten/resisc45"
$EPOCHEN = if ($env:EPOCHEN) { $env:EPOCHEN } else { 10 }
$BATCH = if ($env:BATCH) { $env:BATCH } else { 32 }

# Prüfen, ob das Datenverzeichnis existiert, ansonsten Vorbereitung ausführen
if (-not (Test-Path -Path $DATEN -PathType Container)) {
    python daten_vorbereiten.py
}
python datenqualitaet.py --daten $DATEN

# Hilfsfunktion als Ersatz für den Bash-Befehl $T
function Invoke-Train {
    param([string[]]$ExtraArgs)
    python train.py --daten $DATEN --epochen $EPOCHEN --batch $BATCH @ExtraArgs
}

# --- Trainingsläufe ---

foreach ($seed in 0, 1, 2) {
    Invoke-Train "--methode", "lp", "--seed", "$seed"
}

foreach ($rang in 4, 16, 32) {
    Invoke-Train "--methode", "lora", "--rang", "$rang", "--lr", "5e-4"
}

foreach ($seed in 0, 1, 2) {
    Invoke-Train "--methode", "lora", "--rang", "8", "--lr", "5e-4", "--seed", "$seed"
}

Invoke-Train "--methode", "lora", "--rang", "8", "--lr", "5e-4", "--lora_ziele", "qkv,proj,fc1,fc2"

Invoke-Train "--modell", "resnet50.a1_in1k", "--methode", "lp"
Invoke-Train "--modell", "resnet50.tv_in1k", "--methode", "lp"
Invoke-Train "--modell", "resnet50.a1_in1k", "--methode", "full", "--lr", "3e-4"
Invoke-Train "--modell", "resnet50.tv_in1k", "--methode", "full", "--lr", "3e-4"

Invoke-Train "--methode", "full", "--lr", "1e-4", "--ohne_vortraining"
Invoke-Train "--modell", "resnet50.a1_in1k", "--methode", "full", "--lr", "1e-3", "--ohne_vortraining"

python zusammenfassung.py --daten $DATEN
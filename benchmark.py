# Kostenmessung Station 1 und Studenten aus Station 2: alle Konfigurationen nacheinander auf DEMSELBEN Rechner unter gleichen Bedingungen.
#   python benchmark.py                       # alle Konfigurationen 3x, dann Median je Konfiguration
#   python benchmark.py --nur vit_full        # nur eine Konfiguration, einmal, ohne Median
# Ergebnis: ausgaben/resisc45/kostenmessung.csv und kostenmessung_median.csv. Der Median fasst alle Zeilen der CSV
# zusammen: für eine neue Messreihe die alte Datei löschen oder --ziel ändern.
#
# Gemessen wird pro Konfiguration in einem eigenen Prozess (damit der Speicher-Peak nicht vom Vorgänger stammt):
#   - Trainingsschritt (vorwärts + rückwärts + AdamW) mit Batch 32, Zufallsbildern 224x224 und Zufallslabels:
#     Median-Zeit pro Schritt nach Aufwärmen -> hochgerechnet auf eine Epoche (689 Schritte bei 22.044 Bildern).
#     Ohne Bilderladen und ohne Validierung, also reine Rechenzeit des Modells.
#   - Peak-Speicher während der Trainingsschritte (MPS: torch.mps.driver_allocated_memory, CUDA: max_memory_allocated).
#   - Inferenz: Latenz für 1 Bild (Median, wie latenz_ms in train.py) und Durchsatz bei Batch 32.
#   - LoRA zusätzlich mit zusammengeführten Gewichten (W + s * B A), also ohne Zusatzrechnung zur Inferenzzeit.
# Die Gewichtswerte spielen für Zeit und Speicher keine Rolle; geladen werden trotzdem die vortrainierten Gewichte.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import gc
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import timm
import torch
from torch import nn

from lora import lora_zusammenfuehren
from modelle import baue_modell
from werkzeuge import csv_anhaengen, latenz_ms, waehle_geraet

VIT = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
KONFIGS = {   # name: (modell, methode, rang, lora_ziele)
    "vit_lp": (VIT, "lp", 0, ""),
    "vit_lora_r8_qkv": (VIT, "lora", 8, "qkv"),
    "vit_lora_r8_alle": (VIT, "lora", 8, "qkv,proj,fc1,fc2"),
    "vit_full": (VIT, "full", 0, ""),
    "resnet_a1_lp": ("resnet50.a1_in1k", "lp", 0, ""),
    "resnet_a1_full": ("resnet50.a1_in1k", "full", 0, ""),
    "student_small": ("vit_small_patch16_224.augreg_in21k_ft_in1k", "full", 0, ""),
    "student_tiny": ("vit_tiny_patch16_224.augreg_in21k_ft_in1k", "full", 0, ""),
}

ZAHLEN = ["train_ms_pro_schritt", "train_s_pro_epoche_rechnung", "peak_speicher_train_mb", "latenz_b1_ms",
          "latenz_b1_ms_gemergt", "durchsatz_b32_bilder_s", "durchsatz_b32_gemergt"]


def warten(geraet):
    if geraet.type == "mps":
        torch.mps.synchronize()
    elif geraet.type == "cuda":
        torch.cuda.synchronize()


# Auf der CPU gibt es hier keinen sinnvollen Wert, deshalb nan
def speicher_mb(geraet):
    if geraet.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    if geraet.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return float("nan")


def durchsatz(modell, geraet, batch, wdh=20):
    x = torch.randn(batch, 3, 224, 224, device=geraet)
    zeiten = []
    with torch.no_grad():
        for i in range(wdh + 5):
            warten(geraet)
            t0 = time.perf_counter()
            modell(x)
            warten(geraet)
            if i >= 5:
                zeiten.append(time.perf_counter() - t0)
    return batch / sorted(zeiten)[len(zeiten) // 2]


def messen(name, args):
    modell_name, methode, rang, ziele = KONFIGS[name]
    geraet = waehle_geraet()
    torch.manual_seed(0)
    ziele_t = tuple(ziele.split(",")) if ziele else ("qkv",)
    modell = baue_modell(modell_name, 45, methode, rang or 8, ziele_t, vortrainiert=not args.ohne_download).to(geraet)
    trainierbar = sum(p.numel() for p in modell.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in modell.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    verlust_fn = nn.CrossEntropyLoss()
    x = torch.randn(args.batch, 3, 224, 224, device=geraet)
    # Zufallsdaten reichen, es geht nur um Zeit und Speicher
    y = torch.randint(0, 45, (args.batch,), device=geraet)
    if geraet.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------- Training
    modell.train(methode != "lp")  # wie in train.py
    zeiten, peak = [], 0.0
    for i in range(args.aufwaermen + args.schritte):
        warten(geraet)
        t0 = time.perf_counter()
        verlust = verlust_fn(modell(x), y)
        optimizer.zero_grad()
        verlust.backward()
        optimizer.step()
        warten(geraet)
        # Erste Schritte sind langsamer und zählen nicht mit
        if i >= args.aufwaermen:
            zeiten.append(time.perf_counter() - t0)
        peak = max(peak, speicher_mb(geraet))
    s_schritt = sorted(zeiten)[len(zeiten) // 2]

    # ---------------- Inferenz (Optimizer und Gradienten freigeben, damit nur das Modell im Speicher ist)
    del optimizer, verlust
    modell.zero_grad(set_to_none=True)
    gc.collect()
    if geraet.type == "mps":
        torch.mps.empty_cache()
    modell.eval()
    lat1 = latenz_ms(modell, geraet, 224, args.latenz_wdh)
    dz = durchsatz(modell, geraet, args.batch)
    lat1_merge = dz_merge = ""
    if methode == "lora":
        lora_zusammenfuehren(modell)
        lat1_merge = round(latenz_ms(modell, geraet, 224, args.latenz_wdh), 2)
        dz_merge = round(durchsatz(modell, geraet, args.batch), 1)

    return dict(
        konfig=name, modell=modell_name, methode=methode, rang=rang or "", lora_ziele=ziele, trainierbar=trainierbar,
        geraet=geraet.type, batch=args.batch, train_ms_pro_schritt=round(1000 * s_schritt, 1),
        train_s_pro_epoche_rechnung=round(s_schritt * math.ceil(22044 / args.batch)),
        peak_speicher_train_mb=round(peak), latenz_b1_ms=round(lat1, 2), latenz_b1_ms_gemergt=lat1_merge,
        durchsatz_b32_bilder_s=round(dz, 1), durchsatz_b32_gemergt=dz_merge,
        torch=torch.__version__, timm=timm.__version__, python=platform.python_version(),
        datum=time.strftime("%Y-%m-%d %H:%M"))


def median_schreiben(pfad):
    """Median, Minimum und Maximum je Konfiguration über alle Zeilen der Messdatei."""
    zeilen = defaultdict(list)
    with open(pfad) as f:
        for z in csv.DictReader(f):
            zeilen[z["konfig"]].append(z)
    # Eigene Datei, damit die Rohwerte erhalten bleiben
    ziel = Path(pfad).with_name(Path(pfad).stem + "_median.csv")
    with open(ziel, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["konfig", "wiederholungen"] + [f"{s}_{art}" for art in ("median", "min", "max") for s in ZAHLEN])
        for k, zs in zeilen.items():
            werte = [[float(z[s]) for z in zs if z[s] != ""] for s in ZAHLEN]
            med = [round(statistics.median(v), 2) if v else "" for v in werte]
            w.writerow([k, len(zs)] + med + [min(v) if v else "" for v in werte] + [max(v) if v else "" for v in werte])
            print(k, len(zs), med)
    print(f"\nFertig: {pfad} und {ziel}")


def rechner_info():
    info = {"plattform": platform.platform()}
    if sys.platform == "darwin":
        for schluessel, befehl in [("chip", "machdep.cpu.brand_string"), ("ram_bytes", "hw.memsize"),
                                   ("cpu_kerne", "machdep.cpu.core_count")]:
            try:
                info[schluessel] = subprocess.run(["sysctl", "-n", befehl], capture_output=True, text=True).stdout.strip()
            except Exception:
                pass
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nur", choices=list(KONFIGS), help="nur diese Konfiguration (sonst alle, je eigener Prozess)")
    p.add_argument("--wiederholungen", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--aufwaermen", type=int, default=5)
    p.add_argument("--schritte", type=int, default=30)
    p.add_argument("--latenz_wdh", type=int, default=50)
    p.add_argument("--ziel", default="ausgaben/resisc45/kostenmessung.csv")
    p.add_argument("--ohne_download", action="store_true", help="nur zum Testen: zufällige Gewichte")
    args = p.parse_args()

    if args.nur:                                  # Kindprozess: eine Konfiguration messen, Zeile anhängen
        zeile = messen(args.nur, args)
        print(zeile, flush=True)
        os.makedirs(os.path.dirname(args.ziel) or ".", exist_ok=True)
        csv_anhaengen(args.ziel, zeile)
        return

    print("Rechner:", rechner_info())
    print("Bitte während der Messung keine anderen rechenintensiven Programme laufen lassen und das Netzteil anschließen.")
    weiter = [a for a in sys.argv[1:]]
    for i in range(args.wiederholungen):
        for name in KONFIGS:
            print(f"\n=== {name} ({i + 1}/{args.wiederholungen})", flush=True)
            # Jedes Modell in einem neuen Prozess, sonst bleibt Speicher vom vorigen belegt
            subprocess.run([sys.executable, __file__, "--nur", name] + weiter, check=True)
    median_schreiben(args.ziel)


if __name__ == "__main__":
    main()

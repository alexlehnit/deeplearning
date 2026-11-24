import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # fehlende MPS-Operationen laufen auf der CPU

import argparse
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from daten import lade_daten, transformationen
from lora import lora_zusammenfuehren
from modelle import baue_modell, speichere_checkpoint
from werkzeuge import (anzahl_klassen, ausgabe_ordner, csv_anhaengen, konfusionsmatrix, latenz_ms,
                       setze_seed, waehle_geraet)


# Argumente für Terminal Aufruf des Python Scripts
def argumente():
    p = argparse.ArgumentParser(description="Station 1: Adaptation auf RESISC45")
    p.add_argument("--daten", default="daten/resisc45")
    p.add_argument("--modell", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    p.add_argument("--methode", choices=["lp", "lora", "full"], default="lp")
    p.add_argument("--epochen", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rang", type=int, default=8, help="LoRA-Rang r (alpha = 2r)")
    p.add_argument("--lora_ziele", default="qkv", help="z. B. qkv oder qkv,proj,fc1,fc2")
    p.add_argument("--ohne_vortraining", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worker", type=int, default=4)
    return p.parse_args()



# Alle wichtigen Einstellungen in den Namen, sonst überschreiben sich Läufe
def lauf_name(args):
    ziele = args.lora_ziele.split(",")
    name = f"{args.modell.replace('.', '-')}_{args.methode}"
    if args.methode == "lora":
        name += f"_r{args.rang}" + ("" if ziele == ["qkv"] else "_" + "-".join(ziele))
    if args.ohne_vortraining:
        name += "_scratch"
    return name + f"_lr{args.lr:g}_ep{args.epochen}_s{args.seed}"

# Für Val und Test: Labels, Vorhersagen und mittlerer Verlust
def auswerten(modell, loader, verlust_fn, geraet):
    modell.eval()
    alle_y, alle_p, summe = [], [], 0.0
    with torch.no_grad():
        for bilder, y in loader:
            logits = modell(bilder.to(geraet))
            summe += verlust_fn(logits, y.to(geraet)).item() * len(y)
            alle_y += y.tolist()
            alle_p += logits.argmax(1).cpu().tolist()
    return alle_y, alle_p, summe / len(alle_y)

def trainiere_epoche(modell, loader, verlust_fn, optimizer, scheduler, geraet, nur_kopf):
    modell.train(not nur_kopf)  # bei lp auch die BatchNorm-Statistiken einfrieren
    summe, richtig, anzahl = 0.0, 0, 0
    for bilder, y in loader:
        bilder, y = bilder.to(geraet), y.to(geraet)
        logits = modell(bilder)
        verlust = verlust_fn(logits, y)
        optimizer.zero_grad()
        # Gradienten berechnen und Gewichte anpassen
        verlust.backward()
        optimizer.step()
        # Lernrate sinkt nach jedem Batch, nicht erst nach der Epoche
        scheduler.step()
        summe += verlust.item() * len(y)
        richtig += (logits.argmax(1) == y).sum().item()
        anzahl += len(y)
    return summe / anzahl, richtig / anzahl


def main():
    args = argumente()
    setze_seed(args.seed)
    geraet = waehle_geraet()
    ordner = ausgabe_ordner(args.daten)
    ziele = tuple(args.lora_ziele.split(","))
    vortrainiert = not args.ohne_vortraining
    lauf = lauf_name(args)

    modell = baue_modell(args.modell, anzahl_klassen(args.daten), args.methode, args.rang, ziele, vortrainiert).to(geraet)
    gesamt = sum(p.numel() for p in modell.parameters())
    trainierbar = sum(p.numel() for p in modell.parameters() if p.requires_grad)

    # Jedes Modell erwartet eine eigene Normierung, die steht in der timm-Konfig
    cfg = modell.pretrained_cfg
    groesse = cfg["input_size"][-1]
    train_tf, eval_tf = transformationen(cfg["mean"], cfg["std"], groesse)
    train_ds, val_ds, test_ds, klassen = lade_daten(args.daten, train_tf, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.worker)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=args.worker)
    print(f"Lauf: {lauf} | Gerät: {geraet}")
    print(f"Parameter: {gesamt:,} gesamt, {trainierbar:,} trainierbar")
    print(f"Bilder: {len(train_ds):,} Training, {len(val_ds):,} Validierung, {len(test_ds):,} Test")

    verlust_fn = nn.CrossEntropyLoss()
    # Nur die Parameter übergeben, die wirklich trainiert werden
    optimizer = torch.optim.AdamW([p for p in modell.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochen * len(train_loader))

    writer = SummaryWriter(str(ordner / "runs" / f"{lauf}_{time.strftime('%m%d-%H%M%S')}"))
    checkpoint = ordner / "modelle" / f"{lauf}.pt"
    konfig = dict(modell=args.modell, methode=args.methode, rang=args.rang, lora_ziele=list(ziele),
                  vortrainiert=vortrainiert, klassen=klassen)

    beste_val_acc, beste_epoche, start = -1.0, 0, time.time()
    for epoche in range(1, args.epochen + 1):
        train_verlust, train_acc = trainiere_epoche(modell, train_loader, verlust_fn, optimizer, scheduler, geraet,
                                                    args.methode == "lp")
        val_y, val_p, val_verlust = auswerten(modell, val_loader, verlust_fn, geraet)
        val_acc = accuracy_score(val_y, val_p)
        print(f"Epoche {epoche}: Verlust {train_verlust:.4f} | Train-Acc {train_acc:.3f} "
              f"| Val-Verlust {val_verlust:.4f} | Val-Acc {val_acc:.3f}", flush=True)
        writer.add_scalar("Verlust/train", train_verlust, epoche)
        writer.add_scalar("Verlust/val", val_verlust, epoche)
        writer.add_scalar("Accuracy/train", train_acc, epoche)
        writer.add_scalar("Accuracy/val", val_acc, epoche)
        writer.add_scalar("Lernrate", scheduler.get_last_lr()[0], epoche)
        # Nur das beste Modell nach Val-Accuracy behalten
        if val_acc > beste_val_acc:
            beste_val_acc, beste_epoche = val_acc, epoche
            speichere_checkpoint(modell, checkpoint, konfig)
    zeit = time.time() - start

    # strict=False: bei lp/lora ist nur ein Teil der Gewichte gespeichert
    modell.load_state_dict(torch.load(checkpoint, map_location="cpu")["zustand"], strict=False)
    test_y, test_p, _ = auswerten(modell, test_loader, verlust_fn, geraet)
    test_acc, test_f1 = accuracy_score(test_y, test_p), f1_score(test_y, test_p, average="macro")
    # Vor der Latenzmessung zusammenführen, so würde man es auch einsetzen
    modell = lora_zusammenfuehren(modell)
    gesamt = sum(p.numel() for p in modell.parameters())
    latenz = latenz_ms(modell.eval(), geraet, groesse)
    print(f"Test-Accuracy {test_acc:.4f} | Macro-F1 {test_f1:.4f} | Training {zeit:.0f} s | Latenz {latenz:.1f} ms/Bild")

    fig = konfusionsmatrix(test_y, test_p, klassen, ordner / "konfusion" / f"{lauf}.png", lauf)
    writer.add_figure("Konfusionsmatrix (Test)", fig)
    writer.close()
    csv_anhaengen(ordner / "ergebnisse.csv", dict(
        modell=args.modell, methode=args.methode, rang=args.rang if args.methode == "lora" else "",
        lora_ziele=args.lora_ziele if args.methode == "lora" else "", vortrainiert="ja" if vortrainiert else "nein",
        epochen=args.epochen, beste_epoche=beste_epoche, lr=args.lr, batch=args.batch, seed=args.seed,
        parameter=gesamt, trainierbar=trainierbar, val_acc=round(beste_val_acc, 4), test_acc=round(test_acc, 4),
        test_f1=round(test_f1, 4), zeit_s=round(zeit), latenz_ms=round(latenz, 2), checkpoint=str(checkpoint)))
    print(f"Gespeichert: {ordner}/ergebnisse.csv, {checkpoint}")


if __name__ == "__main__":
    main()

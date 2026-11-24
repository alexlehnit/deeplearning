import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from daten import lade_daten, transformationen
from lora import lora_zusammenfuehren
from modelle import baue_modell, lade_checkpoint, speichere_checkpoint
from werkzeuge import (ausgabe_ordner, bester_checkpoint, csv_anhaengen, konfusionsmatrix, latenz_ms, setze_seed,
                       vorhersagen, waehle_geraet)


def argumente():
    p = argparse.ArgumentParser(description="Station 2: Knowledge Distillation")
    p.add_argument("--daten", default="daten/resisc45")
    p.add_argument("--teacher", default=None)
    p.add_argument("--student", default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--temperatur", type=float, default=4.0)
    p.add_argument("--epochen", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worker", type=int, default=4)
    return p.parse_args()



# Mischung aus normalem Verlust und Abstand zum Teacher.
# alpha sagt, wie stark der Teacher zählt, bei 0 ist es normales Training.
def kd_verlust(logits_s, logits_t, y, alpha, T):
    ce = F.cross_entropy(logits_s, y)
    if alpha == 0:
        return ce
    # Hohe Temperatur macht die Verteilung weicher, so sieht der Student auch,
    # welche Klassen der Teacher ähnlich findet. T * T gleicht die kleineren Gradienten aus.
    kd = F.kl_div(F.log_softmax(logits_s / T, dim=1), F.softmax(logits_t / T, dim=1), reduction="batchmean") * T * T
    return alpha * kd + (1 - alpha) * ce

# Teacher und Student können anders normiert sein, hier wird umgerechnet
def normierung_umrechnen(cfg_s, cfg_t, geraet):
    form = (1, 3, 1, 1)
    mean_s, std_s = torch.tensor(cfg_s["mean"]).view(form).to(geraet), torch.tensor(cfg_s["std"]).view(form).to(geraet)
    mean_t, std_t = torch.tensor(cfg_t["mean"]).view(form).to(geraet), torch.tensor(cfg_t["std"]).view(form).to(geraet)
    return lambda bilder: (bilder * std_s + mean_s - mean_t) / std_t

def main():
    args = argumente()
    setze_seed(args.seed)
    geraet = waehle_geraet()
    ordner = ausgabe_ordner(args.daten)

    # 1. Teacher
    # Ohne Angabe nehmen wir den besten LoRA-Lauf aus Station 1
    teacher_pfad = args.teacher or bester_checkpoint(ordner, "lora")
    teacher, t_konfig = lade_checkpoint(teacher_pfad, geraet)
    lora_zusammenfuehren(teacher)
    for p in teacher.parameters():
        # Der Teacher lernt nicht mehr mit
        p.requires_grad = False

    # 2. Student
    student = baue_modell(args.student, len(t_konfig["klassen"]), methode="full").to(geraet)
    p_student = sum(p.numel() for p in student.parameters())
    p_teacher = sum(p.numel() for p in teacher.parameters())

    # 3. Daten
    cfg_s, cfg_t = student.pretrained_cfg, teacher.pretrained_cfg
    train_tf, eval_tf = transformationen(cfg_s["mean"], cfg_s["std"], cfg_s["input_size"][-1])
    train_ds, val_ds, test_ds, klassen = lade_daten(args.daten, train_tf, eval_tf)
    # Sonst würden die Ausgaben von Teacher und Student nicht zusammenpassen
    assert klassen == t_konfig["klassen"], "Teacher wurde auf anderen Klassen trainiert"
    fuer_teacher = normierung_umrechnen(cfg_s, cfg_t, geraet)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.worker)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=args.worker)

    # 4. Teacher als Referenz
    t_y, t_p = vorhersagen(teacher, test_loader, geraet, fuer_teacher)
    teacher_acc = accuracy_score(t_y, t_p)

    # 5. Optimierer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochen * len(train_loader))
    lauf = (f"student_{args.student.replace('.', '-')}_"
            + ("ohne_kd" if args.alpha == 0 else f"kd_a{args.alpha:g}_T{args.temperatur:g}")
            + f"_lr{args.lr:g}_ep{args.epochen}_s{args.seed}")
    checkpoint = ordner / "modelle" / f"{lauf}.pt"
    konfig = dict(modell=args.student, methode="full", rang=0, lora_ziele=[], vortrainiert=True, klassen=klassen)
    writer = SummaryWriter(str(ordner / "runs" / f"{lauf}_{time.strftime('%m%d-%H%M%S')}"))

    # 6. Training
    beste_val_acc, start = -1.0, time.time()
    for epoche in range(1, args.epochen + 1):
        student.train()
        summe = 0.0
        for bilder, y in train_loader:
            bilder, y = bilder.to(geraet), y.to(geraet)
            with torch.no_grad():
                # Teacher sieht genau dieselben Bilder, ohne KD wird er nicht gebraucht
                logits_t = teacher(fuer_teacher(bilder)) if args.alpha > 0 else None
            logits_s = student(bilder)
            verlust = kd_verlust(logits_s, logits_t, y, args.alpha, args.temperatur)
            optimizer.zero_grad()
            verlust.backward()
            optimizer.step()
            scheduler.step()
            summe += verlust.item() * len(y)
        val_y, val_p = vorhersagen(student, val_loader, geraet)
        val_acc = accuracy_score(val_y, val_p)
        print(f"Epoche {epoche}: Verlust {summe / len(train_ds):.4f} | Val-Acc {val_acc:.3f}", flush=True)
        writer.add_scalar("Verlust/train", summe / len(train_ds), epoche)
        writer.add_scalar("Accuracy/val", val_acc, epoche)
        if val_acc > beste_val_acc:
            beste_val_acc = val_acc
            speichere_checkpoint(student, checkpoint, konfig)
    zeit = time.time() - start

    # 7. Test und CSV
    # Für den Test wieder das beste Modell laden, nicht das letzte
    student.load_state_dict(torch.load(checkpoint, map_location="cpu")["zustand"])
    test_y, test_p = vorhersagen(student, test_loader, geraet)
    test_acc, test_f1 = accuracy_score(test_y, test_p), f1_score(test_y, test_p, average="macro")
    latenz_s = latenz_ms(student.eval(), geraet, cfg_s["input_size"][-1])
    latenz_t = latenz_ms(teacher, geraet, cfg_t["input_size"][-1])
    print(f"Student {test_acc:.4f} (Teacher {teacher_acc:.4f}) | Latenz {latenz_s:.1f} vs. {latenz_t:.1f} ms")
    fig = konfusionsmatrix(test_y, test_p, klassen, ordner / "konfusion" / f"{lauf}.png", lauf)
    writer.add_figure("Konfusionsmatrix (Test)", fig)
    writer.close()
    csv_anhaengen(ordner / "ergebnisse_kd.csv", dict(
        student=args.student, kd="nein" if args.alpha == 0 else "ja", alpha=args.alpha, temperatur=args.temperatur,
        epochen=args.epochen, lr=args.lr, seed=args.seed, parameter=p_student, groesse_mb=round(p_student * 4 / 1e6, 1),
        val_acc=round(beste_val_acc, 4), test_acc=round(test_acc, 4), test_f1=round(test_f1, 4),
        latenz_ms=round(latenz_s, 2), zeit_s=round(zeit), teacher=teacher_pfad, teacher_test_acc=round(teacher_acc, 4),
        teacher_parameter=p_teacher, teacher_latenz_ms=round(latenz_t, 2), checkpoint=str(checkpoint)))


if __name__ == "__main__":
    main()

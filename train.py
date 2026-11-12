#Import von Benötigtien Bibilotheken
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import numpy as np
import argparse


# Argumente für Terminal Aufruf des Python Scripts
def argumente():
    p = argparse.ArgumentParser(description="ViT/ResNet auf eigenem Datensatz trainieren")
    p.add_argument("--daten", default="daten/mini")
    p.add_argument("--modell", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochen", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


#GPU Support / GPU Setzen
def waehle_geraet():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Mac mit Apple-Chip
        return torch.device("mps")
    return torch.device("cpu")


def setze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 1. Daten laden
def lade_daten(pfad, batch, seed):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # Werte dieses ViT
    ])
    ds = datasets.ImageFolder(pfad, transform=tf)                 # ein Ordner pro Klasse
    n_val = n_test = int(len(ds) * 0.15)                           # 70/15/15
    train_ds, val_ds, test_ds = random_split(
        ds, [len(ds) - n_val - n_test, n_val, n_test],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)
    test_loader = DataLoader(test_ds, batch_size=batch)
    return train_loader, val_loader, test_loader, len(ds.classes)


# 2. Modell: vortrainiert, neuer Kopf, nur der Kopf lernt (Linear Probing)
def baue_modell(name, anzahl_klassen, geraet):
    modell = timm.create_model(name, pretrained=True, num_classes=anzahl_klassen).to(geraet)
    for p in modell.parameters():
        p.requires_grad = False
    for p in modell.get_classifier().parameters():
        p.requires_grad = True
    return modell


# 3. + 4. Loss, Optimizer und Training
def trainiere(modell, train_loader, val_loader, geraet, epochen, lr):
    verlust_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in modell.parameters() if p.requires_grad], lr=lr)
    for epoche in range(epochen):
        modell.train()
        for bilder, y in train_loader:
            bilder, y = bilder.to(geraet), y.to(geraet)
            logits = modell(bilder)            # vorwärts
            verlust = verlust_fn(logits, y)    # Fehler
            optimizer.zero_grad()              # alte Gradienten löschen
            verlust.backward()                 # Backpropagation
            optimizer.step()                   # Gewichte anpassen
        val_acc = auswerten(modell, val_loader, geraet)
        print(f"Epoche {epoche + 1}: Loss {verlust.item():.4f}, Val-Acc {val_acc:.3f}")


# Auswertung (Training, Validierung, Test trennen)
def auswerten(modell, loader, geraet):
    modell.eval()
    richtig, anzahl = 0, 0
    with torch.no_grad():
        for bilder, y in loader:
            bilder, y = bilder.to(geraet), y.to(geraet)
            richtig += (modell(bilder).argmax(1) == y).sum().item()
            anzahl += len(y)
    return richtig / anzahl


def main():
    args = argumente()
    geraet = waehle_geraet()
    setze_seed(args.seed)
    print("Gerät:", geraet)
    train_loader, val_loader, test_loader, anzahl_klassen = lade_daten(args.daten, args.batch, args.seed)
    modell = baue_modell(args.modell, anzahl_klassen, geraet)
    trainiere(modell, train_loader, val_loader, geraet, args.epochen, args.lr)
    print(f"Test-Accuracy: {auswerten(modell, test_loader, geraet):.3f}")


if __name__ == "__main__":
    main()

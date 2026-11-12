#Import von Benötigtien Bibilotheken
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import numpy as np

# Setzen von festen varibabeln
DATEN = "daten/mini"
MODELL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
EPOCHEN = 3
LR = 1e-3

#GPU Support
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


geraet = waehle_geraet()
setze_seed(42)
print("Gerät:", geraet)


# 1. Daten laden 
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # Werte dieses ViT
])
ds = datasets.ImageFolder(DATEN, transform=tf)                # ein Ordner pro Klasse
n_test = len(ds) // 5                                          # 20 % zum Testen
train_ds, test_ds = random_split(ds, [len(ds) - n_test, n_test],
                                 generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# 2. Modell: vortrainiert, neuer Kopf, nur der Kopf lernt (Linear Probing)
modell = timm.create_model(MODELL, pretrained=True, num_classes=len(ds.classes)).to(geraet)
for p in modell.parameters():
    p.requires_grad = False
for p in modell.get_classifier().parameters():
    p.requires_grad = True

# 3. Loss und Optimizer 
verlust_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in modell.parameters() if p.requires_grad], lr=LR)

# 4. Training 
for epoche in range(EPOCHEN):
    modell.train()
    for bilder, y in train_loader:
        bilder, y = bilder.to(geraet), y.to(geraet)   # neu
        logits = modell(bilder)            # vorwärts
        verlust = verlust_fn(logits, y)    # Fehler
        optimizer.zero_grad()              # alte Gradienten löschen
        verlust.backward()                 # Backpropagation
        optimizer.step()                   # Gewichte anpassen
    print(f"Epoche {epoche + 1}: letzter Loss {verlust.item():.4f}")

# 5. Test 
modell.eval()
richtig = 0
with torch.no_grad():
    for bilder, y in test_loader:
        bilder, y = bilder.to(geraet), y.to(geraet)
        richtig += (modell(bilder).argmax(1) == y).sum().item()
print(f"Test-Accuracy: {richtig / len(test_ds):.3f}")
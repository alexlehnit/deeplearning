from torchvision import datasets, transforms
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def transformationen(mean, std, groesse=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(groesse, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(groesse * 256 / 224)),
        transforms.CenterCrop(groesse),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf

def lade_daten(ordner, train_tf, eval_tf, anteil_val=0.15, anteil_test=0.15):
    mit_aug = datasets.ImageFolder(ordner, transform=train_tf)
    ohne_aug = datasets.ImageFolder(ordner, transform=eval_tf)
    labels = mit_aug.targets
    alle = list(range(len(labels)))
    rest, test_idx = train_test_split(alle, test_size=anteil_test, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(rest, test_size=anteil_val / (1 - anteil_test),
                                          stratify=[labels[i] for i in rest], random_state=42)
    return Subset(mit_aug, train_idx), Subset(ohne_aug, val_idx), Subset(ohne_aug, test_idx), mit_aug.classes


def lade_meta(ordner):
    pfad = os.path.join(ordner, "meta.json")
    if not os.path.exists(pfad):
        return {}
    with open(pfad) as f:
        return json.load(f)


def lesbarer_name(ordnername):
    return ordnername.replace("_", " ")


def anzahl_bilder_pro_klasse(ds):
    labels = np.array(ds.dataset.targets)[ds.indices]
    return np.bincount(labels, minlength=len(ds.dataset.classes))
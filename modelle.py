import timm
import torch
from lora import lora_einbauen

def baue_modell(name, anzahl_klassen, methode="lp", rang=8, ziele=("qkv",), vortrainiert=True):
    modell = timm.create_model(name, pretrained=vortrainiert, num_classes=anzahl_klassen)
    if methode in ("lp", "lora"):
        for p in modell.parameters():
            p.requires_grad = False
        if methode == "lora":
            lora_einbauen(modell, r=rang, alpha=2 * rang, ziele=ziele)
        for p in modell.get_classifier().parameters():
            p.requires_grad = True
    return modell

def speichere_checkpoint(modell, pfad, konfig):
    if konfig["methode"] in ("lp", "lora") and konfig["vortrainiert"]:
        zustand = {n: p.detach().cpu() for n, p in modell.named_parameters() if p.requires_grad}
        zustand.update({n: b.detach().cpu() for n, b in modell.named_buffers()})
    else:
        zustand = {k: v.detach().cpu() for k, v in modell.state_dict().items()}
    torch.save({"konfig": konfig, "zustand": zustand}, pfad)

def lade_checkpoint(pfad, geraet="cpu"):
    daten = torch.load(pfad, map_location="cpu")
    k = daten["konfig"]
    nur_teilweise = k["methode"] in ("lp", "lora") and k["vortrainiert"]
    modell = baue_modell(k["modell"], len(k["klassen"]), k["methode"], k["rang"], tuple(k["lora_ziele"]),
                         vortrainiert=nur_teilweise)
    modell.load_state_dict(daten["zustand"], strict=not nur_teilweise)
    return modell.to(geraet).eval(), k
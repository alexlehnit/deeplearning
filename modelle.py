import timm
import torch
from lora import lora_einbauen

def baue_modell(name, anzahl_klassen, methode="lp", rang=8, ziele=("qkv",), vortrainiert=True):
    # timm setzt direkt einen neuen Kopf mit unserer Klassenzahl drauf
    modell = timm.create_model(name, pretrained=vortrainiert, num_classes=anzahl_klassen)
    if methode in ("lp", "lora"):
        # Erstmal alles einfrieren
        for p in modell.parameters():
            p.requires_grad = False
        if methode == "lora":
            # alpha = 2 * Rang, dadurch ist die Skalierung bei jedem Rang 2
            lora_einbauen(modell, r=rang, alpha=2 * rang, ziele=ziele)
        # Der Kopf ist neu und muss immer trainiert werden
        for p in modell.get_classifier().parameters():
            p.requires_grad = True
    return modell

def speichere_checkpoint(modell, pfad, konfig):
    # Bei lp und lora reicht das Trainierte, der Rest kommt wieder von timm
    if konfig["methode"] in ("lp", "lora") and konfig["vortrainiert"]:
        zustand = {n: p.detach().cpu() for n, p in modell.named_parameters() if p.requires_grad}
        # Buffer mit speichern, z. B. die BatchNorm-Werte vom ResNet
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
    # Fehlt ein Teil absichtlich, darf load_state_dict nicht meckern
    modell.load_state_dict(daten["zustand"], strict=not nur_teilweise)
    return modell.to(geraet).eval(), k
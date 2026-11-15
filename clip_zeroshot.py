import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import open_clip
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from daten import lade_daten, lade_meta, lesbarer_name
from werkzeuge import ausgabe_ordner, csv_anhaengen, waehle_geraet, konfusionsmatrix, latenz_ms


def main():
    parser = argparse.ArgumentParser(description="Station 3: CLIP Zero-Shot")
    parser.add_argument("--daten", default="daten/resisc45")
    parser.add_argument("--modell", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--worker", type=int, default=4)
    args = parser.parse_args()

    geraet = waehle_geraet()
    ordner = ausgabe_ordner(args.daten)

    modell, _, vorverarbeitung = open_clip.create_model_and_transforms(args.modell, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.modell)
    modell = modell.to(geraet).eval()
    parameter_bild = sum(p.numel() for p in modell.visual.parameters())

    _, _, test_ds, klassen = lade_daten(args.daten, vorverarbeitung, vorverarbeitung)
    print(f"{len(test_ds):,} Testbilder, {len(klassen)} Klassen")
    meta = lade_meta(args.daten)
    namen_varianten = {"Ordnername": [lesbarer_name(k) for k in klassen]}
    if meta.get("klassennamen"):
        namen_varianten["beschreibend"] = [meta["klassennamen"].get(k, lesbarer_name(k)) for k in klassen]
    domaene = meta.get("prompts", ["a photo of {}."])
    prompts = {
        "P0 nur Name": ["{}"],
        "P1 a photo of": ["a photo of a {}."],
        "P2 Domäne": domaene[:1],
        "P3 Domäne-Ensemble": domaene,
    }

    bild_merkmale, labels = [], []
    with torch.no_grad():
        for bilder, y in DataLoader(test_ds, batch_size=args.batch, num_workers=args.worker):
            m = modell.encode_image(bilder.to(geraet))
            bild_merkmale.append(m / m.norm(dim=-1, keepdim=True))
            labels += y.tolist()
    bild_merkmale = torch.cat(bild_merkmale)

    def text_merkmale(namen, vorlagen):
        ergebnis = []
        with torch.no_grad():
            for name in namen:
                t = modell.encode_text(tokenizer([v.format(name) for v in vorlagen]).to(geraet))
                t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
                ergebnis.append(t / t.norm())
        return torch.stack(ergebnis)

    groesse = modell.visual.image_size
    groesse = groesse[0] if isinstance(groesse, (tuple, list)) else groesse
    latenz = latenz_ms(modell.encode_image, geraet, groesse=groesse)
    ergebnisse = []
    for namen_art, namen in namen_varianten.items():
        for prompt_name, vorlagen in prompts.items():
            vorhersage = (bild_merkmale @ text_merkmale(namen, vorlagen).T).argmax(1).cpu().tolist()
            acc, f1 = accuracy_score(labels, vorhersage), f1_score(labels, vorhersage, average="macro")
            ergebnisse.append((namen_art, prompt_name, acc, f1, vorhersage))
            print(f"{namen_art:12s} | {prompt_name:20s} | Accuracy {acc:.4f} | Macro-F1 {f1:.4f}")
            csv_anhaengen(ordner / "ergebnisse_clip.csv", dict(
                modell=args.modell, pretrained=args.pretrained, namen=namen_art, prompt=prompt_name,
                test_acc=round(acc, 4), test_f1=round(f1, 4), parameter_bild=parameter_bild,
                latenz_ms=round(latenz, 2)))

    bester = max(ergebnisse, key=lambda e: e[2])
    titel = f"CLIP {args.modell}: {bester[0]}, {bester[1]}"
    konfusionsmatrix(labels, bester[4], klassen, ordner / "konfusion" / f"clip_{args.modell}.png", titel)
    print(f"Beste Variante: {bester[0]}, {bester[1]}, Accuracy {bester[2]:.4f} | Latenz {latenz:.1f} ms")


if __name__ == "__main__":
    main()

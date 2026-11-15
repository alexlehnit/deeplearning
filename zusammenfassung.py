import argparse
import csv
import statistics
from pathlib import Path


def lies(pfad):
    if not pfad.exists():
        return []
    with open(pfad) as f:
        return list(csv.DictReader(f))


def gruppieren(zeilen, schluessel):
    gruppen = {}
    for z in zeilen:
        gruppen.setdefault(tuple(z[k] for k in schluessel), []).append(z)
    return gruppen


def mittel(zeilen, spalte):
    werte = [float(z[spalte]) for z in zeilen]
    return statistics.mean(werte), (statistics.stdev(werte) if len(werte) > 1 else 0.0)


def prozent(m, s=None):
    return f"{100 * m:.1f} %" + (f" ± {100 * s:.1f}" if s else "")


def tabelle(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
                     + ["| " + " | ".join(str(x) for x in z) + " |" for z in zeilen])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daten", default="daten/resisc45")
    args = parser.parse_args()
    ordner = Path("ausgaben") / Path(args.daten).name
    s1, s2, s3 = lies(ordner / "ergebnisse.csv"), lies(ordner / "ergebnisse_kd.csv"), lies(ordner / "ergebnisse_clip.csv")
    teile = [f"# Ergebnisse: {Path(args.daten).name}", ""]
    vergleich = []

    # Station 1
    if s1:
        schluessel = ["modell", "methode", "rang", "lora_ziele", "vortrainiert", "lr", "epochen"]
        zeilen = []
        for k, g in gruppieren(s1, schluessel).items():
            acc, acc_s = mittel(g, "test_acc")
            f1, _ = mittel(g, "test_f1")
            name = k[0] + ("" if k[4] == "ja" else " (ohne Vortraining)")
            methode = k[1] + (f" r={k[2]}" if k[1] == "lora" else "") + (f" [{k[3]}]" if k[3] not in ("", "qkv") else "")
            zeilen.append((acc, [name, methode, k[5], k[6], len(g), prozent(acc, acc_s), prozent(f1),
                                 f"{int(g[0]['trainierbar']):,}", f"{mittel(g, 'zeit_s')[0]:.0f}", g[0]["latenz_ms"]]))
            if k[4] == "ja":
                familie = "ViT (Station 1)" if k[0].startswith(("vit", "deit")) else "CNN (Station 1)"
                vergleich.append((familie, f"{name} {methode}", "ja", acc, int(g[0]["parameter"]), g[0]["latenz_ms"]))
        teile += ["## Station 1: Adaptation", "",
                  tabelle(["Modell", "Methode", "lr", "Epochen", "Seeds", "Test-Acc", "Macro-F1", "trainierbar",
                           "Zeit (s)", "Latenz (ms)"],
                          [z for _, z in sorted(zeilen, key=lambda x: -x[0])]), ""]

    # Station 2
    if s2:
        teacher_acc = float(s2[0]["teacher_test_acc"])
        gruppen = gruppieren(s2, ["student", "kd", "alpha", "temperatur"])
        ohne = {k[0]: mittel(g, "test_acc")[0] for k, g in gruppen.items() if k[1] == "nein"}
        zeilen = []
        for k, g in gruppen.items():
            acc, acc_s = mittel(g, "test_acc")
            luecke = ""
            if k[1] == "ja" and k[0] in ohne and teacher_acc > ohne[k[0]]:
                luecke = f"{100 * (acc - ohne[k[0]]) / (teacher_acc - ohne[k[0]]):.0f} %"
            kd = "ohne KD" if k[1] == "nein" else f"KD α={k[2]}, T={k[3]}"
            zeilen.append([k[0], kd, len(g), prozent(acc, acc_s), prozent(mittel(g, "test_f1")[0]),
                           f"{int(g[0]['parameter']):,}", g[0]["groesse_mb"], g[0]["latenz_ms"], luecke])
            vergleich.append(("Student (Station 2)", f"{k[0]} {kd}", "ja", acc, int(g[0]["parameter"]), g[0]["latenz_ms"]))
        teile += ["## Station 2: Compression", "",
                  f"Teacher: `{s2[0]['teacher']}`, Test-Acc {prozent(teacher_acc)}, "
                  f"{int(s2[0]['teacher_parameter']):,} Parameter, {s2[0]['teacher_latenz_ms']} ms pro Bild", "",
                  tabelle(["Student", "Variante", "Seeds", "Test-Acc", "Macro-F1", "Parameter", "MB", "Latenz (ms)",
                           "Lücke geschlossen"], zeilen), ""]

    # Station 3
    if s3:
        zeilen = sorted(s3, key=lambda z: -float(z["test_acc"]))
        teile += ["## Station 3: Generalization (CLIP Zero-Shot)", "",
                  tabelle(["CLIP", "Gewichte", "Klassennamen", "Prompt", "Test-Acc", "Macro-F1"],
                          [[z["modell"], z["pretrained"], z["namen"], z["prompt"], prozent(float(z["test_acc"])),
                            prozent(float(z["test_f1"]))] for z in zeilen]), ""]
        for modell, g in gruppieren(s3, ["modell"]).items():
            b = max(g, key=lambda z: float(z["test_acc"]))
            vergleich.append(("CLIP Zero-Shot", f"{modell[0]} ({b['namen']}, {b['prompt']})", "nein",
                              float(b["test_acc"]), int(b["parameter_bild"]), b["latenz_ms"]))

    # Gesamtvergleich: bestes Modell je Familie
    if vergleich:
        beste = {}
        for v in vergleich:
            if v[0] not in beste or v[3] > beste[v[0]][3]:
                beste[v[0]] = v
        teile += ["## Vergleich: Kann CLIP den ViT ersetzen?", "",
                  tabelle(["Familie", "bestes Modell", "auf den Daten trainiert", "Test-Acc", "Parameter", "Latenz (ms)"],
                          [[v[0], v[1], v[2], prozent(v[3]), f"{v[4]:,}", v[5]]
                           for v in sorted(beste.values(), key=lambda v: -v[3])]), ""]

    text = "\n".join(teile)
    (ordner / "zusammenfassung.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

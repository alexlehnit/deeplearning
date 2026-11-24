import math
import torch
from torch import nn



# Ersetzt eine normale Linear-Schicht: W bleibt fest, gelernt werden nur A und B
class LoRALinear(nn.Module):
    
    def __init__(self, original: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        # Damit die Stärke der Korrektur nicht direkt vom Rang abhängt
        self.skalierung = alpha / r
        # A und B sind klein, zusammen deutlich weniger Parameter als W
        self.A = nn.Parameter(torch.empty(r, original.in_features))
        # B startet mit 0, am Anfang rechnet das Modell also wie das vortrainierte
        self.B = nn.Parameter(torch.zeros(original.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # Normale Ausgabe plus die gelernte Korrektur
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.skalierung


# Wo die Schichten in einem timm-ViT-Block liegen
ZIELE = {
    "qkv": lambda block: (block.attn, "qkv"),
    "proj": lambda block: (block.attn, "proj"),
    "fc1": lambda block: (block.mlp, "fc1"),
    "fc2": lambda block: (block.mlp, "fc2"),
}
def lora_einbauen(modell, r=8, alpha=16, ziele=("qkv",)):
    for block in modell.blocks:
        for ziel in ziele:
            eltern, name = ZIELE[ziel](block)
            # Alte Schicht durch die LoRA-Variante austauschen
            setattr(eltern, name, LoRALinear(getattr(eltern, name), r=r, alpha=alpha))
    return modell


def lora_zusammenfuehren(modell):
    """Rechnet W + (alpha/r) * B @ A fest ein, damit Latenz und Parameter ohne LoRA-Zusatzaufwand gemessen werden."""
    for eltern in list(modell.modules()):
        for name, kind in list(eltern.named_children()):
            if isinstance(kind, LoRALinear):
                with torch.no_grad():
                    # Korrektur fest auf W addieren, das Ergebnis bleibt gleich
                    kind.original.weight += kind.skalierung * (kind.B @ kind.A)
                setattr(eltern, name, kind.original)
    return modell


# Kleiner Test: vor und nach dem Zusammenführen muss dasselbe rauskommen
if __name__ == "__main__":
    test = nn.Sequential(LoRALinear(nn.Linear(5, 7)))
    nn.init.normal_(test[0].B)
    x = torch.randn(3, 5)
    vorher = test(x)
    assert torch.allclose(vorher, lora_zusammenfuehren(test)(x), atol=1e-5) and isinstance(test[0], nn.Linear)

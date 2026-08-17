import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss multiclase (soft), ignorando `ignore_index` tanto en el
    calculo de cada clase como al promediar. Pensada para complementar a
    CrossEntropyLoss en problemas con clases muy minoritarias (mani, sorgo):
    los pesos por clase de CE ayudan pero no alcanzan cuando una clase es una
    fraccion muy chica de los pixeles utiles; Dice compara directamente la
    superposicion predicha/real por clase, sin importar cuantos pixeles tiene.
    """
    def __init__(self, num_classes, ignore_index=0, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        # float32: softmax/sumas de dice son sensibles a underflow en fp16
        # cuando se entrena con autocast.
        probs = torch.softmax(logits.float(), dim=1)
        mask = (target != self.ignore_index)

        target_valido = target.clone()
        target_valido[~mask] = 0
        one_hot = torch.zeros_like(probs).scatter_(1, target_valido.unsqueeze(1), 1)

        mask_f = mask.unsqueeze(1).float()
        probs = probs * mask_f
        one_hot = one_hot * mask_f

        dims = (0, 2, 3)
        interseccion = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice_por_clase = (2 * interseccion + self.smooth) / (union + self.smooth)

        clases = [c for c in range(self.num_classes) if c != self.ignore_index]
        return 1 - dice_por_clase[clases].mean()


class CEDiceLoss(nn.Module):
    """CrossEntropy (con pesos por clase) + Dice, combinadas por promedio
    ponderado. CE da buen gradiente pixel a pixel desde el arranque; Dice
    empuja a mejorar la superposicion de las clases raras, que CE por si sola
    tiende a subestimar aun con pesos altos.
    """
    def __init__(self, weight, num_classes, ignore_index=0, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (1 - self.dice_weight) * self.ce(logits, target) + \
            self.dice_weight * self.dice(logits, target)

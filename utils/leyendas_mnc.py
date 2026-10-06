"""Leyendas oficiales MNC. Las mascaras conservan los codigos originales."""
import numpy as np

REMAP_2021 = np.zeros(256, dtype=np.uint8)
REMAP_2021[10], REMAP_2021[11], REMAP_2021[16] = 2, 3, 4
REMAP_2021[[12, 13, 14, 15, 17, 18, 19, 21, 22, 26, 28]] = 1
CODIGOS_2021 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 26, 28, 255}


def verificar_codigos(codigos, ignorar_codigo20=False):
    desconocidos = set(map(int, codigos)) - CODIGOS_2021
    if ignorar_codigo20:
        desconocidos.discard(20)
    if desconocidos:
        raise ValueError(f'Codigos MNC 2021 sin documentar: {sorted(desconocidos)}')

"""
Calcula la distribucion de clases y los pesos sugeridos para CrossEntropyLoss
sobre el conjunto de tiles de entrenamiento.

Uso:
    python3 src/0b_calcular_pesos.py
"""
import numpy as np
from pathlib import Path

DIR = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba21")
TILES_TRAIN = ['20HLH','20HLJ','20HMJ','20HMK','20HNK',
               '20HLG','20HMG','20HMH','20HNG',
               '19HGC','20HKH']
NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani", 5:"Sorgo"}

conteos = np.zeros(6, dtype=np.int64)
for t in TILES_TRAIN:
    f = DIR / f"dataset_{t}.npz"
    if not f.exists():
        print(f"{t}: no existe, salteado")
        continue
    Y = np.load(f)["Y"]
    for c in range(6):
        conteos[c] += (Y == c).sum()

total = conteos[1:].sum()
print("\nDistribucion en TRAIN (excluye NoData):")
for c in range(1, 6):
    print(f"  {NOMBRES[c]:<8}: {conteos[c]:>14,} px  ({conteos[c]/total*100:6.3f}%)")
print(f"  {'NoData':<8}: {conteos[0]:>14,} px  (ignorado)")

print("\nPesos — raiz del inverso de frecuencia (recomendado):")
pesos = [0.0]
for c in range(1, 6):
    p = np.sqrt(total / (5 * conteos[c])) if conteos[c] > 0 else 0.0
    pesos.append(round(float(p), 3))
print(f"  PESOS = {pesos}")

print("\nPesos — inverso puro (mas agresivo, puede desestabilizar):")
pesos2 = [0.0]
for c in range(1, 6):
    p = total / (5 * conteos[c]) if conteos[c] > 0 else 0.0
    pesos2.append(round(float(p), 3))
print(f"  PESOS = {pesos2}")

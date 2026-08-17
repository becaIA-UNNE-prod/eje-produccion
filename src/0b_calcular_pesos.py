"""
Calcula la distribucion de clases y los pesos sugeridos para CrossEntropyLoss
sobre el conjunto de tiles de entrenamiento.

Uso:
    python3 src/0b_calcular_pesos.py
"""
import numpy as np
from pathlib import Path

# Mismo dataset y split de TRAIN que 1_train_cordoba.py (Cordoba completa,
# 5 clases sin Sorgo).
DIR = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")
TILES_TRAIN = ['19HGC', '20HLG', '20HLH', '20HMJ', '20HMK', '20HNK', '20JML',
               '20HKH', '20HMG', '20HNG', '20HPG', '20HPH', '20JNL']
NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani"}
NUM_CLASSES = 5

conteos = np.zeros(NUM_CLASSES, dtype=np.int64)
for t in TILES_TRAIN:
    f = DIR / f"dataset_{t}_Y.npy"
    if not f.exists():
        print(f"{t}: no existe, salteado")
        continue
    Y = np.load(f, mmap_mode="r")
    for c in range(NUM_CLASSES):
        conteos[c] += (Y == c).sum()

total = conteos[1:].sum()
print("\nDistribucion en TRAIN (excluye NoData):")
for c in range(1, NUM_CLASSES):
    print(f"  {NOMBRES[c]:<8}: {conteos[c]:>14,} px  ({conteos[c]/total*100:6.3f}%)")
print(f"  {'NoData':<8}: {conteos[0]:>14,} px  (ignorado)")

n = NUM_CLASSES - 1
print("\nPesos — raiz del inverso de frecuencia (recomendado):")
pesos = [0.0]
for c in range(1, NUM_CLASSES):
    p = np.sqrt(total / (n * conteos[c])) if conteos[c] > 0 else 0.0
    pesos.append(round(float(p), 3))
print(f"  PESOS = {pesos}")

print("\nPesos — inverso puro (mas agresivo, puede desestabilizar):")
pesos2 = [0.0]
for c in range(1, NUM_CLASSES):
    p = total / (n * conteos[c]) if conteos[c] > 0 else 0.0
    pesos2.append(round(float(p), 3))
print(f"  PESOS = {pesos2}")

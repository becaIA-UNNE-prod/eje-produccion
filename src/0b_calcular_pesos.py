"""
Calcula la distribucion de clases y los pesos sugeridos para CrossEntropyLoss
sobre el conjunto de tiles de entrenamiento.

Uso:
    python3 src/0b_calcular_pesos.py
    python3 src/0b_calcular_pesos.py --dirs <DIR_TRAIN> <DIR_TRAIN_EXTRA>
(con --dirs: los 13 tiles de TRAIN 2019-2020 de cada dir, como
1_train_cordoba.py --dir-train/--dir-train-extra; la ultima linea es
"PESOS_ARGS ..." para pasarle a --pesos)
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.multicampana import SPLIT, contar_etiquetas, pesos_desde_conteos

# Mismo dataset y split de TRAIN que 1_train_cordoba.py (Cordoba completa,
# 5 clases sin Sorgo).
DIR = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")
TILES_TRAIN = ['19HGC', '20HLG', '20HLH', '20HMJ', '20HMK', '20HNK', '20JML',
               '20HKH', '20HMG', '20HNG', '20HPG', '20HPH', '20JNL',
               '20HLJ_2017', '20HLK_2017', '20JLL_2017', '20JNL_2017']
NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani"}
NUM_CLASSES = 5

def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, default=None)
    return ap


def seleccionar_rutas(dirs=None):
    if dirs:
        rutas = [d / f"dataset_{t}_Y.npy" for d in dirs
                 for t in SPLIT['train']]
    else:
        rutas = [DIR / f"dataset_{t}_Y.npy" for t in TILES_TRAIN]

    return rutas


def contar_rutas(rutas, estricto=False):
    conteos = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in rutas:
        if not f.exists():
            if estricto:
                raise FileNotFoundError(f)
            print(f"{f}: no existe, salteado")
            continue
        Y = np.load(f, mmap_mode="r")
        conteos += contar_etiquetas(Y)

    return conteos


def imprimir_pesos(conteos):
    pesos = pesos_desde_conteos(conteos)
    total = conteos[1:].sum()
    print("\nDistribucion en TRAIN (excluye NoData):")
    for c in range(1, NUM_CLASSES):
        print(f"  {NOMBRES[c]:<8}: {conteos[c]:>14,} px  ({conteos[c]/total*100:6.3f}%)")
    print(f"  {'NoData':<8}: {conteos[0]:>14,} px  (ignorado)")

    n = NUM_CLASSES - 1
    print("\nPesos — raiz del inverso de frecuencia (recomendado):")
    print(f"  PESOS = {pesos}")
    pesos_sqrt = pesos

    print("\nPesos — inverso puro (mas agresivo, puede desestabilizar):")
    pesos2 = [0.0]
    for c in range(1, NUM_CLASSES):
        p = total / (n * conteos[c]) if conteos[c] > 0 else 0.0
        pesos2.append(round(float(p), 3))
    print(f"  PESOS = {pesos2}")
    print("PESOS_ARGS " + " ".join(str(p) for p in pesos_sqrt))


def main(argv=None):
    args = crear_parser().parse_args(argv)
    conteos = contar_rutas(seleccionar_rutas(args.dirs), estricto=bool(args.dirs))
    imprimir_pesos(conteos)


if __name__ == "__main__":
    main()

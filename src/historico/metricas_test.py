"""
Herramienta histórica oct-abr (20HLJ/20HLK); para TEST vigente usar
evaluar_test_actual.py. Tabla de metricas por clase calculada contra la
inferencia completa en disco (el GeoTIFF que escribe 2_inferencia.py), no
sobre el mosaico de parches de 6_generar_comparacion_visual.py.

A diferencia de 5_generar_matriz_confusion.py (que reporta solo el recall
por clase y la accuracy global), aca se reportan precision, recall, F1 e
IoU por clase mas los promedios macro, que es lo que hace comparables estas
corridas contra la bibliografia de segmentacion.

Uso:
    python3 -m src.historico.metricas_test --modelo fcrtransunet --tiles 20HLJ 20HLK
"""
import argparse
import os
from pathlib import Path
from utils.rutas import LEGACY_PREDICTIONS

import numpy as np
import rasterio

# Mismo remapeo que src.historico.comparar_inferencia_etiqueta: la etiqueta en disco
# trae los codigos IDVER originales, no el esquema 0-4 del entrenamiento.
REMAP_5 = np.zeros(256, dtype=np.int64)
REMAP_5[10] = 2   # Maiz
REMAP_5[11] = 3   # Soja
REMAP_5[16] = 4   # Mani
for _c in (12, 13, 14, 15, 17, 18, 19, 21, 22):
    REMAP_5[_c] = 1
REMAP_5[255] = 0

NOMBRES = {1: "Fondo", 2: "Maiz", 3: "Soja", 4: "Mani"}

def confusion_tile(tile, args):
    """Acumula la confusion leyendo por bloques: el tile entero (10980^2) en
    int64 no entra comodo en memoria junto con la etiqueta."""
    n = 5
    sufijo = "" if args.modelo == "unet" else f"_{args.modelo}"
    conf = np.zeros((n, n), dtype=np.int64)
    ruta_pred = args.dir_pred / f"prediccion_{tile}{sufijo}.tif"
    ruta_lab = Path(args.dir_mascaras) / f"etiqueta_mnc_{tile}_10m.tif"
    with rasterio.open(ruta_pred) as sp, rasterio.open(ruta_lab) as sl:
        alto, ancho = sp.height, sp.width
        for y in range(0, alto, 1024):
            h = min(1024, alto - y)
            w_ = rasterio.windows.Window(0, y, ancho, h)
            p = sp.read(1, window=w_).astype(np.int64)
            t = REMAP_5[sl.read(1, window=w_)]
            m = t > 0                      # ignore_index=0, igual que en el train
            np.add.at(conf, (t[m], p[m]), 1)
    return conf


def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="fcrtransunet",
                         choices=["unet", "transunet", "utae", "fcrtransunet"])
    parser.add_argument("--tiles", nargs="+", default=["20HLJ", "20HLK"])
    parser.add_argument("--dir_mascaras", default="./mascaras_procesadas")
    parser.add_argument("--dir-pred", required=os.environ.get("CORDOBA_REGISTRO_LIVIANO") == "1", type=Path, default=LEGACY_PREDICTIONS)
    return parser


def main(argv=None):
    args = crear_parser().parse_args(argv)
    n = 5
    conf_total = np.zeros((n, n), dtype=np.int64)

    print(f"Modelo: {args.modelo}\n")
    for tile in args.tiles:
        c = confusion_tile(tile, args)
        conf_total += c
        acc = np.trace(c[1:, 1:]) / c[1:, 1:].sum() * 100
        print(f"  {tile}: accuracy (sin NoData) = {acc:.2f}%")

    print()
    clases = [c for c in range(1, n) if conf_total[c].sum() > 0]
    print(f"{'Clase':8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'IoU':>8s} {'Pixeles':>14s}")
    f1s, ious = [], []
    for c in clases:
        tp = conf_total[c, c]
        fn = conf_total[c, :].sum() - tp
        fp = conf_total[:, c].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        f1s.append(f1); ious.append(iou)
        print(f"{NOMBRES[c]:8s} {prec*100:9.2f}% {rec*100:7.2f}% {f1*100:7.2f}% {iou*100:7.2f}% {conf_total[c].sum():14,d}")

    acc_g = np.trace(conf_total[1:, 1:]) / conf_total[1:, 1:].sum() * 100
    print(f"\nAccuracy global (sin NoData): {acc_g:.2f}%")
    print(f"F1 macro: {np.mean(f1s)*100:.2f}%   mIoU: {np.mean(ious)*100:.2f}%")

    # Solo cultivos: es la metrica que importa para el producto, Fondo domina el
    # conteo y tapa el desempeno real sobre Maiz/Soja/Mani.
    cult = [c for c in clases if c >= 2]
    if cult:
        idx = np.ix_(cult, cult)
        sub = conf_total[idx]
        acc_c = np.trace(sub) / conf_total[cult, :].sum() * 100
        print(f"Accuracy solo cultivos (Maiz/Soja/Mani, contra todas las predichas): {acc_c:.2f}%")


if __name__ == "__main__":
    main()

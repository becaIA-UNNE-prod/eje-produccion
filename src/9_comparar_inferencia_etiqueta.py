"""
Compara la inferencia real de un modelo (el GeoTIFF que escribe
2_inferencia.py) contra la etiqueta real del tile -- a diferencia de
6_generar_comparacion_visual.py (que arma un mosaico con parches del
dataset .npz), esto usa el mapa de clasificacion completo tal como quedo
escrito en disco.

Uso:
    python3 src/9_comparar_inferencia_etiqueta.py --tile 20JML --modelo fcrtransunet
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rutas import LEGACY_PREDICTIONS, LEGACY_FIGURES

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORES = {0: [0, 0, 0], 1: [70, 130, 180], 2: [255, 165, 0], 3: [0, 100, 0], 4: [165, 42, 42]}
NOMBRES = {1: "Fondo", 2: "Maiz", 3: "Soja", 4: "Mani"}

# etiqueta_mnc_<TILE>_10m.tif trae los codigos IDVER originales (10=Maiz,
# 11=Soja, 16=Mani, etc.), no el esquema 0-4 -- ese remapeo se aplica al
# vuelo al generar el dataset de entrenamiento (ver REMAP_5 en
# 03c_generar_dataset_cordoba_f16.py), no queda guardado en el .tif.
REMAP_5 = np.zeros(256, dtype=np.int64)
REMAP_5[10] = 2   # Maiz
REMAP_5[11] = 3   # Soja
REMAP_5[16] = 4   # Mani
for _c in (12, 13, 14, 15, 17, 18, 19, 21, 22):  # resto (incl. Sorgo) -> Fondo
    REMAP_5[_c] = 1
REMAP_5[255] = 0  # Sin datos -> NoData


def clase_a_rgb(mapa):
    img = np.zeros((*mapa.shape, 3), dtype=np.uint8)
    for c, color in COLORES.items():
        img[mapa == c] = color
    return img


def leer_downsampled(ruta, out_shape):
    with rasterio.open(ruta) as src:
        return src.read(1, out_shape=out_shape, resampling=Resampling.nearest)


parser = argparse.ArgumentParser()
parser.add_argument("--tile", default="20JML")
parser.add_argument("--modelo", default="fcrtransunet",
                     choices=["unet", "transunet", "utae", "fcrtransunet"])
parser.add_argument("--dir_mascaras", default="./mascaras_procesadas")
parser.add_argument("--max_lado", type=int, default=2048)
parser.add_argument("--dir-pred", type=Path, default=LEGACY_PREDICTIONS)
parser.add_argument("--dir-salida", type=Path, default=LEGACY_FIGURES)
args = parser.parse_args()

sufijo = "" if args.modelo == "unet" else f"_{args.modelo}"
ruta_pred = args.dir_pred / f"prediccion_{args.tile}{sufijo}.tif"
ruta_etiqueta = Path(args.dir_mascaras) / f"etiqueta_mnc_{args.tile}_10m.tif"

with rasterio.open(ruta_pred) as src:
    alto, ancho = src.height, src.width
escala = min(1.0, args.max_lado / max(alto, ancho))
out_shape = (max(1, int(alto * escala)), max(1, int(ancho * escala)))

pred = leer_downsampled(ruta_pred, out_shape)
target = REMAP_5[leer_downsampled(ruta_etiqueta, out_shape)]

# Igual que 6_generar_comparacion_visual.py: el mapa de errores solo evalua
# sobre clases de cultivo (Maiz/Soja/Mani) -- Fondo y NoData quedan grises,
# porque el interes real es donde se equivoca el modelo entre cultivos.
mapa_error = np.zeros((*target.shape, 3), dtype=np.uint8)
mascara = target > 1
mapa_error[mascara & (target == pred)] = [0, 200, 0]
mapa_error[mascara & (target != pred)] = [200, 0, 0]
mapa_error[~mascara] = [50, 50, 50]

acc = (pred[mascara] == target[mascara]).mean() * 100 if mascara.sum() > 0 else float("nan")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(f"Inferencia vs. etiqueta — Tile {args.tile} ({args.modelo})\n"
             f"Accuracy en clases de cultivo: {acc:.1f}%", fontsize=13)
axes[0].imshow(clase_a_rgb(target)); axes[0].set_title("Etiqueta (real)"); axes[0].axis('off')
axes[1].imshow(clase_a_rgb(pred));   axes[1].set_title("Inferencia (prediccion)"); axes[1].axis('off')
axes[2].imshow(mapa_error);          axes[2].set_title("Mapa de errores\n(Verde=Acierto, Rojo=Error, Gris=Fondo)"); axes[2].axis('off')

leyenda = [Patch(color=[c/255 for c in COLORES[k]], label=v) for k, v in NOMBRES.items()]
fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
args.dir_salida.mkdir(parents=True, exist_ok=True)
ruta_salida = args.dir_salida / f"comparacion_inferencia_{args.tile}_{args.modelo}.png"
plt.savefig(ruta_salida, dpi=140, bbox_inches='tight')
plt.close()
print(f"Guardado: {ruta_salida}")

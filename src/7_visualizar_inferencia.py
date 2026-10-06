"""
Convierte un mapa de clasificacion (GeoTIFF de 1 banda, salida de
2_inferencia.py) en una imagen PNG a color, para poder verlo sin un SIG.
El tile completo (10980x10980) se lee downsampleado a un tamano razonable
para pantalla -- no tiene sentido una imagen a resolucion nativa.

Uso:
    python3 src/7_visualizar_inferencia.py prediccion_20HMK.tif
    python3 src/7_visualizar_inferencia.py prediccion_20HMK_fcrtransunet.tif --max_lado 3000
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Misma paleta que 6_generar_comparacion_visual.py (esquema vigente, 5 clases sin Sorgo)
COLORES = {0: [0, 0, 0], 1: [70, 130, 180], 2: [255, 165, 0], 3: [0, 100, 0], 4: [165, 42, 42]}
NOMBRES = {1: "Fondo", 2: "Maiz", 3: "Soja", 4: "Mani"}

parser = argparse.ArgumentParser()
parser.add_argument("tif", help="Ruta al GeoTIFF de clasificacion (1 banda, 0-4)")
parser.add_argument("--max_lado", type=int, default=2048,
                     help="Lado maximo (px) de la imagen de salida (downsampling)")
parser.add_argument("--salida", default=None, help="Ruta de salida (default: mismo nombre, .png)")
args = parser.parse_args()

ruta_tif = Path(args.tif)
with rasterio.open(ruta_tif) as src:
    alto, ancho = src.height, src.width
    escala = min(1.0, args.max_lado / max(alto, ancho))
    out_h, out_w = max(1, int(alto * escala)), max(1, int(ancho * escala))
    mapa = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest)

img = np.zeros((*mapa.shape, 3), dtype=np.uint8)
for clase, color in COLORES.items():
    img[mapa == clase] = color

ruta_salida = Path(args.salida) if args.salida else ruta_tif.with_suffix(".png")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.set_title(f"Clasificacion — {ruta_tif.stem}\n({ancho}x{alto} px, mostrado a {out_w}x{out_h})")
ax.axis('off')
leyenda = [Patch(color=[c / 255 for c in COLORES[k]], label=v) for k, v in NOMBRES.items()]
fig.legend(handles=leyenda, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {ruta_salida}")

"""Lectura y representación de mapas para las comparaciones históricas."""
from functools import wraps
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ESTILO_GRAFICO = {"font.size": 9, "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e",
                  "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b",
                  "axes.spines.top": False, "axes.spines.right": False}


def con_estilo(funcion):
    @wraps(funcion)
    def ejecutar(*args, **kwargs):
        with plt.rc_context(ESTILO_GRAFICO):
            return funcion(*args, **kwargs)
    return ejecutar


def leer_reducido(ruta, factor, remap=None):
    with rasterio.open(ruta) as s:
        a = s.read(1, out_shape=(s.height // factor, s.width // factor), resampling=Resampling.nearest)
    return remap[a] if remap is not None else a


def a_rgb(a, colores):
    img = np.zeros((*a.shape, 3), dtype=np.uint8)
    for c, col in colores.items():
        img[a == c] = col
    return img

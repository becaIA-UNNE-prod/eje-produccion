"""
Compara dos variantes de los modelos dic-may (--comparacion):
  1718  "base" (pipeline_dicmay.sh, entrenado solo con 2019-2020) vs "+17-18"
        (pipeline_dicmay_1718.sh, 2019-2020 + 2017-2018 Nivel3), los dos
        inferidos sobre los composites 2018-2019 con mascara SCL
  scl   modelos base inferidos sobre los composites 2018-2019 de 02c (sin
        mascara de nubes) vs los de 02d (SCL), ver rehacer_2018_2019_scl.sh
Lee las salidas de src.historico.evaluar_contra_mnc (resumen.json / metricas.csv) y los
historial.npy de cada entrenamiento.

Genera en --dir-salida (default $Y3/comparacion_<comparacion>):
  metricas_<eval>.png        barras OA/kappa/F1 por grupo de tiles, 4 modelos
  kappa_por_tile_2018_2019.png   kappa por tile, base vs +17-18 (los 17 tiles)
  curvas_entrenamiento.png   Val Loss / Val Acc por epoca de los 4 entrenamientos
  mapa_<TILE>.png            MNC 2018/2019 vs predicciones base y +17-18
  comparacion.csv            todas las metricas + delta (+17-18 - base)

Uso:
    python3 -m src.historico.comparar_con_1718 --comparacion 1718
    python3 -m src.historico.comparar_con_1718 --comparacion scl
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.figuras.comparacion_1718 import con_estilo, leer_reducido as leer_mapa_reducido, a_rgb as mapa_a_rgb

Y3 = Path("/mnt/yacy_3/prod_bis/ferreyra")
DIR_MASCARAS = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
MODELOS = ["unet", "fcrtransunet"]
TITULOS = {"2018_2019": ("Interanual: 2018-19 vs MNC 2018/2019", ["train", "val", "test", "todos"]),
           "2019_2020_dicmay": ("Mismo anio: 2019-20 vs MNC 2020", ["val", "test", "todos"])}
# Por variante: dir de evaluacion por evaluacion, dir de predicciones 2018-2019 y
# dir del entrenamiento ({m} = modelo; None = no graficar curvas). La primera
# variante es la de referencia (barras con trama, delta = segunda - primera).
COMPARACIONES = {
    "1718": {
        "base":   {"evals": {"2018_2019": "evaluacion_2018_2019_scl",
                             "2019_2020_dicmay": "evaluacion_2019_2020_dicmay"},
                   "pred": "predicciones_2018_2019_scl", "exp": "exp_cordoba_dicmay_{m}_f16"},
        "+17-18": {"evals": {"2018_2019": "evaluacion_2018_2019_con1718",
                             "2019_2020_dicmay": "evaluacion_2019_2020_dicmay_con1718"},
                   "pred": "predicciones_2018_2019_con1718", "exp": "exp_cordoba_dicmay_con1718_{m}_f16"},
    },
    "scl": {
        "02c":     {"evals": {"2018_2019": "evaluacion_2018_2019"}, "pred": "predicciones_2018_2019", "exp": None},
        "02d SCL": {"evals": {"2018_2019": "evaluacion_2018_2019_scl"}, "pred": "predicciones_2018_2019_scl", "exp": None},
    },
}
VARIANTES = COMPARACIONES["1718"]
EVALS = TITULOS
REF, NUEVA = list(VARIANTES)
METRICAS = ["OA", "kappa", "F1_macro", "F1_Maiz", "F1_Soja", "F1_Mani"]

# Hue = modelo (slots 1 y 2 de la paleta categorica), relleno = variante:
# base con trama y color claro, +17-18 solido.
COLOR = {"unet": "#2a78d6", "fcrtransunet": "#eb6834"}
CLARO = {"unet": "#a9c8ee", "fcrtransunet": "#f6c0aa"}
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

# Mismos colores de clase que src.historico.comparar_inferencia_etiqueta
COLORES_CLASE = {0: [255, 255, 255], 1: [70, 130, 180], 2: [255, 165, 0], 3: [0, 100, 0], 4: [165, 42, 42]}
NOMBRES_CLASE = {1: "Fondo", 2: "Maiz", 3: "Soja", 4: "Mani"}
REMAP_1819 = np.zeros(256, dtype=np.int64)
REMAP_1819[[0, 3]] = 2
REMAP_1819[[1, 2]] = 3
REMAP_1819[4] = 4
REMAP_1819[[5, 6, 7, 8, 9, 11, 12, 13, 14, 15]] = 1




def series():
    for m in MODELOS:
        for v in VARIANTES:
            yield m, v


def estilo(m, v):
    if v == REF:
        return dict(color=CLARO[m], edgecolor=COLOR[m], hatch="////", linewidth=0)
    return dict(color=COLOR[m], linewidth=0)


def leyenda_series(ax, **kw):
    hs = [Patch(label=f"{m} {v}", **estilo(m, v)) for m, v in series()]
    ax.legend(handles=hs, frameon=False, **kw)


def cargar_resumen(eval_key, var):
    p = Y3 / VARIANTES[var]["evals"][eval_key] / "resumen.json"
    return json.loads(p.read_text()) if p.exists() else None


@con_estilo
def graficar_metricas(eval_key, salida):
    titulo, grupos = EVALS[eval_key]
    res = {v: cargar_resumen(eval_key, v) for v in VARIANTES}
    if any(r is None for r in res.values()):
        print(f"{eval_key}: falta resumen.json, salteado")
        return []
    filas = []
    fig, axs = plt.subplots(1, len(grupos), figsize=(3.6 * len(grupos), 3.8), sharey=True)
    ancho = 0.2
    x = np.arange(len(METRICAS))
    for ax, g in zip(np.atleast_1d(axs), grupos):
        for i, (m, v) in enumerate(series()):
            r = res[v].get(f"{m}/{g}")
            if not r:
                continue
            vals = [r.get(k, np.nan) for k in METRICAS]
            ax.bar(x + (i - 1.5) * ancho, vals, ancho * 0.9, **estilo(m, v))
            filas.append({"evaluacion": eval_key, "grupo": g, "modelo": m, "variante": v,
                          **{k: round(float(val), 4) for k, val in zip(METRICAS, vals)}})
        ax.set_xticks(x, [k.replace("F1_", "F1 ").replace("F1 macro", "F1\nmacro") for k in METRICAS])
        ax.set_title(f"{g} ({len(res[REF].get(f'unet/{g}', {}).get('tiles', []))} tiles)")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRILLA, linewidth=0.8)
        ax.set_axisbelow(True)
    leyenda_series(np.atleast_1d(axs)[0], loc="upper left", fontsize=8)
    fig.suptitle(titulo, color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(salida / f"metricas_{eval_key}.png", dpi=140)
    plt.close(fig)
    return filas


@con_estilo
def graficar_kappa_tiles(salida):
    datos = {}
    for v, cfg in VARIANTES.items():
        p = Y3 / cfg["evals"]["2018_2019"] / "metricas.csv"
        if not p.exists():
            print("kappa por tile: falta metricas.csv, salteado")
            return
        for f in csv.DictReader(open(p)):
            if f["tile"] != "TOTAL":
                datos[(f["modelo"], v, f["tile"])] = (float(f["kappa"]), f["grupo"])
    tiles = sorted({t for (_, _, t) in datos},
                   key=lambda t: datos.get(("fcrtransunet", REF, t), (0,))[0])
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    y = np.arange(len(tiles))
    for ax, m in zip(axs, MODELOS):
        for i, t in enumerate(tiles):
            b = datos.get((m, REF, t), (np.nan,))[0]
            n = datos.get((m, NUEVA, t), (np.nan,))[0]
            ax.plot([b, n], [i, i], color=GRILLA, linewidth=2, zorder=1)
        b = [datos.get((m, REF, t), (np.nan,))[0] for t in tiles]
        n = [datos.get((m, NUEVA, t), (np.nan,))[0] for t in tiles]
        ax.scatter(b, y, s=48, facecolor="white", edgecolor=COLOR[m], linewidth=2, zorder=2, label=REF)
        ax.scatter(n, y, s=48, color=COLOR[m], edgecolor="white", linewidth=1.5, zorder=3, label=NUEVA)
        ax.set_yticks(y, [f"{t} ({datos[(m, REF, t)][1]})" if (m, REF, t) in datos else t for t in tiles])
        ax.set_xlim(0.2, 0.9)
        ax.xaxis.grid(True, color=GRILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("kappa (tile completo)")
        d = np.nanmean(np.array(n) - np.array(b))
        ax.set_title(f"{m}  (delta medio {d:+.3f})")
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"Kappa por tile, 2018-19 vs MNC 2018/2019: {REF} vs {NUEVA} (rol del tile en el split 2019-20)",
                 color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(salida / "kappa_por_tile_2018_2019.png", dpi=140)
    plt.close(fig)


@con_estilo
def graficar_curvas(salida):
    if any(cfg["exp"] is None for cfg in VARIANTES.values()):
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m, v in series():
        p = Y3 / "dataset" / VARIANTES[v]["exp"].format(m=m) / "historial.npy"
        if not p.exists():
            continue
        h = np.load(p)
        ls = "--" if v == REF else "-"
        for ax, col in zip(axs, (3, 4)):
            ax.plot(h[:, 0], h[:, col], ls, color=COLOR[m], linewidth=2, label=f"{m} {v}")
            k = np.argmin(h[:, 3])
            ax.plot(h[k, 0], h[k, col], "o", color=COLOR[m], markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5)
    for ax, t in zip(axs, ("Val Loss", "Val Accuracy (parches)")):
        ax.set_title(t)
        ax.set_xlabel("epoca")
        ax.yaxis.grid(True, color=GRILLA, linewidth=0.8)
    axs[0].legend(frameon=False)
    fig.suptitle("Curvas de validacion (punto = best_model.pth, minimo de Val Loss)", color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(salida / "curvas_entrenamiento.png", dpi=140)
    plt.close(fig)


def leer_reducido(ruta, factor, remap=None):
    return leer_mapa_reducido(ruta, factor, remap)

def a_rgb(a):
    return mapa_a_rgb(a, COLORES_CLASE)

@con_estilo
def graficar_mapa(tile, salida, factor=8):
    paneles = [("MNC 2018/2019", DIR_MASCARAS / f"etiqueta_mnc1819_{tile}_10m.tif", REMAP_1819)]
    for m, v in series():
        suf = "" if m == "unet" else f"_{m}"
        paneles.append((f"{m} {v}", Y3 / VARIANTES[v]["pred"] / f"prediccion_{tile}{suf}.tif", None))
    paneles = [p for p in paneles if p[1].exists()]
    if len(paneles) < 2:
        print(f"mapa {tile}: faltan archivos, salteado")
        return
    ref = leer_reducido(paneles[0][1], factor, paneles[0][2])
    fig, axs = plt.subplots(1, len(paneles), figsize=(3.4 * len(paneles), 3.9))
    for ax, (nombre, ruta, remap) in zip(axs, paneles):
        a = leer_reducido(ruta, factor, remap)
        ax.imshow(a_rgb(a), interpolation="nearest")
        if remap is None:
            v = (ref > 0) & (a > 0)
            nombre += f"\nacuerdo con MNC: {(ref[v] == a[v]).mean():.3f}"
        ax.set_title(nombre, fontsize=9)
        ax.axis("off")
    fig.legend(handles=[Patch(color=np.array(COLORES_CLASE[k]) / 255, label=n) for k, n in NOMBRES_CLASE.items()],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{tile} - campania 2018-2019 (blanco = sin dato)", color=TINTA, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(salida / f"mapa_{tile}.png", dpi=140)
    plt.close(fig)


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparacion", choices=COMPARACIONES, default="1718")
    ap.add_argument("--dir-salida", type=Path, default=None)
    ap.add_argument("--tiles-mapa", nargs="*", default=["20HLJ", "20HLK", "20HNK", "20HMJ", "20HMK"])
    return ap


def ejecutar_comparacion(a):
    global VARIANTES, REF, NUEVA, EVALS
    VARIANTES = COMPARACIONES[a.comparacion]
    REF, NUEVA = list(VARIANTES)
    EVALS = {e: TITULOS[e] for e in VARIANTES[REF]["evals"]}
    a.dir_salida = a.dir_salida or Y3 / f"comparacion_{a.comparacion}"
    a.dir_salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for e in EVALS:
        filas += graficar_metricas(e, a.dir_salida)
    graficar_kappa_tiles(a.dir_salida)
    graficar_curvas(a.dir_salida)
    for t in a.tiles_mapa:
        graficar_mapa(t, a.dir_salida)

    # Tabla con delta (+17-18 - base) por evaluacion/grupo/modelo
    idx = {(f["evaluacion"], f["grupo"], f["modelo"], f["variante"]): f for f in filas}
    filas += [{**{c: f[c] for c in ("evaluacion", "grupo", "modelo")}, "variante": "delta",
               **{m: round(f[m] - idx[k][m], 4) for m in METRICAS}}
              for f in filas
              if f["variante"] == NUEVA and (k := (f["evaluacion"], f["grupo"], f["modelo"], REF)) in idx]
    if filas:
        with open(a.dir_salida / "comparacion.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        print(f"\n{'evaluacion':17s} {'grupo':6s} {'modelo':13s} {'variante':8s} " + " ".join(f"{m:>8s}" for m in METRICAS))
        for f in sorted(filas, key=lambda f: (f["evaluacion"], f["grupo"], f["modelo"], f["variante"])):
            print(f"{f['evaluacion']:17s} {f['grupo']:6s} {f['modelo']:13s} {f['variante']:8s} "
                  + " ".join(f"{f[m]:8.3f}" for m in METRICAS))
    print(f"\nGraficos en {a.dir_salida}")


def main(argv=None):
    return ejecutar_comparacion(crear_parser().parse_args(argv))


if __name__ == "__main__":
    main()

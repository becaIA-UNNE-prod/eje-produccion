"""
Evalua mapas de prediccion completos (salida de 2_inferencia.py) contra una
capa MNC-INTA reproyectada a 10 m, pixel a pixel sobre el tile entero.

Pensado para medir la generalizacion interanual del modelo dic-may (ver
pipeline_dicmay.sh): entrenado con imagenes + MNC 2019-2020, evaluado sobre
imagenes 2018-2019 contra el MNC 2018/2019 (01b_mascaras_mnc_2018_2019.py).
Tambien sirve con --leyenda 2020 para tener la referencia "mismo anio" con la
misma metrica (tile completo, no parches).

Se cuentan solo pixeles con etiqueta (clase > 0) y con prediccion (> 0: fuera
de la huella de las escenas 2_inferencia.py escribe 0). Los tiles se reportan
agrupados segun su rol en el split de 1_train_cordoba.py -- en 2018-2019 son
todos de un anio no visto, pero los de TRAIN son lugares vistos.

Uso:
    python3 src/11_evaluar_contra_mnc.py --leyenda 1819 --modelos unet fcrtransunet \\
        --dir-pred /mnt/yacy_3/prod_bis/ferreyra/predicciones_2018_2019 \\
        --dir-salida /mnt/yacy_3/prod_bis/ferreyra/evaluacion_2018_2019
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_MASCARAS = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
NOMBRES = ["Fondo", "Maiz", "Soja", "Mani"]   # clases 1..4

# Mismos remapeos que 03c_generar_dataset_cordoba_f16.py (2020) y
# 01b_mascaras_mnc_2018_2019.py (2018/2019).
REMAP_2020 = np.zeros(256, dtype=np.int64)
REMAP_2020[10], REMAP_2020[11], REMAP_2020[16] = 2, 3, 4
REMAP_2020[[12, 13, 14, 15, 17, 18, 19, 21, 22]] = 1
REMAP_1819 = np.zeros(256, dtype=np.int64)
REMAP_1819[[0, 3]] = 2
REMAP_1819[[1, 2]] = 3
REMAP_1819[4] = 4
REMAP_1819[[5, 6, 7, 8, 9, 11, 12, 13, 14, 15]] = 1
LEYENDAS = {"2020": (REMAP_2020, "etiqueta_mnc_{}_10m.tif"),
            "1819": (REMAP_1819, "etiqueta_mnc1819_{}_10m.tif")}

TILES_TRAIN = ['19HGC', '20HLG', '20HLH', '20HMJ', '20HMK', '20HNK', '20JML',
               '20HKH', '20HMG', '20HNG', '20HPG', '20HPH', '20JNL']
TILES_VAL = ['20HKJ', '20HMH']
TILES_TEST = ['20HLK', '20HLJ']
ROL = {**{t: "train" for t in TILES_TRAIN}, **{t: "val" for t in TILES_VAL},
       **{t: "test" for t in TILES_TEST}}


def matriz(pred_path, mask_path, remap):
    with rasterio.open(pred_path) as p, rasterio.open(mask_path) as m:
        if (p.transform, p.shape, p.crs) != (m.transform, m.shape, m.crs):
            raise ValueError(f"{pred_path.name}: grilla distinta a {mask_path.name}")
        pred = p.read(1).astype(np.int64)
        y = remap[m.read(1)]
    v = (y > 0) & (pred > 0)
    return np.bincount(y[v] * 5 + pred[v], minlength=25).reshape(5, 5)[1:, 1:]


def metricas(cm):
    n = cm.sum()
    if n == 0:
        return {"pixeles": 0}
    oa = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    rec = np.divide(np.diag(cm), cm.sum(1), out=np.full(4, np.nan), where=cm.sum(1) > 0)
    pre = np.divide(np.diag(cm), cm.sum(0), out=np.full(4, np.nan), where=cm.sum(0) > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.full(4, np.nan), where=(pre + rec) > 0)
    r = {"pixeles": int(n), "OA": oa, "kappa": (oa - pe) / (1 - pe) if pe < 1 else np.nan,
         "F1_macro": np.nanmean(f1)}
    for i, c in enumerate(NOMBRES):
        r[f"F1_{c}"], r[f"prec_{c}"], r[f"rec_{c}"] = f1[i], pre[i], rec[i]
    return r


def graficar(cm, titulo, ruta):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), NOMBRES); ax.set_yticks(range(4), NOMBRES)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("MNC (referencia)")
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leyenda", choices=LEYENDAS, required=True)
    ap.add_argument("--modelos", nargs="+", default=["unet", "fcrtransunet"])
    ap.add_argument("--dir-pred", type=Path, required=True)
    ap.add_argument("--dir-salida", type=Path, required=True)
    ap.add_argument("--tiles", nargs="*", default=TILES_TRAIN + TILES_VAL + TILES_TEST)
    a = ap.parse_args()
    remap, patron = LEYENDAS[a.leyenda]
    a.dir_salida.mkdir(parents=True, exist_ok=True)

    filas, resumen = [], {}
    for mod in a.modelos:
        suf = "" if mod == "unet" else f"_{mod}"
        cms = {}
        for t in a.tiles:
            pred, mask = a.dir_pred / f"prediccion_{t}{suf}.tif", DIR_MASCARAS / patron.format(t)
            if not pred.exists() or not mask.exists():
                print(f"{mod} {t}: falta {'prediccion' if not pred.exists() else 'mascara'}, salteado")
                continue
            cms[t] = matriz(pred, mask, remap)
            m = metricas(cms[t])
            filas.append({"modelo": mod, "grupo": ROL.get(t, "-"), "tile": t, **m})
            print(f"{mod:13s} {t} ({ROL.get(t, '-'):5s}) OA={m['OA']:.3f} kappa={m['kappa']:.3f} "
                  f"F1 Maiz={m['F1_Maiz']:.3f} Soja={m['F1_Soja']:.3f} Mani={m['F1_Mani']:.3f}", flush=True)
            np.save(a.dir_salida / f"cm_{t}{suf}.npy", cms[t])
        for grupo in ("train", "val", "test", "todos"):
            ts = [t for t in cms if grupo == "todos" or ROL.get(t) == grupo]
            if not ts:
                continue
            cm = sum(cms[t] for t in ts)
            m = metricas(cm)
            filas.append({"modelo": mod, "grupo": grupo, "tile": "TOTAL", **m})
            resumen[f"{mod}/{grupo}"] = {**m, "tiles": ts, "cm": cm.tolist()}
            graficar(cm, f"{mod} - {grupo} ({len(ts)} tiles) - MNC {a.leyenda}\n"
                         f"OA={m['OA']:.3f} kappa={m['kappa']:.3f} F1macro={m['F1_macro']:.3f}",
                     a.dir_salida / f"matriz_{mod}_{grupo}.png")

    with open(a.dir_salida / "metricas.csv", "w", newline="") as f:
        campos = list(dict.fromkeys(k for fila in filas for k in fila))
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()} for fila in filas)
    (a.dir_salida / "resumen.json").write_text(json.dumps(resumen, indent=1, default=float))

    print("\nResumen (pixeles de tile completo):")
    for k, m in resumen.items():
        print(f"  {k:24s} OA={m['OA']:.3f} kappa={m['kappa']:.3f} F1macro={m['F1_macro']:.3f} "
              f"F1 Maiz={m['F1_Maiz']:.3f} Soja={m['F1_Soja']:.3f} Mani={m['F1_Mani']:.3f}")
    print(f"\nResultados en {a.dir_salida}")

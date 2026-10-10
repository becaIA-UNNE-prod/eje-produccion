"""
Composites mensuales a partir de descargas COG (formato de 00_descargar_stac.py
y de sentinel2_cordoba_2018_2019: <BASE>/<TILE>/<AAAAMMDD>/<BANDA>.tif +
<BASE>/productos.json).

Mismo criterio que 02b_composites_2019_2020.py (una escena real por mes, no
mediana -- asi se generaron los composites con los que se entreno), pero
adaptado a que aca hay varias fechas por mes sin filtrar por nubes y muchas
escenas de borde de orbita (~38% nodata):
  - las fechas del mes se ordenan por nubosidad de la escena (productos.json)
    y solo se usan las de nubosidad <= --max-nube (si ninguna cumple, todas);
  - se arranca de la menos nublada y los pixeles sin dato se rellenan con la
    siguiente (mosaico "primer pixel valido"), hasta superar --umbral % de
    cobertura o quedarse sin fechas;
  - productos con processing baseline >= 04.00 traen offset +1000, salvo que
    earth-search ya lo haya sacado (earthsearch:boa_offset_applied=True, que es
    el caso de los COGs reprocesados "_1_L2A" -- verificado comparando contra
    la version "_0_" de la misma escena). Solo se resta si el flag es False,
    para quedar en la misma escala que los composites 2019-2020.

Salida: <OUT>/<TILE>_<AAAAMM>_median.tif (4 bandas B02 B03 B04 B08, uint16,
deflate; el nombre "_median" se mantiene por compatibilidad con 03c /
2_inferencia.py) y <OUT>/fechas_usadas.json con las fechas de cada mosaico.

Uso:
    python3 -m src.datos.composites_cog \\
        --base /mnt/yacy_3/prod_bis/ferreyra/sentinel2_cordoba_2018_2019 \\
        --out  /mnt/yacy_3/prod_bis/ferreyra/composites_2018_2019 \\
        --meses 201812 201901 201902 201903 201904 201905
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

BANDAS = ["B02", "B03", "B04", "B08"]


def generar_composite(base, out, tile, mes, info, umbral, max_nube):
    ruta_salida = out / f"{tile}_{mes}_median.tif"
    if ruta_salida.exists():
        return tile, mes, "ya existe", None

    fechas = [d.name for d in (base / tile).iterdir()
              if d.is_dir() and d.name[:6] == mes
              and all((d / f"{b}.tif").exists() for b in BANDAS)]
    if not fechas:
        return tile, mes, "SIN FECHAS", None
    nube = lambda f: info.get((tile, f), {}).get("cloud", 100.0)
    fechas.sort(key=nube)
    claras = [f for f in fechas if nube(f) <= max_nube]
    fechas = claras or fechas

    mosaico = None
    usadas = []
    for f in fechas:
        bandas = []
        for b in BANDAS:
            with rasterio.open(base / tile / f / f"{b}.tif") as src:
                if mosaico is None and not bandas:
                    meta = src.meta.copy()
                bandas.append(src.read(1))
        img = np.stack(bandas)
        prod = info.get((tile, f), {})
        if (prod.get("baseline") or "00.00") >= "04.00" and prod.get("boa_offset_applied") is False:
            img = np.where(img > 0, np.clip(img.astype(np.int32) - 1000, 1, None), 0).astype(np.uint16)
        valido = (img > 0).all(axis=0)
        if mosaico is None:
            mosaico = np.where(valido, img, 0).astype(np.uint16)
            cubierto = valido
        else:
            nuevo = valido & ~cubierto
            if not nuevo.any():
                continue
            mosaico[:, nuevo] = img[:, nuevo]
            cubierto = cubierto | nuevo
        usadas.append(f)
        if cubierto.mean() * 100 >= umbral:
            break

    pct = cubierto.mean() * 100
    meta.update(count=4, dtype="uint16", driver="GTiff", nodata=0,
                compress="deflate", predictor=2, tiled=True,
                blockxsize=512, blockysize=512)
    tmp = ruta_salida.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(mosaico)
        dst.descriptions = tuple(BANDAS)
    os.replace(tmp, ruta_salida)
    estado = f"OK ({'+'.join(usadas)}, {pct:.1f}% validos)"
    if pct < umbral:
        estado += f" -- ATENCION: cobertura < {umbral}%"
    return tile, mes, estado, {"fechas": usadas, "cobertura": round(pct, 2)}


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meses", nargs="+", required=True)
    ap.add_argument("--tiles", nargs="*", default=None, help="default: todos los de --base")
    ap.add_argument("--umbral", type=float, default=99.0,
                    help="%% de cobertura a partir del cual se deja de rellenar")
    ap.add_argument("--max-nube", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)

    productos = json.loads((a.base / "productos.json").read_text()) \
        if (a.base / "productos.json").exists() else []
    info = {(p["tile"], p["date"]): p for p in productos}
    tiles = a.tiles or sorted(d.name for d in a.base.iterdir() if d.is_dir())
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tiles)} tiles x {len(a.meses)} meses -> {a.out}", flush=True)

    registro_path = a.out / "fechas_usadas.json"
    registro = json.loads(registro_path.read_text()) if registro_path.exists() else {}
    fallos = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(generar_composite, a.base, a.out, t, m, info, a.umbral, a.max_nube)
                for t in tiles for m in a.meses]
        for fut in as_completed(futs):
            tile, mes, estado, reg = fut.result()
            print(f"{tile} {mes}: {estado}", flush=True)
            if reg:
                registro[f"{tile}_{mes}"] = reg
            if estado == "SIN FECHAS":
                fallos.append((tile, mes))
    registro_path.write_text(json.dumps(dict(sorted(registro.items())), indent=1))

    print(f"\nListo. Fallos: {len(fallos)} {fallos if fallos else ''}", flush=True)
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

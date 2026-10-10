"""
Composites mensuales con mascara de nubes (SCL de Sentinel-2 L2A), a partir de
descargas COG de 00_descargar_stac.py (<BASE>/<TILE>/<AAAAMMDD>/<BANDA>.tif +
<BASE>/productos.json).

Reemplaza a 02c_composites_cog.py, que elige la escena menos nublada del mes y
solo rellena los pixeles SIN DATO: si el mes no tiene ninguna escena limpia, las
nubes quedan en el composite (ej. 20HNK mar-2019: escena de 30% de nubes, con
una mancha que los modelos clasifican entera como Fondo o como Mani). Aca:
  - se enmascaran nubes, sombras y cirros segun la SCL (clases 3, 8, 9, 10,
    mas 0/1 sin dato/saturado), dilatado DILATACION px de 20 m para cubrir los
    bordes de nube que la SCL no marca;
  - ademas, B02 > UMBRAL_AZUL se toma como nube aunque la SCL diga despejado:
    en escenas muy nubladas la SCL falla (20HNK 20190311, 71% de nubes: el
    99.5% de lo que la SCL marca despejado tiene B02 > 1500; en escenas limpias
    menos del 0.1% de los pixeles despejados pasan ese umbral);
  - las escenas se usan de la menos nublada a la mas nublada (nubosidad de la
    escena, como 02c): cada pixel toma el valor de la primera en la que esta
    despejado, asi una escena muy nublada solo rellena lo que las limpias no
    cubren;
  - lo que queda sin ningun valor despejado se rellena con el valor de menor
    B02 entre todas las escenas (composite de "minimo azul", el menos nublado
    disponible), para no dejar huecos, y se reporta.

Version anterior (v1, 2/10/2026): ordenaba por cantidad de pixeles despejados
segun la SCL y confiaba solo en la SCL. Metio escenas de 71-79% de nubes por
delante de escenas limpias de media orbita (20HNK mar-2019).

Dos etapas, para bajar las bandas solo de las escenas que se usan:
  seleccionar  con la SCL ya bajada de todos los candidatos, recorre las escenas
               de la menos a la mas nublada y se queda con las que aportan al
               menos --min-aporte % de pixeles despejados nuevos (segun la
               SCL), hasta --umbral % despejado o --max-escenas. Escribe
               seleccion_scl.json.
  componer     arma los composites con las escenas elegidas.

Salida: <OUT>/<TILE>_<AAAAMM>_median.tif (4 bandas B02 B03 B04 B08, uint16,
mismo formato y escala que 02c -- el nombre "_median" se mantiene por
compatibilidad con 03c / 2_inferencia.py) y <OUT>/fechas_usadas.json con las
fechas, % despejado y % con dato de cada composite.

Uso:
    python3 -m src.datos.descargar_stac buscar --inicio 2018-12-01 --fin 2019-05-31 --out <BASE> --tiles-file tiles.txt --max-nodata 100
    python3 -m src.datos.descargar_stac descargar --out <BASE> --bandas SCL
    python3 -m src.datos.composites_scl seleccionar --base <BASE> --meses 201812 ... 201905
    python3 -m src.datos.descargar_stac descargar --out <BASE> --seleccion <BASE>/seleccion_scl.json
    python3 -m src.datos.composites_scl componer --base <BASE> --out <COMPOSITES>
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation

BANDAS = ["B02", "B03", "B04", "B08"]
# SCL: 0 sin dato, 1 saturado/defectuoso, 3 sombra de nube, 8/9 nube media/alta,
# 10 cirro. El resto (2 zona oscura, 4 vegetacion, 5 suelo, 6 agua, 7 sin
# clasificar, 11 nieve) se considera despejado.
SCL_MALO = np.zeros(256, dtype=bool)
SCL_MALO[[0, 1, 3, 8, 9, 10]] = True
SCL_NUBE = np.zeros(256, dtype=bool)
SCL_NUBE[[3, 8, 9, 10]] = True
DILATACION = 4   # px de 20 m (~80 m)
UMBRAL_AZUL = 1500   # B02 (reflectancia x 10000) por encima del cual se toma como nube


def despejado_20m(ruta_scl):
    """Mascara de pixeles despejados a 20 m (resolucion nativa de la SCL)."""
    with rasterio.open(ruta_scl) as s:
        scl = s.read(1)
    nube = binary_dilation(SCL_NUBE[scl], iterations=DILATACION) if DILATACION else SCL_NUBE[scl]
    return ~SCL_MALO[scl] & ~nube


def candidatos(base, tile, mes, info, con_bandas):
    d = base / tile
    if not d.exists():
        return []
    fechas = [f.name for f in d.iterdir() if f.is_dir() and f.name[:6] == mes
              and (f / "SCL.tif").exists() and (tile, f.name) in info
              and (not con_bandas or all((f / f"{b}.tif").exists() for b in BANDAS))]
    return sorted(fechas, key=lambda f: info[(tile, f)].get("cloud", 100.0))


def seleccionar_tile_mes(base, tile, mes, info, umbral, max_escenas, min_aporte):
    fechas = candidatos(base, tile, mes, info, con_bandas=False)
    if not fechas:
        return tile, mes, None
    cubierto = None
    elegidas = []
    for f in fechas:   # ya ordenadas de menor a mayor nubosidad de escena
        m = despejado_20m(base / tile / f / "SCL.tif")
        if cubierto is None:
            cubierto = np.zeros_like(m)
        if (m & ~cubierto).mean() * 100 < min_aporte and elegidas:
            continue
        elegidas.append(f)
        cubierto |= m
        if cubierto.mean() * 100 >= umbral or len(elegidas) >= max_escenas:
            break
    return tile, mes, {"fechas": elegidas, "despejado_previsto": round(float(cubierto.mean() * 100), 2),
                       "candidatas": len(fechas)}


def componer_tile_mes(base, out, tile, mes, fechas, info):
    ruta_salida = out / f"{tile}_{mes}_median.tif"
    if ruta_salida.exists():
        return tile, mes, "ya existe", None
    faltan = [f for f in fechas if not all((base / tile / f / f"{b}.tif").exists() for b in BANDAS + ["SCL"])]
    if faltan:
        return tile, mes, f"FALTAN ESCENAS {faltan}", None

    # mismo orden que la seleccion: de menor a mayor nubosidad de escena
    fechas = sorted(fechas, key=lambda f: info.get((tile, f), {}).get("cloud", 100.0))
    mosaico = respaldo = None
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
        # SCL 20 m -> grilla 10 m (cada pixel de 20 m son 2x2 de 10 m)
        claro = np.repeat(np.repeat(despejado_20m(base / tile / f / "SCL.tif"), 2, axis=0), 2, axis=1)
        claro = claro[:img.shape[1], :img.shape[2]] & valido & (img[0] <= UMBRAL_AZUL)
        if mosaico is None:
            mosaico = np.zeros_like(img)
            respaldo = np.zeros_like(img)
            azul_min = np.full(valido.shape, np.iinfo(np.uint16).max, dtype=np.uint16)
            con_claro = np.zeros(valido.shape, dtype=bool)
        nuevo = claro & ~con_claro
        mosaico[:, nuevo] = img[:, nuevo]
        con_claro |= nuevo
        # respaldo "minimo azul": por pixel, el valor valido de menor B02
        mejor = valido & (img[0] < azul_min)
        respaldo[:, mejor] = img[:, mejor]
        azul_min[mejor] = img[0][mejor]
    con_valido = azul_min < np.iinfo(np.uint16).max

    hueco = ~con_claro & con_valido
    mosaico[:, hueco] = respaldo[:, hueco]
    pct_claro, pct_valido = con_claro.mean() * 100, con_valido.mean() * 100
    meta.update(count=4, dtype="uint16", driver="GTiff", nodata=0,
                compress="deflate", predictor=2, tiled=True,
                blockxsize=512, blockysize=512)
    tmp = ruta_salida.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(mosaico)
        dst.descriptions = tuple(BANDAS)
    os.replace(tmp, ruta_salida)
    estado = f"OK ({'+'.join(fechas)}, {pct_claro:.1f}% despejado, {pct_valido:.1f}% con dato)"
    if pct_claro < 95:
        estado += " -- ATENCION: menos de 95% despejado"
    return tile, mes, estado, {"fechas": fechas, "despejado": round(float(pct_claro), 2),
                               "cobertura": round(float(pct_valido), 2)}


def cargar_info(base):
    productos = json.loads((base / "productos.json").read_text())
    return {(p["tile"], p["date"]): p for p in productos}


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["seleccionar", "componer"])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="dir de composites (componer)")
    ap.add_argument("--meses", nargs="+", default=None, help="AAAAMM (seleccionar)")
    ap.add_argument("--tiles", nargs="*", default=None, help="default: todos los de --base")
    ap.add_argument("--seleccion", type=Path, default=None, help="default: <BASE>/seleccion_scl.json")
    ap.add_argument("--umbral", type=float, default=99.0, help="%% despejado para dejar de sumar escenas")
    ap.add_argument("--max-escenas", type=int, default=8)
    ap.add_argument("--min-aporte", type=float, default=0.2, help="%% minimo de pixeles despejados nuevos")
    ap.add_argument("--workers", type=int, default=8)
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)

    info = cargar_info(a.base)
    ruta_sel = a.seleccion or a.base / "seleccion_scl.json"
    tiles = a.tiles or sorted(d.name for d in a.base.iterdir() if d.is_dir())

    if a.modo == "seleccionar":
        sel = json.loads(ruta_sel.read_text()) if ruta_sel.exists() else {}
        fallos = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(seleccionar_tile_mes, a.base, t, m, info, a.umbral, a.max_escenas, a.min_aporte)
                    for t in tiles for m in a.meses if f"{t}_{m}" not in sel]
            for fut in as_completed(futs):
                t, m, r = fut.result()
                if r is None:
                    fallos.append(f"{t}_{m}")
                    print(f"{t} {m}: SIN CANDIDATAS con SCL", flush=True)
                    continue
                sel[f"{t}_{m}"] = r
                print(f"{t} {m}: {len(r['fechas'])} de {r['candidatas']} escenas -> "
                      f"{r['despejado_previsto']:.1f}% despejado ({'+'.join(r['fechas'])})", flush=True)
        ruta_sel.write_text(json.dumps(dict(sorted(sel.items())), indent=1))
        n = sum(len(v["fechas"]) for v in sel.values())
        bajos = sorted((v["despejado_previsto"], k) for k, v in sel.items() if v["despejado_previsto"] < 95)
        print(f"\n{len(sel)} tile-mes, {n} escenas -> {ruta_sel}")
        print(f"tile-mes con menos de 95% despejado: {bajos if bajos else 'ninguno'}")
        if fallos:
            raise SystemExit(f"Sin candidatas: {fallos}")

    else:
        if a.out is None:
            raise SystemExit("componer necesita --out")
        sel = json.loads(ruta_sel.read_text())
        a.out.mkdir(parents=True, exist_ok=True)
        registro_path = a.out / "fechas_usadas.json"
        registro = json.loads(registro_path.read_text()) if registro_path.exists() else {}
        fallos = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(componer_tile_mes, a.base, a.out, k[:5], k[6:], v["fechas"], info)
                    for k, v in sel.items() if k[:5] in tiles]
            for fut in as_completed(futs):
                t, m, estado, reg = fut.result()
                print(f"{t} {m}: {estado}", flush=True)
                if reg:
                    registro[f"{t}_{m}"] = reg
                if estado.startswith("FALTAN"):
                    fallos.append((t, m))
        registro_path.write_text(json.dumps(dict(sorted(registro.items())), indent=1))
        print(f"\nListo. Fallos: {len(fallos)} {fallos if fallos else ''}", flush=True)
        if fallos:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

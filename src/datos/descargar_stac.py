#!/usr/bin/env python3
"""
Descarga Sentinel-2 L2A (COGs de earth-search / AWS, bandas B02 B03 B04 B08)
para una lista de tiles y un rango de fechas. Generaliza
/mnt/yacy_3/prod_bis/ferreyra/descargar_2018_2019.py (con el que se bajo
sentinel2_cordoba_2018_2019) para poder bajar otras ventanas con el mismo
formato de salida:

    <OUT>/<TILE>/<AAAAMMDD>/<BANDA>.tif   +   <OUT>/productos.json

productos.json guarda, por (tile, fecha), nubosidad y % nodata de la escena,
processing baseline y earthsearch:boa_offset_applied (02c_composites_cog.py
lo usa para elegir fechas y para sacar el offset +1000 solo si earth-search
no lo saco ya).

Uso:
    python3 -m src.datos.descargar_stac buscar    --inicio 2020-05-01 --fin 2020-05-31 --out <DIR> --tiles-file tiles.txt
    python3 -m src.datos.descargar_stac descargar --out <DIR>
    python3 -m src.datos.descargar_stac verificar --out <DIR>

Cada producto guarda tambien la URL de la SCL (Scene Classification de L2A,
20 m), que usa 02d_composites_scl.py para enmascarar nubes. Flujo en dos
etapas (ver 02d): bajar solo la SCL de todos los candidatos, elegir escenas con
02d seleccionar, y bajar las bandas solo de las elegidas:
    python3 -m src.datos.descargar_stac buscar    ... --max-nodata 100
    python3 -m src.datos.descargar_stac descargar --out <DIR> --bandas SCL
    python3 -m src.datos.composites_scl seleccionar --base <DIR> --meses ...
    python3 -m src.datos.descargar_stac descargar --out <DIR> --seleccion <DIR>/seleccion_scl.json

--max-nodata (default 50) descarta escenas con mas de ese % sin dato. Con 50
quedan afuera las escenas de borde de orbita (~53% sin dato en 20HNK), que en
meses nublados pueden ser las unicas limpias de esa parte del tile.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BBOX = [-66.93, -35.33, -60.69, -29.79]   # Cordoba
MAX_CLOUD = 80
MAX_NODATA = 50
BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
SCL = {"scl": "SCL"}
N_WORKERS = 6
STAC_URL = "https://earth-search.aws.element84.com/v1/search"


def buscar(args):
    tiles = [t.strip().upper().lstrip("T") for t in Path(args.tiles_file).read_text().split() if t.strip()]
    print(f"{len(tiles)} tiles: {', '.join(tiles)}", flush=True)
    body = {"collections": ["sentinel-2-l2a"],
            "datetime": f"{args.inicio}T00:00:00Z/{args.fin}T23:59:59Z",
            "bbox": BBOX, "limit": 100}
    mejores = {}
    page = 1
    while True:
        for intento in range(5):
            try:
                r = requests.post(STAC_URL, json=body, timeout=60)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"  pagina {page} intento {intento+1} fallo: {e}", flush=True)
                time.sleep(10)
        else:
            sys.exit("No se pudo consultar STAC. Revisar red/proxy.")
        data = r.json()
        for item in data.get("features", []):
            p = item["properties"]
            tile = p.get("grid:code", "").replace("MGRS-", "")
            if tile not in tiles:
                continue
            if p.get("eo:cloud_cover", 100) > MAX_CLOUD:
                continue
            nodata = p.get("s2:nodata_pixel_percentage", 100)
            if nodata > args.max_nodata:
                continue
            # algunos items viejos solo traen hrefs s3:// (jp2 sin COG): no se pueden bajar por https
            href = lambda a: item["assets"].get(a, {}).get("href", "")
            bands = {n: href(a) for a, n in BANDS.items() if href(a).startswith("https")}
            if len(bands) != 4:
                continue
            bands.update({n: href(a) for a, n in SCL.items() if href(a).startswith("https")})
            fecha = p["datetime"][:10].replace("-", "")
            k = (tile, fecha)
            if k not in mejores or nodata < mejores[k]["nodata"]:
                mejores[k] = {"tile": tile, "date": fecha, "nodata": nodata,
                              "cloud": p.get("eo:cloud_cover"),
                              "baseline": p.get("s2:processing_baseline"),
                              "boa_offset_applied": p.get("earthsearch:boa_offset_applied"),
                              "bands": bands}
        print(f"Pagina {page}: {len(mejores)} productos utiles", flush=True)
        nxt = next((l for l in data.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or "body" not in nxt:
            break
        body = nxt["body"]
        page += 1
    productos = sorted(mejores.values(), key=lambda x: (x["tile"], x["date"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "productos.json").write_text(json.dumps(productos, indent=1))
    print(f"\nTotal: {len(productos)} productos -> {out / 'productos.json'}")
    for t in tiles:
        print(f"  {t}: {sum(1 for p in productos if p['tile'] == t)}")
    faltan = [t for t in tiles if not any(p["tile"] == t for p in productos)]
    if faltan:
        print(f"ATENCION: sin productos para {faltan}")


def bajar_archivo(url, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".part")
    ultimo = None
    for intento in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                esperado = int(r.headers.get("Content-Length", 0))
                if destino.exists() and esperado and destino.stat().st_size == esperado:
                    return "ya estaba"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            if esperado and tmp.stat().st_size != esperado:
                raise IOError(f"tamano {tmp.stat().st_size} != {esperado}")
            tmp.rename(destino)
            return "ok"
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            ultimo = e
            time.sleep(15 * (intento + 1))
    return f"ERROR: {ultimo}"


def archivos(args):
    """(producto, banda, url) a bajar/verificar segun --bandas y --seleccion."""
    out = Path(args.out)
    cache = out / "productos.json"
    if not cache.exists():
        sys.exit("Primero corre: buscar")
    productos = json.loads(cache.read_text())
    if args.seleccion:
        sel = json.loads(Path(args.seleccion).read_text())
        elegidas = {(k[:5], f) for k, v in sel.items() for f in v["fechas"]}
        productos = [p for p in productos if (p["tile"], p["date"]) in elegidas]
    bandas = args.bandas or list(BANDS.values()) + list(SCL.values())
    return [(p, b, p["bands"][b]) for p in productos for b in bandas if b in p["bands"]]


def descargar(args):
    out = Path(args.out)
    lista = archivos(args)
    tareas = []
    for p, banda, url in lista:
        d = out / p["tile"] / p["date"] / f"{banda}.tif"
        if d.exists() and d.stat().st_size > 0:
            continue
        tareas.append((url, d))
    total = len(lista)
    print(f"{total} archivos en total, {len(tareas)} pendientes", flush=True)
    hechos, errores = total - len(tareas), 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(bajar_archivo, u, d): d for u, d in tareas}
        for fut in as_completed(futs):
            d = futs[fut]
            res = fut.result()
            hechos += 1
            if res.startswith("ERROR"):
                errores += 1
            print(f"[{hechos}/{total}] {d.relative_to(out)} {res}", flush=True)
    print(f"\nFin. Errores: {errores}. Si hay errores, volve a correr 'descargar'.")
    if errores:
        sys.exit(1)


def verificar(args):
    import rasterio
    out = Path(args.out)
    lista = archivos(args)
    malos, ok = [], 0
    for p, banda, _ in lista:
        f = out / p["tile"] / p["date"] / f"{banda}.tif"
        if not f.exists():
            malos.append((f, "falta")); continue
        if f.stat().st_size == 0:
            malos.append((f, "0 bytes")); continue
        try:
            with rasterio.open(f) as src:
                src.read(1, window=((0, 64), (0, 64)))
        except Exception as e:
            malos.append((f, f"no abre: {e}")); continue
        ok += 1
    print(f"OK: {ok} / {len(lista)}")
    print(f"Problemas: {len(malos)}")
    for f, m in malos[:50]:
        print(f"  {f.relative_to(out)} -> {m}")
    if malos:
        sys.exit(1)


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["buscar", "descargar", "verificar"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--inicio", help="AAAA-MM-DD (buscar)")
    ap.add_argument("--fin", help="AAAA-MM-DD (buscar)")
    ap.add_argument("--tiles-file", help="txt con un tile por linea (buscar)")
    ap.add_argument("--max-nodata", type=float, default=MAX_NODATA,
                    help="%% maximo de pixeles sin dato por escena (buscar)")
    ap.add_argument("--bandas", nargs="*", default=None,
                    help="solo estas bandas, ej. SCL (descargar/verificar; default: B02 B03 B04 B08 SCL)")
    ap.add_argument("--seleccion", default=None,
                    help="json de 02d_composites_scl.py seleccionar: solo esas escenas (descargar/verificar)")
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)
    {"buscar": buscar, "descargar": descargar, "verificar": verificar}[a.modo](a)


if __name__ == "__main__":
    main()

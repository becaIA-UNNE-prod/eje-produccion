"""Lectura completa por bloques, con cache invalidada por cambios del archivo."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import rasterio


def firma(ruta):
    p = Path(ruta).resolve()
    s = p.stat()
    return {'ruta': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'ctime_ns': s.st_ctime_ns, 'inode': s.st_ino}


def verificar_raster(ruta):
    antes = firma(ruta)
    try:
        with rasterio.Env(GDAL_CACHEMAX=64 * 1024**2):
            with rasterio.open(ruta) as src:
                bloques = 0
                for _, ventana in src.block_windows(1):
                    # Todas las bandas: abrir el header no prueba el stream TIFF.
                    src.read(window=ventana)
                    bloques += 1
    except Exception as exc:
        raise ValueError(f'{ruta}: lectura de bloques fallida: {exc}') from exc
    if firma(ruta) != antes:
        raise ValueError(f'{ruta}: cambio durante la lectura')
    return {'firma': antes, 'bloques': bloques}


def guardar_cache(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(datos, indent=2)); tmp.replace(ruta)


def auditar_rasters(rutas, cache_path, workers=2):
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    pendientes, fallos = [], {}
    for p in map(Path, rutas):
        try:
            f = firma(p)
            if cache.get(str(p), {}).get('firma') != f:
                pendientes.append(p)
        except Exception as exc:
            fallos[str(p)] = str(exc)
    print(f'Integridad: {len(rutas)} rasters; {len(pendientes)} pendientes de lectura completa', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(verificar_raster, p): p for p in pendientes}
        for n, fut in enumerate(as_completed(futuros), 1):
            p = futuros[fut]
            try:
                cache[str(p)] = fut.result()
            except Exception as exc:
                cache.pop(str(p), None)
                fallos[str(p)] = str(exc)
                print(f'CORRUPTO: {p}: {exc}', flush=True)
            if n % 10 == 0 or n == len(pendientes):
                guardar_cache(cache_path, cache)
                print(f'Integridad: leidos {n}/{len(pendientes)}; fallos={len(fallos)}', flush=True)
    guardar_cache(cache_path, cache)
    return fallos

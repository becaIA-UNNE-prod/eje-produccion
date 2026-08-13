"""
Genera composites mensuales para la campana 2019-2020 a partir de imagenes
Sentinel-2 L2A descargadas.

Diferencias clave respecto a versiones anteriores:
  - Verifica cobertura real de pixeles validos (>=80%) antes de usar una fecha,
    descartando escenas de borde de orbita que reportan poca nubosidad
    justamente porque tienen pocos pixeles.
  - Usa lista fija de meses: si falta alguno, el tile no se genera parcial.
  - Escribe a archivo temporal y renombra al final, para evitar composites
    corruptos si el proceso se interrumpe.

Uso:
    python3 src/02b_composites_2019_2020.py
"""
import numpy as np
import rasterio
from pathlib import Path

BASE_S2 = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2019_2020")
DIR_OUT = Path("/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020")

BANDAS = ["B02", "B03", "B04", "B08"]
MESES = ["201910", "201911", "201912", "202001", "202002", "202003", "202004"]
UMBRAL_VALIDEZ = 80.0

TILES = ['19HGC','19HGD','19HGE','19JGF','20HKH','20HKJ','20HKK','20HLG',
         '20HLH','20HLJ','20HLK','20HMG','20HMH','20HMJ','20HMK','20HNG',
         '20HNK','20HPG','20HPH','20JML','20JNL']


def cobertura_valida(ruta, umbral=UMBRAL_VALIDEZ):
    """Devuelve el % de pixeles con datos leyendo submuestreado."""
    with rasterio.open(ruta) as src:
        d = src.read(1, out_shape=(max(1, src.height // 50),
                                   max(1, src.width // 50)))
    return (d > 0).sum() / d.size * 100


def elegir_fecha(tile, mes):
    """Elige la primera fecha del mes cuya banda B04 supere el umbral."""
    for fecha_dir in sorted(d for d in (BASE_S2 / tile).iterdir()
                            if d.is_dir() and d.name[:6] == mes):
        fecha = fecha_dir.name
        archivos = [fecha_dir / f"{tile}_{fecha}_{b}.jp2" for b in BANDAS]
        if not all(a.exists() for a in archivos):
            continue
        try:
            pct = cobertura_valida(archivos[2])   # B04
            if pct >= UMBRAL_VALIDEZ:
                return fecha, archivos, pct
        except Exception:
            continue
    return None


def generar_composite(tile, mes):
    ruta_salida = DIR_OUT / f"{tile}_{mes}_median.tif"
    if ruta_salida.exists():
        print(f"{tile} {mes}: ya existe", flush=True)
        return True

    elegida = elegir_fecha(tile, mes)
    if not elegida:
        print(f"{tile} {mes}: SIN FECHA VALIDA", flush=True)
        return False

    fecha, archivos, pct = elegida
    with rasterio.open(archivos[0]) as ref:
        meta = ref.meta.copy()
        meta.update(count=4, dtype="float32", driver="GTiff")

    tmp = ruta_salida.with_suffix(".tif.tmp")
    with rasterio.open(tmp, 'w', **meta) as dst:
        for i, arch in enumerate(archivos, start=1):
            with rasterio.open(arch) as src:
                dst.write(src.read(1).astype(np.float32), i)
    tmp.rename(ruta_salida)
    print(f"{tile} {mes}: OK ({fecha}, {pct:.0f}% validos)", flush=True)
    return True


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Generando composites para {len(TILES)} tiles x {len(MESES)} meses\n", flush=True)

    fallos = []
    for tile in TILES:
        for mes in MESES:
            if not generar_composite(tile, mes):
                fallos.append((tile, mes))

    print(f"\nGeneracion completa. Fallos: {len(fallos)}", flush=True)
    if fallos:
        print(f"Detalle: {fallos}", flush=True)

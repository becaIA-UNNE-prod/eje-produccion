"""
Script de verificacion del pipeline completo.
Chequea integridad de descargas, composites, mascaras y datasets .npz
antes de lanzar el entrenamiento.

Uso:
    python3 src/0_verificar_pipeline.py --etapa todas
    python3 src/0_verificar_pipeline.py --etapa descarga
    python3 src/0_verificar_pipeline.py --etapa datasets
"""
import argparse
import numpy as np
import rasterio
from pathlib import Path

# ── Configuracion ─────────────────────────────────────────────────────────────
DIR_S2 = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2019_2020")
DIR_COMPOSITES = Path("/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020")
DIR_MASCARAS = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
DIR_DATASETS = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_mnc")

BANDAS = ["B02", "B03", "B04", "B08"]
MESES_FIJOS = ["201910","201911","201912","202001","202002","202003","202004"]
MIN_BYTES = 1_000_000  # una banda real pesa decenas de MB

REMAP_MNC = np.zeros(256, dtype=np.int64)
REMAP_MNC[10]=3; REMAP_MNC[11]=2; REMAP_MNC[12]=1; REMAP_MNC[13]=5
REMAP_MNC[14]=4; REMAP_MNC[15]=1; REMAP_MNC[16]=1; REMAP_MNC[17]=3
REMAP_MNC[18]=2; REMAP_MNC[19]=1; REMAP_MNC[21]=1; REMAP_MNC[22]=1
REMAP_MNC[255]=0

NOMBRES = {0:"NoData", 1:"Fondo", 2:"Maiz", 3:"Soja", 4:"Mani", 5:"Sorgo"}


def separador(titulo):
    print("\n" + "="*70)
    print(f"  {titulo}")
    print("="*70)


# ── ETAPA A: Verificar descargas ──────────────────────────────────────────────
def verificar_descarga():
    separador("ETAPA A — VERIFICACION DE DESCARGAS")
    completos, incompletos = [], []

    for tile_dir in sorted(d for d in DIR_S2.iterdir() if d.is_dir()):
        tile = tile_dir.name
        fechas_ok = 0
        detalles = []
        for fecha_dir in sorted(d for d in tile_dir.iterdir() if d.is_dir()):
            fecha = fecha_dir.name
            archivos = [fecha_dir / f"{tile}_{fecha}_{b}.jp2" for b in BANDAS]
            faltan = [b for b, a in zip(BANDAS, archivos)
                      if not a.exists() or a.stat().st_size < MIN_BYTES]
            if not faltan:
                fechas_ok += 1
            else:
                detalles.append(f"{fecha}: faltan {faltan}")
        if fechas_ok == 7:
            completos.append(tile)
            print(f"  OK       {tile}: 7/7 fechas completas")
        else:
            incompletos.append(tile)
            print(f"  FALTA    {tile}: {fechas_ok}/7 fechas completas")

    print(f"\n  Resumen: {len(completos)} completos, {len(incompletos)} incompletos")
    print(f"  Tiles aptos: {completos}")
    return completos


# ── ETAPA B: Verificar composites ─────────────────────────────────────────────
def verificar_composites():
    separador("ETAPA B — VERIFICACION DE COMPOSITES")
    if not DIR_COMPOSITES.exists():
        print("  El directorio de composites no existe todavia.")
        return []

    tiles = sorted(set(f.name.split('_')[0] for f in DIR_COMPOSITES.glob("*_median.tif")))
    aptos = []

    for tile in tiles:
        presentes = [m for m in MESES_FIJOS
                     if (DIR_COMPOSITES / f"{tile}_{m}_median.tif").exists()]
        n_canales = len(presentes) * 4
        if len(presentes) == 7:
            aptos.append(tile)
            print(f"  OK       {tile}: 7/7 meses -> {n_canales} canales")
        else:
            faltan = [m for m in MESES_FIJOS if m not in presentes]
            print(f"  DESCARTE {tile}: {len(presentes)}/7 meses (faltan {faltan})")

    print(f"\n  Resumen: {len(aptos)} tiles con los 7 meses completos")
    print(f"  Tiles aptos: {aptos}")
    return aptos


# ── ETAPA C: Verificar mascaras y distribucion de clases ──────────────────────
def verificar_mascaras():
    separador("ETAPA C — VERIFICACION DE MASCARAS MNC Y DISTRIBUCION DE CLASES")
    mascaras = sorted(DIR_MASCARAS.glob("etiqueta_mnc_*_10m.tif"))
    if not mascaras:
        print("  No hay mascaras MNC generadas todavia.")
        return []

    aptos = []
    for f in mascaras:
        tile = f.stem.replace("etiqueta_mnc_", "").replace("_10m", "")
        with rasterio.open(f) as src:
            Y = REMAP_MNC[src.read(1)]
        total = Y.size
        pct = {c: (Y == c).sum() / total * 100 for c in range(6)}
        pct_nodata = pct[0]
        pct_cultivos = pct[2] + pct[3] + pct[4] + pct[5]

        dist = " | ".join(f"{NOMBRES[c]}={pct[c]:.1f}%" for c in range(6))
        if pct_nodata > 70:
            print(f"  DESCARTE {tile}: NoData={pct_nodata:.1f}% (>70%)")
            print(f"           {dist}")
        elif pct_cultivos < 5:
            print(f"  DESCARTE {tile}: cultivos={pct_cultivos:.1f}% (<5%)")
            print(f"           {dist}")
        else:
            aptos.append(tile)
            print(f"  OK       {tile}: cultivos={pct_cultivos:.1f}%")
            print(f"           {dist}")

    print(f"\n  Resumen: {len(aptos)} tiles con distribucion de clases aceptable")
    print(f"  Tiles aptos: {aptos}")
    return aptos


# ── ETAPA D: Verificar datasets .npz ──────────────────────────────────────────
def verificar_datasets():
    separador("ETAPA D — VERIFICACION DE DATASETS .NPZ")
    if not DIR_DATASETS.exists():
        print("  El directorio de datasets no existe todavia.")
        return []

    shapes, corruptos, aptos = {}, [], []
    for f in sorted(DIR_DATASETS.glob("*.npz")):
        try:
            d = np.load(f)
            sx, sy = d["X"].shape, d["Y"].shape
            shapes[f.name] = sx
            aptos.append(f.name)
            print(f"  OK       {f.name}: X={sx} Y={sy}")
        except Exception as e:
            corruptos.append(f.name)
            print(f"  CORRUPTO {f.name}: {type(e).__name__}")

    if shapes:
        canales = {s[1] for s in shapes.values()}
        print(f"\n  Canales distintos encontrados: {sorted(canales)}")
        if len(canales) == 1:
            print("  APTO PARA ENTRENAR — todos los tiles tienen los mismos canales")
        else:
            print("  NO ENTRENAR — canales inconsistentes entre tiles")
            for nombre, s in shapes.items():
                print(f"    {nombre}: {s[1]} canales")

    if corruptos:
        print(f"\n  Archivos corruptos a regenerar: {corruptos}")

    return aptos


# ── ETAPA E: Distribucion de clases en el dataset de train ────────────────────
def verificar_pesos(tiles_train):
    separador("ETAPA E — DISTRIBUCION DE CLASES Y PESOS SUGERIDOS")
    if not tiles_train:
        print("  No se especificaron tiles de train.")
        return

    conteos = np.zeros(6, dtype=np.int64)
    for tile in tiles_train:
        ruta = DIR_DATASETS / f"dataset_mnc_{tile}.npz"
        if not ruta.exists():
            print(f"  {tile}: dataset no encontrado, salteando")
            continue
        Y = np.load(ruta)["Y"]
        for c in range(6):
            conteos[c] += (Y == c).sum()

    total = conteos[1:].sum()
    if total == 0:
        print("  Sin datos validos.")
        return

    print("  Distribucion en TRAIN (sin NoData):")
    for c in range(1, 6):
        print(f"    {NOMBRES[c]:<8}: {conteos[c]:>14,} px ({conteos[c]/total*100:5.2f}%)")

    print("\n  Pesos sugeridos (raiz del inverso de frecuencia):")
    pesos = [0.0]
    for c in range(1, 6):
        p = np.sqrt(total / (5 * conteos[c])) if conteos[c] > 0 else 0.0
        pesos.append(round(float(p), 3))
    print(f"    pesos = {pesos}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--etapa', default='todas',
                        choices=['todas','descarga','composites','mascaras','datasets','pesos'])
    parser.add_argument('--tiles_train', nargs='*', default=[],
                        help='Tiles de train para calcular pesos')
    args = parser.parse_args()

    if args.etapa in ('todas', 'descarga'):
        verificar_descarga()
    if args.etapa in ('todas', 'composites'):
        verificar_composites()
    if args.etapa in ('todas', 'mascaras'):
        verificar_mascaras()
    if args.etapa in ('todas', 'datasets'):
        verificar_datasets()
    if args.etapa in ('todas', 'pesos') and args.tiles_train:
        verificar_pesos(args.tiles_train)

    print("\n" + "="*70)
    print("  VERIFICACION COMPLETA")
    print("="*70)

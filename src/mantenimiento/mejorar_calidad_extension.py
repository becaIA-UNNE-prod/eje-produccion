#!/usr/bin/env python3
"""Ampliar escenas de dos meses y promover compuestos solo si no empeoran."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from utils.integridad_raster import auditar_rasters

BASE = Path('/mnt/yacy_1/prod/ferreyra/extension_este_multicampana')
COMPUESTOS = (('2018_2019', '20HNJ_201903'), ('2020_2021', '20HNH_202103'))


def crear_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'salida', type=Path,
        help='Directorio existente de resultados y respaldos',
    )
    return parser


def ejecutar_comando(raiz_repo, *argumentos):
    subprocess.run(
        [sys.executable, '-u', *map(str, argumentos)],
        cwd=raiz_repo, check=True,
    )


def mejorar_compuesto(base, salida, campana, clave, raiz_repo=ROOT):
    """Reparar un compuesto; devolver su registro tras promoverlo."""
    fuentes = base / ('sentinel2_' + campana)
    compuestos = base / ('composites_' + campana + '_scl')
    tile, mes = clave.split('_')
    anterior = json.loads((compuestos / 'fechas_usadas.json').read_text())[clave]
    productos = json.loads((fuentes / 'productos.json').read_text())
    fechas = sorted({
        producto['date'] for producto in productos
        if producto['tile'] == tile
        and producto['date'].startswith(mes)
        and (fuentes / tile / producto['date'] / 'SCL.tif').exists()
    })
    seleccion_original = json.loads((fuentes / 'seleccion_scl.json').read_text())
    entrada = {
        **seleccion_original[clave],
        'fechas': fechas,
        'nota': ('Ampliacion por cobertura real inferior a prevista; '
                 'todas las candidatas disponibles del mes'),
    }
    seleccion = salida / f'seleccion_mejora_{campana}.json'
    seleccion.write_text(json.dumps({clave: entrada}, indent=2) + '\n')
    print(
        f"MEJORA {clave}: {len(seleccion_original[clave]['fechas'])} "
        f'-> {len(fechas)} escenas', flush=True,
    )
    ejecutar_comando(
        raiz_repo, 'src/00_descargar_stac.py', 'descargar',
        '--out', fuentes, '--seleccion', seleccion,
    )
    ejecutar_comando(
        raiz_repo, 'src/00_descargar_stac.py', 'verificar',
        '--out', fuentes, '--seleccion', seleccion,
    )
    rasters = [
        fuentes / tile / fecha / (banda + '.tif')
        for fecha in fechas for banda in ['B02', 'B03', 'B04', 'B08', 'SCL']
    ]
    fallos = auditar_rasters(
        rasters, salida / f'integridad_mejora_{campana}.json', workers=2,
    )
    if fallos:
        raise ValueError(str(fallos))
    temporales = salida / ('compuestos_mejorados_' + campana)
    ejecutar_comando(
        raiz_repo, 'src/02d_composites_scl.py', 'componer',
        '--base', fuentes, '--out', temporales, '--seleccion', seleccion,
        '--tiles', tile, '--workers', '1',
    )
    mejorado = temporales / (clave + '_median.tif')
    fallos = auditar_rasters(
        [mejorado], salida / f'integridad_compuesto_mejora_{campana}.json',
        workers=1,
    )
    if fallos:
        raise ValueError(str(fallos))
    nuevo = json.loads((temporales / 'fechas_usadas.json').read_text())[clave]
    if (nuevo['despejado'] < anterior['despejado']
            or nuevo['cobertura'] < anterior['cobertura']):
        raise ValueError(
            f'La mejora empeora cobertura: {anterior} -> {nuevo}; '
            'original preservado'
        )
    destino = compuestos / mejorado.name
    respaldo = salida / (clave + '_antes_mejora.tif')
    if not respaldo.exists():
        shutil.copy2(destino, respaldo)
    shutil.copy2(mejorado, destino.with_suffix('.partial.tif'))
    destino.with_suffix('.partial.tif').replace(destino)
    ruta_registro = compuestos / 'fechas_usadas.json'
    registro = json.loads(ruta_registro.read_text())
    registro[clave] = nuevo
    registro_temporal = ruta_registro.with_suffix('.json.tmp')
    registro_temporal.write_text(json.dumps(registro, indent=2) + '\n')
    registro_temporal.replace(ruta_registro)
    ruta_seleccion = fuentes / 'seleccion_scl.json'
    respaldo_seleccion = salida / f'seleccion_antes_mejora_{campana}.json'
    if not respaldo_seleccion.exists():
        shutil.copy2(ruta_seleccion, respaldo_seleccion)
    seleccion_original[clave] = entrada
    seleccion_temporal = ruta_seleccion.with_suffix('.json.tmp')
    seleccion_temporal.write_text(
        json.dumps(seleccion_original, indent=2) + '\n',
    )
    seleccion_temporal.replace(ruta_seleccion)
    return {'key': clave, 'antes': anterior, 'despues': nuevo}


def mejorar_calidad(base, salida, raiz_repo=ROOT):
    """Mantener el reporte acumulado independiente en cada ejecución."""
    reportes = []
    for campana, clave in COMPUESTOS:
        reporte = mejorar_compuesto(base, salida, campana, clave, raiz_repo)
        reportes.append(reporte)
        (salida / 'mejoras_calidad.json').write_text(
            json.dumps(reportes, indent=2) + '\n',
        )
        print(
            f'MEJORA VERIFICADA {clave}: '
            f'{reporte["antes"]} -> {reporte["despues"]}', flush=True,
        )
    return reportes


def main(argv=None):
    args = crear_parser().parse_args(argv)
    mejorar_calidad(BASE, args.salida)


if __name__ == '__main__':
    main()

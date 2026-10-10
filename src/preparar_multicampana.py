"""Preparacion/auditoria de datos multicampana. Nunca ejecuta entrenamiento."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multicampana import SPLIT, validar_campanas, pesos_desde_conteos
from utils.leyendas_mnc import verificar_codigos
from utils.integridad_raster import auditar_rasters

from utils.rutas import MULTICAMPANA_ROOT as Y3
from src.datos.campanas import BANDAS, MASCARAS, LOGS, campanas_preparacion
TILES = sum(SPLIT.values(), [])
CAMPANAS = campanas_preparacion()


def cargar_script(nombre):
    spec = importlib.util.spec_from_file_location('mascaras2021', ROOT/'src'/nombre)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def comprobar_grillas(campanas, exigir_2021=False):
    for nombre, comp, meses, leyenda, patron, directorio in campanas:
        for tile in TILES:
            grilla = None
            for mes in meses:
                ruta = comp/f'{tile}_{mes}_median.tif'
                with rasterio.open(ruta) as src:
                    actual = (src.crs, src.transform, src.shape)
                    if (src.count != 4 or (src.dtypes != ('uint16',)*4
                            and not (leyenda == '2020' and src.dtypes == ('float32',)*4))):
                        raise ValueError(f'{ruta}: bandas/dtype inesperados')
                    # 2019/2020 no tiene descripciones; el orden esta definido
                    # por 02b/02c (B02 B03 B04 B08) y se documenta como legacy.
                    if src.descriptions != BANDAS and not (leyenda == '2020' and src.descriptions == (None,)*4):
                        raise ValueError(f'{ruta}: orden de bandas {src.descriptions}')
                    if grilla is not None and actual != grilla:
                        raise ValueError(f'{ruta}: grilla distinta entre meses')
                    grilla = actual
            mascara = MASCARAS/patron.format(tile)
            if mascara.exists() or leyenda != '2021' or exigir_2021:
                with rasterio.open(mascara) as m:
                    if (m.crs, m.transform, m.shape) != grilla:
                        raise ValueError(f'{mascara}: grilla incompatible')
                    if m.count != 1 or m.dtypes != ('uint8',):
                        raise ValueError(f'{mascara}: formato incompatible')
        print(f'{nombre}: 17 tiles x 6 composites; grillas y bandas OK', flush=True)


def json_atomico(ruta, contenido):
    tmp = ruta.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(contenido, indent=2, ensure_ascii=False)); tmp.replace(ruta)


def comprobar_procedencia(campana, ignorar_codigo20=False):
    nombre, comp, meses, leyenda, patron, directorio = campana
    for tile in TILES:
        archivos = [directorio/f'dataset_{tile}_{s}.npy' for s in ('X','Y')]
        if any(p.exists() for p in archivos):
            meta = directorio/f'dataset_{tile}_procedencia.json'
            if not all(p.exists() for p in archivos) or not meta.exists():
                raise ValueError(f'{nombre}/{tile}: salida incompleta o sin procedencia; no se reutiliza')
            datos = json.loads(meta.read_text())
            refs = [comp/f'{tile}_{mes}_median.tif' for mes in meses]
            esperados = {'leyenda': leyenda, 'meses': meses,
                         'composites': [str(p.resolve()) for p in refs],
                         'composites_mtime_ns': [p.stat().st_mtime_ns for p in refs],
                         'mascara': str((MASCARAS/patron.format(tile)).resolve()),
                         'mascara_mtime_ns': (MASCARAS/patron.format(tile)).stat().st_mtime_ns,
                         'bandas': list(BANDAS), 'normalizacion': 10000, 'leyenda_version': 1,
                         'codigos_ignorados': [20] if ignorar_codigo20 and leyenda == '2021' else []}
            if any(datos.get(k) != v for k, v in esperados.items()):
                raise ValueError(f'{nombre}/{tile}: procedencia distinta; no se reutiliza')


def preparar(workers, ignorar_codigo20=False):
    mod = cargar_script('01d_mascaras_mnc_2020_2021.py')
    codes = mod.verificar_fuente(mod.MNC, ignorar_codigo20)
    comprobar_grillas(CAMPANAS)
    # Dos datasets nuevos de ~90 GiB, mas margen para temporales.
    if shutil.disk_usage(Y3).free < 210 * 1024**3:
        raise RuntimeError('Se requieren al menos 210 GiB libres para la preparacion')
    LOGS.mkdir(parents=True, exist_ok=True)
    rutas_composites = [d/f'{t}_{mes}_median.tif' for _, d, meses, _, _, _ in CAMPANAS
                       for t in TILES for mes in meses]
    fallos_integridad = auditar_rasters(rutas_composites, LOGS/'integridad_composites.json')
    if fallos_integridad:
        raise RuntimeError(f'Composites ilegibles: {fallos_integridad}')
    json_atomico(LOGS/'labels_2021.json', {'fuente': str(mod.MNC), 'conteos': codes,
        'informe': '/mnt/yacy_1/prod/ferreyra/MNC_INTA/2020_2021/informe_mnc_2020_2021.pdf',
        'tabla': 2, 'pagina_impresa': 7, 'codigos_ignorados': [20] if ignorar_codigo20 else [], 'verificado_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    for tile in TILES:
        mod.generar(tile, mod.MNC, CAMPANAS[2][1], MASCARAS, ignorar_codigo20)
    comprobar_grillas(CAMPANAS, exigir_2021=True)
    for campana in (CAMPANAS[2], CAMPANAS[0]):
        nombre, comp, meses, leyenda, patron, directorio = campana
        comprobar_procedencia(campana, ignorar_codigo20)
        comando = [sys.executable, '-u', str(ROOT/'src/03c_generar_dataset_cordoba_f16.py'),
                   '--tiles', *TILES, '--meses', *meses, '--leyenda', leyenda,
                   '--dir-composites', str(comp), '--dir-salida', str(directorio),
                   '--workers', str(workers), '--estricto']
        if ignorar_codigo20 and leyenda == '2021':
            comando.append('--ignorar-codigo20')
        print(f'Generando dataset {nombre}: {directorio}', flush=True)
        subprocess.run(comando, check=True, cwd=ROOT)
    campanas, informe, conteos = validar_campanas([(n, str(d)) for n, _, _, _, _, d in CAMPANAS])
    informe_final = {'estado': 'completo', 'campanas': informe, 'split': SPLIT,
                     'conteos_train': conteos.tolist(), 'pesos': pesos_desde_conteos(conteos),
                     'finalizado_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    json_atomico(LOGS/'auditoria.json', informe_final)
    json_atomico(LOGS/'campanas.json', dict(campanas))
    print(f'PREPARACION COMPLETA; pesos TRAIN: {informe_final["pesos"]}. No se entreno.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--solo-verificar', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ignorar-codigo20', action='store_true')
    a = ap.parse_args()
    if a.workers < 1:
        ap.error('--workers debe ser positivo')
    if a.solo_verificar:
        mod = cargar_script('01d_mascaras_mnc_2020_2021.py')
        mod.verificar_fuente(mod.MNC, a.ignorar_codigo20)
        comprobar_grillas(CAMPANAS)
        print('PRECHECK OK; sin cambios', flush=True)
    else:
        preparar(a.workers, a.ignorar_codigo20)

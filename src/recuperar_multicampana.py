"""Reconstruye composites corruptos, audita entradas y reanuda solo preparacion."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.integridad_raster import auditar_rasters, verificar_raster


def cargar(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


prep = cargar('prep', ROOT/'src/preparar_multicampana.py')
comp = cargar('compositor', ROOT/'src/02d_composites_scl.py')
LOGS = prep.LOGS


def reparar(ruta, respaldo):
    ruta = Path(ruta)
    tile, mes = ruta.stem.removesuffix('_median').split('_')
    campana = next(c for c in prep.CAMPANAS if c[1] == ruta.parent)
    nombre, dir_comp, meses, leyenda, patron, dir_dataset = campana
    if leyenda not in ('1819', '2021'):
        raise RuntimeError(f'{ruta}: composite legacy; se requiere reconstruccion especifica, no se improvisa')
    base = prep.Y3/('sentinel2_cordoba_2018_2019' if leyenda == '1819' else 'sentinel2_cordoba_2020_2021')
    seleccion = json.loads((base/'seleccion_scl.json').read_text())
    fechas = seleccion[f'{tile}_{mes}']['fechas']
    print(f'Reconstruyendo {ruta.name}; fechas originales={fechas}', flush=True)
    for fecha in fechas:
        for banda in comp.BANDAS + ['SCL']:
            verificar_raster(base/tile/fecha/f'{banda}.tif')
    salida = respaldo/'reconstruidos'/nombre
    salida.mkdir(parents=True, exist_ok=True)
    _, _, estado, registro = comp.componer_tile_mes(base, salida, tile, mes, fechas, comp.cargar_info(base))
    nuevo = salida/ruta.name
    if registro is None or not nuevo.exists():
        raise RuntimeError(f'No se reconstruyo {ruta}: {estado}')
    verificar_raster(nuevo)
    # El original se conserva; solo se publica un TIFF con todos los bloques legibles.
    original = respaldo/'composites_corruptos'/nombre/ruta.name
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ruta, original)
    os.replace(nuevo, ruta)
    verificar_raster(ruta)
    registro_path = dir_comp/'fechas_usadas.json'
    previo = json.loads(registro_path.read_text()) if registro_path.exists() else {}
    if registro_path.exists():
        copia = respaldo/'registros'/nombre/'fechas_usadas.json'
        copia.parent.mkdir(parents=True, exist_ok=True)
        if not copia.exists():
            shutil.copy2(registro_path, copia)
    previo[f'{tile}_{mes}'] = registro
    prep.json_atomico(registro_path, dict(sorted(previo.items())))
    # Un tile ya generado no debe seguir apuntando a composites reemplazados.
    afectados = []
    for meta in dir_dataset.glob('dataset_*_procedencia.json'):
        datos = json.loads(meta.read_text())
        if str(ruta.resolve()) in datos.get('composites', []):
            t = datos['tile']
            for archivo in [dir_dataset/f'dataset_{t}_{s}.npy' for s in ('X', 'Y')] + [meta]:
                if archivo.exists():
                    destino = respaldo/'datasets_a_regenerar'/nombre/archivo.name
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    archivo.rename(destino)
            afectados.append(t)
    resultado = {'ruta': str(ruta), 'original_conservado': str(original),
                 'fechas': fechas, 'estado': estado, 'tiles_invalidados': afectados}
    print(f'REPARADO: {resultado}', flush=True)
    return resultado


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS/'recuperacion.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No cambiar entradas si existe una preparacion activa.
        with open(LOGS/'preparacion.lock', 'a') as p_lock:
            fcntl.flock(p_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inicio = time.strftime('%Y%m%dT%H%M%S')
            respaldo = LOGS/f'recuperacion_{inicio}'
            respaldo.mkdir()
            for nombre in ('preparacion.log', 'preparacion.exitcode', 'preparacion.pid'):
                ruta = LOGS/nombre
                if ruta.exists():
                    shutil.copy2(ruta, respaldo/nombre)
            (LOGS/'recuperacion.pid').write_text(str(os.getpid())+'\n')
            reparaciones = []
            conocido = prep.Y3/'composites_2020_2021_scl/20JNL_202104_median.tif'
            try:
                verificar_raster(conocido)
            except ValueError:
                reparaciones.append(reparar(conocido, respaldo))
            rutas = [d/f'{t}_{mes}_median.tif' for _, d, meses, _, _, _ in prep.CAMPANAS
                     for t in prep.TILES for mes in meses]
            cache = LOGS/'integridad_composites.json'
            fallos = auditar_rasters(rutas, cache)
            for ruta in fallos:
                reparaciones.append(reparar(ruta, respaldo))
            if fallos:
                fallos = auditar_rasters(rutas, cache)
            if fallos:
                raise RuntimeError(f'Persisten composites corruptos: {fallos}')
            prep.json_atomico(LOGS/'recuperacion.json', {'estado': 'composites_verificados',
                'respaldo': str(respaldo), 'reparaciones': reparaciones, 'rasters_verificados': len(rutas)})
        # Liberar preparacion.lock antes de entrar al wrapper que lo adquiere.
        print('Entradas verificadas; reanudando preparacion sin entrenamiento', flush=True)
        with open(LOGS/'preparacion.log', 'w') as out:
            r = subprocess.run(['bash', 'src/preparar_multicampana.sh', '--ignorar-codigo20'],
                               cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'Preparacion fallo con codigo {r.returncode}; ver preparacion.log')
        resumen = json.loads((LOGS/'recuperacion.json').read_text())
        resumen['estado'] = 'preparacion_completa'
        prep.json_atomico(LOGS/'recuperacion.json', resumen)
        print('RECUPERACION Y PREPARACION COMPLETAS', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (LOGS/'recuperacion.exitcode').write_text('1\n')
        raise
    else:
        (LOGS/'recuperacion.exitcode').write_text('0\n')


# Validar antes de cargar dependencias o iniciar operaciones.
if __name__ == "__main__":
    from src.soporte.cli_fija import validar_entrada_fija
    validar_entrada_fija('recuperar_multicampana')

"""Reconstruye composites corruptos, audita entradas y reanuda solo preparacion."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import fcntl

ROOT = Path(__file__).resolve().parents[2]
from utils.integridad_raster import auditar_rasters, verificar_raster


from utils.corridas import lock_runtime, registrar_rutas
from utils.rutas import MULTICAMPANA_ROOT

LOGS = MULTICAMPANA_ROOT / 'logs_multicampana'


def cargar_dependencias():
    """Cargar comandos existentes solo al ejecutar, hasta su propia migración."""
    preparacion = importlib.import_module('src.datos.preparar_multicampana')
    compositor = importlib.import_module('src.datos.composites_scl')
    return preparacion, compositor


def reparar(ruta, respaldo, preparacion=None, compositor=None):
    if preparacion is None or compositor is None:
        preparacion, compositor = cargar_dependencias()
    prep, comp = preparacion, compositor
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


def main(preparacion=None, compositor=None, logs=None):
    if preparacion is None or compositor is None:
        preparacion, compositor = cargar_dependencias()
    prep = preparacion
    runtime_preparation = runtime_recovery = None
    managed = logs is None and bool(os.environ.get('CORDOBA_RUN_DIR'))
    logs = LOGS if logs is None else Path(logs)
    records = Path(os.environ['CORDOBA_INVENTARIOS_DIR']) if managed else logs
    records.mkdir(parents=True, exist_ok=True)
    if managed:
        destino = os.environ.get('CORDOBA_RECOVERY_ROOT')
        if not destino:
            raise ValueError('Indicar CORDOBA_RECOVERY_ROOT fuera del repo para respaldos pesados de reparación')
        backup_base = Path(destino).expanduser().resolve()
        from utils.corridas import ruta_externa
        ruta_externa(backup_base)
        registrar_rutas(os.environ['CORDOBA_RUN_DIR'], salidas=[('respaldos_reparacion', backup_base)])
    else:
        backup_base = logs
    backup_base.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs/'recuperacion.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No cambiar entradas si existe una preparacion activa.
        with open(logs/'preparacion.lock', 'a') as p_lock:
            fcntl.flock(p_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runtime_recovery = lock_runtime('recuperacion_multicampana') if managed else None
            runtime_preparation = lock_runtime('preparacion_multicampana') if managed else None
            inicio = time.strftime('%Y%m%dT%H%M%S')
            respaldo = backup_base/f'recuperacion_{inicio}_{os.getpid()}'
            respaldo.mkdir()
            for nombre in ('preparacion.log', 'preparacion.exitcode', 'preparacion.pid'):
                ruta = logs/nombre
                if ruta.exists():
                    shutil.copy2(ruta, respaldo/nombre)
            (records/'recuperacion.pid').write_text(str(os.getpid())+'\n')
            reparaciones = []
            conocido = prep.Y3/'composites_2020_2021_scl/20JNL_202104_median.tif'
            try:
                verificar_raster(conocido)
            except ValueError:
                reparaciones.append(reparar(conocido, respaldo, preparacion, compositor))
            rutas = [d/f'{t}_{mes}_median.tif' for _, d, meses, _, _, _ in prep.CAMPANAS
                     for t in prep.TILES for mes in meses]
            cache = records/'integridad_composites.json'
            fallos = auditar_rasters(rutas, cache)
            for ruta in fallos:
                reparaciones.append(reparar(ruta, respaldo, preparacion, compositor))
            if fallos:
                fallos = auditar_rasters(rutas, cache)
            if fallos:
                raise RuntimeError(f'Persisten composites corruptos: {fallos}')
            prep.json_atomico(records/'recuperacion.json', {'estado': 'composites_verificados',
                'respaldo': str(respaldo), 'reparaciones': reparaciones, 'rasters_verificados': len(rutas)})
        if runtime_preparation is not None:
            runtime_preparation.close()
        # Liberar preparacion.lock antes de entrar al wrapper que lo adquiere.
        print('Entradas verificadas; reanudando preparacion sin entrenamiento', flush=True)
        preparation_log = Path(os.environ['CORDOBA_LOG_DIR'])/'preparacion.log' if managed else logs/'preparacion.log'
        with open(preparation_log, 'w') as out:
            r = subprocess.run(['bash', 'scripts/preparacion/preparar_multicampana.sh', '--ignorar-codigo20'],
                               cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'Preparacion fallo con codigo {r.returncode}; ver preparacion.log')
        resumen = json.loads((records/'recuperacion.json').read_text())
        resumen['estado'] = 'preparacion_completa'
        prep.json_atomico(records/'recuperacion.json', resumen)
        print('RECUPERACION Y PREPARACION COMPLETAS', flush=True)


def ejecutar():
    """Conservar el exitcode de la entrada operativa anterior."""
    try:
        main()
    except Exception:
        (Path(os.environ.get('CORDOBA_RUN_DIR', str(LOGS))) / 'recuperacion.exitcode').write_text('1\n')
        raise
    else:
        (Path(os.environ.get('CORDOBA_RUN_DIR', str(LOGS))) / 'recuperacion.exitcode').write_text('0\n')


if __name__ == '__main__':
    ejecutar()

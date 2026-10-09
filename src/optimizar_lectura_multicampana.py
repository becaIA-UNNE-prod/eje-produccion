#!/usr/bin/env python3
"""Preparar caches verificadas y reanudar en el siguiente checkpoint completo."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cache_lectura import firma, cargar_cache
from utils.multicampana import SPLIT


def guardar(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb', buffering=0) as source:
        for data in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def copiar_verificado(source, target, reserva):
    before = firma(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reutilizar solo copias completas cuya identidad siga verificada.
    receipt = target.with_suffix(target.suffix + '.sha256.json')
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['origen'] == before and saved['destino'] == firma(target):
            return saved
    if shutil.disk_usage(target.parent).free - before['size'] < reserva:
        raise RuntimeError('Espacio insuficiente manteniendo reserva: ' + str(target))
    temp = target.with_suffix(target.suffix + '.partial')
    digest = hashlib.sha256()
    started = time.monotonic()
    with source.open('rb', buffering=0) as src, temp.open('wb', buffering=0) as dst:
        for data in iter(lambda: src.read(8 * 1024**2), b''):
            digest.update(data)
            # write() puede ser parcial con buffering=0.
            view = memoryview(data)
            while view:
                written = dst.write(view)
                if not written:
                    raise OSError('Escritura incompleta: ' + str(temp))
                view = view[written:]
        os.fsync(dst.fileno())
    if firma(source) != before or checksum(temp) != digest.hexdigest():
        raise RuntimeError('La copia no coincide o cambio el origen: ' + str(source))
    temp.replace(target)
    result = {'origen': before, 'destino': firma(target), 'sha256': digest.hexdigest()}
    guardar(receipt, result)
    print(f'COPIA OK {target} GiB={before["size"]/2**30:.2f} segundos={time.monotonic()-started:.1f}', flush=True)
    return result


def preparar(exp, manifest):
    config = json.loads((exp/'config_multicampana.json').read_text())
    roots = {'2018_2019': Path('/var/tmp/ferreyra_cordoba_ia_unne/train_cache'),
             '2019_2020': Path('/mnt/yacy_1/prod/ferreyra/cache_multicampana_train')}
    required = {}
    for name, target_root in roots.items():
        source_root = Path(config['campanas'][name]['directorio'])
        sources = [source_root/f'dataset_{tile}{suffix}' for tile in SPLIT['train']
                   for suffix in ('_X.npy', '_Y.npy')]
        required[name] = sum(source.stat().st_size for source in sources)
        target_root.mkdir(parents=True, exist_ok=True)
        reserve = (40 if name == '2018_2019' else 100) * 2**30
        existing = sum((target_root/source.name).stat().st_size for source in sources
                       if (target_root/source.name).exists())
        if shutil.disk_usage(target_root).free < required[name] - existing + reserve:
            raise RuntimeError(f'No hay espacio para {name} y reserva de {reserve/2**30:.0f} GiB')
        print(f'PLAN {name}: {required[name]/2**30:.2f} GiB en {target_root}', flush=True)
    result = {'estado': 'en_preparacion', 'rutas': {}, 'fecha': datetime.datetime.now().isoformat()}
    for name, target_root in roots.items():
        source_root = Path(config['campanas'][name]['directorio'])
        reserve = (40 if name == '2018_2019' else 100) * 2**30
        for tile in SPLIT['train']:
            source = source_root/f'dataset_{tile}'
            target = target_root/f'dataset_{tile}'
            record = {'destino': str(target), 'archivos': {}}
            for suffix in ('_X.npy', '_Y.npy'):
                record['archivos'][suffix] = copiar_verificado(Path(str(source)+suffix), Path(str(target)+suffix), reserve)
            result['rutas'][str(source)] = record
            guardar(manifest, result)
    result['estado'] = 'verificado'
    guardar(manifest, result)
    cargar_cache(manifest)


def proceso_actual(pid, exp):
    path = Path('/proc')/str(pid)
    if not path.exists():
        return False
    command = (path/'cmdline').read_bytes()
    if not command:
        return False  # proceso zombie, ya termino
    if b'1_train_cordoba.py' not in command or str(exp).encode() not in command:
        raise RuntimeError('PID reutilizado o proceso ajeno; no detener')
    return True


def detener(pid, exp):
    if not proceso_actual(pid, exp):
        return
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(path.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {pid}
    while True:
        children = {child for child, parent in processes.items() if parent in descendants}
        if children <= descendants:
            break
        descendants |= children
    for child in sorted(descendants, reverse=True):
        try:
            os.kill(child, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(60):
        if not proceso_actual(pid, exp):
            return
        time.sleep(1)
    raise RuntimeError('El entrenamiento anterior sigue activo; no relanzar')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir-exp', type=Path, required=True)
    parser.add_argument('--pid-train', type=int)
    parser.add_argument('--solo-copiar', action='store_true')
    parser.add_argument('--sin-esperar-epoca', action='store_true', help='usar ultimo checkpoint de corrida ya detenida')
    parser.add_argument('--usar-copia-pid', type=int, help='esperar copia existente sin duplicarla')
    args = parser.parse_args()
    exp = args.dir_exp.resolve()
    manifest = exp/'cache_lectura.json'
    status = exp/'optimizacion_lectura.json'
    if args.solo_copiar:
        preparar(exp, manifest)
        return
    if not args.pid_train:
        parser.error('--pid-train requerido para relanzamiento')
    guardar(status, {'estado': 'preparando_caches', 'pid_coordinador': os.getpid()})
    if args.usar_copia_pid:
        copy_proc = Path('/proc') / str(args.usar_copia_pid)
        while copy_proc.exists():
            cmd = (copy_proc/'cmdline').read_bytes()
            if not cmd:
                break
            if b'--solo-copiar' not in cmd or str(exp).encode() not in cmd:
                raise RuntimeError('PID de copia reutilizado o ajeno')
            time.sleep(2)
    else:
        subprocess.run(['ionice','-c2','-n7',sys.executable,'-u',str(Path(__file__).resolve()),
                        '--dir-exp',str(exp),'--solo-copiar'], check=True, cwd=ROOT)
    cargar_cache(manifest)
    checkpoint = exp/'ultimo_checkpoint.pth'
    baseline = checkpoint.stat().st_mtime_ns
    guardar(status, {'estado': 'recuperando_checkpoint' if args.sin_esperar_epoca else 'esperando_checkpoint', 'cache':str(manifest)})
    print('Caches listas. Usando ultimo checkpoint.' if args.sin_esperar_epoca else 'Caches listas. Esperando proximo checkpoint.', flush=True)
    deadline = time.monotonic() + 3*3600
    while not args.sin_esperar_epoca and checkpoint.stat().st_mtime_ns == baseline:
        if not proceso_actual(args.pid_train, exp):
            guardar(status, {'estado':'cache_lista_corrida_terminada','cache':str(manifest)})
            print('La corrida termino; no reiniciarla automaticamente.', flush=True)
            return
        if time.monotonic() > deadline:
            raise TimeoutError('No hubo nuevo checkpoint en tres horas; no detener corrida')
        time.sleep(2)
    import torch
    # Revisar primero una copia consistente del checkpoint atomico recien guardado.
    snapshot = exp/'checkpoint_antes_precarga_paralela.pth'
    shutil.copyfile(checkpoint, snapshot)
    state = torch.load(snapshot, map_location='cpu')
    epoch = state['epoch']
    if epoch >= 100 or state['sin_mejora'] >= 30:
        guardar(status, {'estado':'cache_lista_corrida_finalizando','epoch':epoch,'cache':str(manifest)})
        return
    detener(args.pid_train, exp)
    # Si hubo otro checkpoint entre la lectura y la senal, usar el ultimo completo.
    shutil.copyfile(checkpoint, snapshot)
    state = torch.load(snapshot, map_location='cpu')
    epoch = state['epoch']
    del state
    time.sleep(3)  # permitir liberar flock del supervisor y workers
    old_log = exp/f'train_antes_paralelo_epoch{epoch}.log'
    shutil.copyfile(exp/'train.log', old_log)
    for name in ('train.exitcode','fin.txt'):
        source = exp/name
        if source.exists():
            source.rename(exp/(name+'.antes_paralelo'))
    command_args = ['bash','src/entrenar_multicampana_unet.sh',str(exp),'--lectura-bloques',
                    '--bloque-parches','512','--bloques-pool','16','--workers','4','--prefetch','2',
                    '--anticipar-pool','--cache-lectura',str(manifest),'--reanudar',str(snapshot)]
    command = 'nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ' + shlex.join(command_args) + ' >> ' + shlex.quote(str(exp/'train.log')) + ' 2>&1 < /dev/null & echo $!'
    pid = int(subprocess.check_output(command,shell=True,cwd=ROOT,text=True).strip())
    record = {'estado':'relanzado','pid':pid,'epoch_recuperada':epoch,'experimento':str(exp),
              'cache':str(manifest),'comando':command,'fecha':datetime.datetime.now().isoformat()}
    guardar(status, record)
    guardar(exp/'lanzamiento_precarga_paralela.json', record)
    latest = Path('/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/ultimo_entrenamiento_unet.json')
    guardar(latest, record)
    time.sleep(5)
    if not (Path('/proc')/str(pid)).exists():
        raise RuntimeError('El relanzamiento termino; revisar train.log')
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',flush=True)
        raise

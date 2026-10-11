#!/usr/bin/env python3
"""Ejecuta un comando con registros livianos y destinos pesados explícitos."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rutas import RUNS_ROOT, RUNTIME_ROOT
from utils.corridas import guardar_json, ruta_externa, capacidad, registrar_rutas
from scripts.registrar_rutas_corrida import pares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nombre', required=True)
    parser.add_argument('--etapa', default='manual')
    parser.add_argument('--base', type=Path, default=RUNS_ROOT)
    parser.add_argument('--padre', type=Path, help='Corrida anterior o relacionada; no reutiliza su carpeta')
    parser.add_argument('--entrada', action='append', default=[], metavar='ROL=RUTA')
    parser.add_argument('--salida', action='append', default=[], metavar='ROL=RUTA')
    parser.add_argument('--prevision-gib', action='append', default=[], metavar='ROL=GIB',
                        help='Espacio adicional estimado, incluidos temporales, por salida')
    parser.add_argument('--reserva-gib', type=float, default=100)
    parser.add_argument('comando', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.comando[1:] if args.comando[:1] == ['--'] else args.comando
    if not command or any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in (args.nombre, args.etapa)):
        parser.error('Indicar nombre/etapa válidos y un comando después de --')
    try:
        base = ruta_externa(args.base if args.base.is_absolute() else ROOT / args.base)
        inputs, outputs = pares(args.entrada), pares(args.salida)
        if len(dict(outputs)) != len(outputs) or len(dict(inputs)) != len(inputs):
            raise ValueError('Los roles de entradas/salidas no deben repetirse')
        inputs = [(k, ruta_externa(v)) for k, v in inputs]
        outputs = [(k, ruta_externa(v)) for k, v in outputs]
        forecasts = {role: float(value) for role, value in pares(args.prevision_gib)}
        if not (0 <= args.reserva_gib < float('inf')) or any(not (0 <= value < float('inf')) for value in forecasts.values()):
            raise ValueError('Previsión y reserva deben ser números finitos no negativos')
        if not set(forecasts) <= dict(outputs).keys():
            raise ValueError('Cada previsión requiere una --salida con el mismo rol')
        disks = {}
        for role, path in outputs:
            if role in forecasts:
                info = capacidad(path)
                group = disks.setdefault(info['dispositivo'], [info, 0])
                group[1] += forecasts[role] * 2**30
        for info, expected in disks.values():
            if info['disponible_bytes'] < expected + args.reserva_gib * 2**30:
                raise ValueError('Espacio insuficiente para salidas, temporales y reserva: ' + info['filesystem'])
        if base == RUNS_ROOT.resolve() and not base.parent.is_dir():
            raise ValueError('Almacenamiento de corridas no disponible; montar el disco o indicar --base explícita')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    run_id = uuid.uuid4().hex[:12]
    run = base / f'{stamp}_{args.etapa}_{args.nombre}_{run_id}'
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('logs/workers', 'configuracion', 'inventarios'):
        (run / folder).mkdir(parents=True)
    config = ROOT / 'config/split_extension_este.json'
    def git(*items):
        return subprocess.check_output(['git', *items], cwd=ROOT)
    metadata = {
        'version': 2, 'id': run_id, 'nombre': args.nombre, 'etapa': args.etapa,
        'inicio_utc': now.isoformat(), 'cwd': str(ROOT), 'comando': command,
        'git_head': git('rev-parse', 'HEAD').decode().strip(),
        'git_status': git('status', '--porcelain=v1').decode(),
        'git_diff_sha256': hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest(),
        'split_config': str(config),
        'split_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'run_dir': str(run), 'estado': 'ejecutando',
        'padre': str(args.padre.expanduser().resolve()) if args.padre else os.environ.get('CORDOBA_RUN_DIR'),
        'entradas': {}, 'salidas': {}, 'prevision_gib': forecasts,
        'reserva_gib': args.reserva_gib,
    }
    record = run / 'run.json'
    guardar_json(record, metadata)
    for source in sorted((ROOT / 'config').glob('*.json')):
        (run / 'configuracion' / source.name).write_bytes(source.read_bytes())
    registrar_rutas(run, inputs, outputs)
    state = {'id': run_id, 'estado': 'ejecutando', 'inicio_utc': metadata['inicio_utc'], 'pid_supervisor': os.getpid()}
    guardar_json(run / 'estado.json', state)
    runtime = Path(os.environ.get('CORDOBA_RUNTIME_ROOT', RUNTIME_ROOT)).expanduser().resolve()
    ruta_externa(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CORDOBA_RUN_DIR=str(run), CORDOBA_LOG_DIR=str(run / 'logs'),
               CORDOBA_INVENTARIOS_DIR=str(run / 'inventarios'), CORDOBA_RUNTIME_ROOT=str(runtime),
               CORDOBA_PY_REGISTRO=sys.executable, CORDOBA_REGISTRO_LIVIANO='1', PYTHONUNBUFFERED='1')
    print(f'Corrida: {run}', flush=True)
    print(f'Log: {run / "logs/coordinador.log"}', flush=True)
    code = 1
    child = None
    def forward(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)
    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (run / 'logs/coordinador.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['pid_comando'] = child.pid
            guardar_json(run / 'estado.json', state)
            code = child.wait()
    except OSError as error:
        state['error'] = str(error)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        code = code if code >= 0 else 128 - code
        state.update(estado='finalizado' if code == 0 else 'fallo', exitcode=code,
                     fin_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        guardar_json(run / 'estado.json', state)
        # Releer: los shells registran sus destinos efectivos durante la ejecución.
        metadata = json.loads(record.read_text())
        metadata.update({key: state[key] for key in ('estado', 'exitcode', 'fin_utc')})
        if 'error' in state:
            metadata['error'] = state['error']
        guardar_json(record, metadata)
        (run / 'exitcode').write_text(str(code) + '\n')
        guardar_json(runtime / 'punteros' / f'ultima_{args.etapa}_{args.nombre}.json',
                     {'id': run_id, 'run_dir': str(run), 'estado': state['estado'], 'exitcode': code})
    return code


if __name__ == '__main__':
    sys.exit(main())

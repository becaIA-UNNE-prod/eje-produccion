#!/usr/bin/env python3
"""Ejecuta un comando en una corrida identificada, con log y procedencia."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rutas import RUNS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nombre', required=True, help='Nombre breve: letras, números, _ o -')
    parser.add_argument('--base', type=Path, default=RUNS_ROOT)
    parser.add_argument('comando', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.comando[1:] if args.comando[:1] == ['--'] else args.comando
    if not command or not re.fullmatch(r'[A-Za-z0-9_-]+', args.nombre):
        parser.error('Indicar --nombre válido y un comando después de --')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = args.base.expanduser()
    if not base.is_absolute():
        base = ROOT / base
    if base.resolve() == ROOT or ROOT in base.resolve().parents:
        parser.error('La carpeta de corridas debe estar fuera del repositorio; usar /mnt/yacy_*')
    if base == RUNS_ROOT and not base.parent.is_dir():
        parser.error('Almacenamiento de corridas no disponible; montar el disco o indicar --base explícita')
    run = base / f'{args.nombre}_{stamp}_{os.getpid()}'
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'figuras', 'predicciones', 'metricas', 'inventarios'):
        (run / folder).mkdir()
    config = ROOT / 'config/split_extension_este.json'
    def git(*items):
        return subprocess.check_output(['git', *items], cwd=ROOT)
    metadata = {
        'inicio_utc': stamp, 'cwd': str(ROOT), 'comando': command,
        'git_head': git('rev-parse', 'HEAD').decode().strip(),
        'git_status': git('status', '--porcelain=v1').decode(),
        'git_diff_sha256': hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest(),
        'split_config': str(config),
        'split_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'run_dir': str(run), 'estado': 'ejecutando',
    }
    record = run / 'run.json'
    def save():
        tmp = record.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        tmp.replace(record)
    save()
    env = dict(os.environ, CORDOBA_RUN_DIR=str(run), PYTHONUNBUFFERED='1')
    print(f'Corrida: {run}', flush=True)
    print(f'Log: {run / "logs/stdout.log"}', flush=True)
    code = 1
    try:
        with (run / 'logs/stdout.log').open('wb') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        code = result.returncode
    except OSError as error:
        metadata['error'] = str(error)
    finally:
        metadata.update(estado='finalizado' if code == 0 else 'fallo', exitcode=code,
                        fin_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        (run / 'exitcode').write_text(str(code) + '\n')
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main())

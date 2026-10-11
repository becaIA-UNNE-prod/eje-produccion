"""Registros livianos; las rutas de datos y resultados se conservan explícitas."""
import json
import os
from pathlib import Path
import shutil
from utils.rutas import ROOT, RUNTIME_ROOT


def guardar_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def ruta_externa(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('La ruta debe estar fuera del repositorio')
    return path


def capacidad(path):
    parent = Path(path)
    while not parent.exists():
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    return {'filesystem': str(parent), 'dispositivo': parent.stat().st_dev,
            'total_bytes': usage.total, 'disponible_bytes': usage.free}


def registrar_rutas(run, entradas=(), salidas=()):
    """Actualizar solo metadatos pequeños; no crear los destinos registrados."""
    record = Path(run) / 'run.json'
    if not record.is_file():
        return  # Compatibilidad con CORDOBA_RUN_DIR manual anterior.
    data = json.loads(record.read_text())
    for key, pairs in [('entradas', entradas), ('salidas', salidas)]:
        for role, value in pairs:
            path = ruta_externa(value)
            data.setdefault(key, {})[role] = str(path)
            if key == 'salidas':
                if path == Path(run).resolve() or Path(run).resolve() in path.parents:
                    raise ValueError('Las salidas pesadas requieren destinos fuera de la carpeta de registros')
                data.setdefault('capacidad_inicial', {})[role] = capacidad(path)
    guardar_json(record, data)


def lock_runtime(nombre):
    """Segundo lock, compartido con los shells; mantener también el lock histórico."""
    import fcntl
    directory = RUNTIME_ROOT / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / (nombre + '.lock')).open('a')
    fcntl.flock(stream, fcntl.LOCK_EX)
    return stream

"""Caches de TRAIN verificadas, sin alterar originales ni split."""
import json
from pathlib import Path


def firma(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino}


def cargar_cache(path):
    if path is None:
        return {}
    manifest = json.loads(Path(path).read_text())
    if manifest.get('estado') != 'verificado':
        raise ValueError('Cache incompleta o sin verificar')
    routes = {}
    for source, record in manifest['rutas'].items():
        target = record['destino']
        for suffix in ('_X.npy', '_Y.npy'):
            if firma(source + suffix) != record['archivos'][suffix]['origen']:
                raise ValueError('Origen cambiado: ' + source + suffix)
            if firma(target + suffix) != record['archivos'][suffix]['destino']:
                raise ValueError('Cache cambiada: ' + target + suffix)
        routes[source] = target
    print(f'Cache verificada: {len(routes)} pares X/Y', flush=True)
    return routes

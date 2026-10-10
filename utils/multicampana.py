"""Split, validacion estricta y pesos comunes para entrenamiento multicampana."""
from pathlib import Path
import re
import numpy as np
from src.evaluacion.metricas import metricas_confusion  # compatibilidad

# SPLIT historico para preparar/reproducir datos antiguos.
# El entrenador y evaluador vigentes cargan config/split_extension_este.json.
SPLIT = {
    'train': ['19HGC', '20HLG', '20HLH', '20HMJ', '20HMK', '20HNK', '20JML',
              '20HKH', '20HMG', '20HNG', '20HPG', '20HPH', '20JNL'],
    'val': ['20HKJ', '20HMH'],
    'test': ['20HLK', '20HLJ'],
}


def normalizar_campanas(entradas):
    campanas, nombres, rutas = [], set(), set()
    for nombre, directorio in entradas:
        if not re.fullmatch(r'\d{4}_\d{4}', nombre):
            raise ValueError(f'Campana invalida: {nombre}; usar AAAA_AAAA')
        a, b = map(int, nombre.split('_'))
        if b != a + 1:
            raise ValueError(f'Campana no consecutiva: {nombre}')
        ruta = str(Path(directorio).expanduser().resolve())
        if nombre in nombres or ruta in rutas:
            raise ValueError(f'Campana o directorio duplicado: {nombre} {ruta}')
        nombres.add(nombre); rutas.add(ruta)
        campanas.append((nombre, ruta))
    if not campanas:
        raise ValueError('Se requiere al menos una campana')
    return campanas


def contar_etiquetas(y, bloque=32):
    if y.dtype.kind not in 'iu' or not y.size:
        raise ValueError('Etiquetas vacias o no enteras')
    conteos = np.zeros(5, dtype=np.int64)
    for i in range(0, len(y), bloque):
        parte = np.asarray(y[i:i + bloque])
        if parte.min() < 0 or parte.max() > 4:
            raise ValueError('Etiquetas fuera del rango 0..4')
        conteos += np.bincount(parte.ravel(), minlength=5)
    return conteos


def pesos_desde_conteos(conteos):
    conteos = np.asarray(conteos, dtype=np.int64)
    total = int(conteos[1:].sum())
    if total == 0:
        raise ValueError('TRAIN no contiene pixeles validos')
    return [0.0] + [round(float(np.sqrt(total / (4 * n))), 3) if n else 0.0
                    for n in conteos[1:]]


def validar_dataset(directorio, split=None, canales=24, tamano=256, cache=None):
    split = split or SPLIT
    cache = cache or {}
    registros, conteos_train = {}, np.zeros(5, dtype=np.int64)
    for rol, tiles in split.items():
        registros[rol] = []
        for tile in tiles:
            base = Path(directorio) / f'dataset_{tile}'
            lectura = cache.get(str(base), str(base))
            x = np.load(lectura + '_X.npy', mmap_mode='r', allow_pickle=False)
            y = np.load(lectura + '_Y.npy', mmap_mode='r', allow_pickle=False)
            if (x.ndim != 4 or y.ndim != 3 or len(x) != len(y) or len(x) == 0
                    or x.shape[1:] != (canales, tamano, tamano)
                    or y.shape[1:] != (tamano, tamano)):
                raise ValueError(f'{base}: dimensiones incompatibles {x.shape}/{y.shape}')
            if x.dtype != np.float16 or y.dtype != np.uint8:
                raise ValueError(f'{base}: se requiere X float16 / Y uint8')
            conteos = contar_etiquetas(y)
            if conteos[1:].sum() == 0:
                raise ValueError(f'{base}: sin etiquetas validas')
            # Recorrido acotado: X nunca se materializa entero en RAM.
            for i in range(0, len(x), 16):
                if not np.isfinite(x[i:i + 16]).all():
                    raise ValueError(f'{base}: X contiene NaN/Inf')
            if rol == 'train':
                conteos_train += conteos
            registros[rol].append({'tile': tile, 'parches': len(x),
                                    'shape': list(x.shape), 'conteos': conteos.tolist()})
    return registros, conteos_train


def validar_campanas(entradas, split=None, canales=24, tamano=256, cache=None):
    campanas = normalizar_campanas(entradas)
    informe, total = {}, np.zeros(5, dtype=np.int64)
    for nombre, directorio in campanas:
        print(f'Validando {nombre}: {directorio}', flush=True)
        registros, conteos = validar_dataset(directorio, split, canales, tamano, cache=cache)
        informe[nombre] = {'directorio': directorio, 'split': registros}
        total += conteos
    return campanas, informe, total


def cargar_campanas(campanas, split=None, cache=None):
    from torch.utils.data import ConcatDataset
    from utils.tile_dataset import TileDatasetMmap
    split = split or SPLIT
    cache = cache or {}
    por_campana, conjuntos = {}, {rol: [] for rol in split}
    for nombre, directorio in campanas:
        por_campana[nombre] = {}
        for rol, tiles in split.items():
            partes = [TileDatasetMmap(cache.get(str(Path(directorio) / f'dataset_{t}'),
                                                str(Path(directorio) / f'dataset_{t}')),
                                      augment=(rol == 'train')) for t in tiles]
            ds = ConcatDataset(partes)
            por_campana[nombre][rol] = ds
            conjuntos[rol].append(ds)
    return {rol: ConcatDataset(partes) for rol, partes in conjuntos.items()}, por_campana

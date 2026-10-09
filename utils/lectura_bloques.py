"""Mezcla acotada en page cache: lectura secuencial antes del muestreo."""
import os
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from torch.utils.data import ConcatDataset, Sampler


def hojas(dataset, offset=0):
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from hojas(child, offset)
            offset += len(child)
    else:
        if dataset.idx is not None:
            raise ValueError("Lectura por bloques requiere fraccion=1")
        yield offset, dataset


def precargar(array, start, stop):
    # read() calienta el page cache sin convertir float16 a float32 ni retener
    # una segunda copia en Python. El offset incluye el header del archivo NPY.
    stride = array.strides[0]
    remaining = (stop - start) * stride
    with open(array.filename, 'rb', buffering=0) as source:
        source.seek(array.offset + start * stride)
        while remaining:
            data = source.read(min(8 * 1024**2, remaining))
            if not data:
                raise OSError('Lectura incompleta: ' + str(array.filename))
            remaining -= len(data)


class MezclaBloques(Sampler):
    """Cada parche una vez por epoca; mezcla varios bloques de tiles/campanas.

    El pool por defecto ocupa aproximadamente 24 GiB de X float16. Cambia
    el orden aleatorio global por uno aleatorio dentro de pools, conservando
    las muestras, augmentacion y split. No requiere archivos nuevos.
    """
    def __init__(self, dataset, bloque=512, bloques_pool=16, seed=42, anticipar=False):
        if bloque <= 0 or bloques_pool <= 0:
            raise ValueError('bloque y bloques_pool deben ser positivos')
        self.dataset, self.seed = dataset, seed
        self.anticipar = anticipar
        self.bloques_pool, self.epoch = bloques_pool, 0
        self.blocks = [(offset, ds, start, min(start + bloque, len(ds)))
                       for offset, ds in hojas(dataset)
                       for start in range(0, len(ds), bloque)]

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _pool(self, selected, number):
        started = time.monotonic()
        groups, ids = {}, []
        for k in selected:
            offset, ds, start, stop = self.blocks[k]
            device = os.stat(ds.X.filename).st_dev
            groups.setdefault(device, []).append((ds, start, stop))
            ids.append(np.arange(offset + start, offset + stop))

        def warm(blocks):
            # Un lector por dispositivo: no provocar seeks concurrentes en HDD.
            for ds, start, stop in blocks:
                precargar(ds.X, start, stop)
                precargar(ds.Y, start, stop)

        with ThreadPoolExecutor(max_workers=min(3, len(groups))) as readers:
            futures = [readers.submit(warm, blocks) for blocks in groups.values()]
            for future in futures:
                future.result()
        indices = np.concatenate(ids)
        print(f'IO pool epoca={self.epoch + 1} pool={number} parches={len(indices)} '
              f'precarga_s={time.monotonic()-started:.1f} discos={len(groups)}', flush=True)
        return indices

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.blocks))
        pools = [order[pos:pos + self.bloques_pool]
                 for pos in range(0, len(order), self.bloques_pool)]
        if not self.anticipar:
            for number, selected in enumerate(pools, 1):
                indices = self._pool(selected, number)
                rng.shuffle(indices)
                yield from map(int, indices)
            return
        # Dos pools como maximo: actual + siguiente (aprox. 48 GiB de X).
        # Este thread solo hace read() y nunca toca torch/CUDA ni RNG.
        with ThreadPoolExecutor(max_workers=1) as producer:
            future = producer.submit(self._pool, pools[0], 1)
            for number in range(len(pools)):
                indices = future.result()
                if number + 1 < len(pools):
                    future = producer.submit(self._pool, pools[number + 1], number + 2)
                rng.shuffle(indices)
                yield from map(int, indices)

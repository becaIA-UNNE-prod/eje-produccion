import numpy as np
import torch


class TileDatasetMmap(torch.utils.data.Dataset):
    """Parches (X, Y) de un tile guardados como ``<ruta_base>_X.npy`` /
    ``<ruta_base>_Y.npy`` y leidos bajo demanda via memory-mapping, en vez de
    cargar el tile entero en RAM.

    Con varios tiles en un ConcatDataset (entrenamiento multitile), esto es lo
    que evita agotar la RAM del proceso: cada __getitem__ trae a memoria solo
    el parche que pide ese batch puntual (el resto del archivo lo administra
    el page cache del sistema operativo, no el heap de Python). Los .npy deben
    generarse sin comprimir (ver 03_generar_datasets_npz.py /
    03b_convertir_datasets_a_mmap.py) -- un .npz comprimido no se puede
    memory-mapear, numpy siempre descomprime el array entero al leerlo.

    remap: tabla opcional (np.array) para remapear las clases de Y al vuelo.
    fraccion: si < 1.0, fija (una sola vez, por indice) un subconjunto de los
    parches del tile en vez de usarlos todos, sin copiar el resto a RAM.
    augment: si True, aplica flips y rotaciones aleatorias de 90 grados a cada
    parche. Transformaciones validas para imagenes satelitales (un lote de
    soja rotado o espejado sigue siendo soja) -- usar solo en TRAIN, nunca en
    val/test.
    """

    def __init__(self, ruta_base, remap=None, fraccion=1.0, seed=42, augment=False):
        self.X = np.load(f"{ruta_base}_X.npy", mmap_mode="r")
        self.Y = np.load(f"{ruta_base}_Y.npy", mmap_mode="r")
        if len(self.X) != len(self.Y):
            raise ValueError(
                f"{ruta_base}: X e Y con distinta cantidad de parches "
                f"({len(self.X)} vs {len(self.Y)})")
        self.remap = remap
        self.augment = augment

        if fraccion < 1.0:
            n = max(1, int(len(self.X) * fraccion))
            rng = np.random.default_rng(seed)
            self.idx = rng.choice(len(self.X), size=n, replace=False)
        else:
            self.idx = None

        print(f"  {ruta_base}: {len(self)} parches (mmap)")

    def __len__(self):
        return len(self.idx) if self.idx is not None else len(self.X)

    def __getitem__(self, i):
        j = self.idx[i] if self.idx is not None else i
        x = torch.from_numpy(np.array(self.X[j], dtype=np.float32))
        y = np.array(self.Y[j], dtype=np.int64)
        if self.remap is not None:
            y = self.remap[y]
        y = torch.from_numpy(y)

        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
                y = torch.rot90(y, k, dims=(0, 1))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
                y = torch.flip(y, dims=(1,))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(1,))
                y = torch.flip(y, dims=(0,))
            x = x.contiguous()
            y = y.contiguous()

        return x, y

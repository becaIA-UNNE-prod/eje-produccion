import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.tile_dataset import TileDatasetMmap
from utils.lectura_bloques import MezclaBloques, precargar


class LecturaBloquesTest(unittest.TestCase):
    def test_cobertura_reproducibilidad_y_batches(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            datasets = []
            for tile, length in enumerate([13, 17, 9]):
                base = Path(directory) / str(tile)
                x = np.arange(length * 4 * 4 * 4, dtype=np.float16).reshape(length, 4, 4, 4)
                y = np.ones((length, 4, 4), dtype=np.uint8)
                np.save(str(base) + '_X.npy', x)
                np.save(str(base) + '_Y.npy', y)
                datasets.append(TileDatasetMmap(base))
            ds = ConcatDataset([ConcatDataset(datasets[:2]), datasets[2]])
            sampler = MezclaBloques(ds, bloque=4, bloques_pool=3)
            first = list(sampler)
            self.assertEqual(sorted(first), list(range(len(ds))))
            self.assertEqual(first, list(sampler))
            sampler.set_epoch(1)
            second = list(sampler)
            self.assertEqual(sorted(second), sorted(first))
            self.assertNotEqual(first, second)
            for workers in (0, 2):
                loader = DataLoader(ds, batch_size=5, sampler=sampler, num_workers=workers)
                count = 0
                for x, y in loader:
                    count += len(x)
                    self.assertTrue(np.isfinite(x.numpy()).all())
                    self.assertEqual(y.shape[1:], (4, 4))
                self.assertEqual(count, len(ds))
            with patch('utils.lectura_bloques.precargar') as warm:
                list(sampler)
                self.assertEqual(warm.call_count, 2 * len(sampler.blocks))
            async_sampler = MezclaBloques(ds, bloque=4, bloques_pool=3, anticipar=True)
            self.assertEqual(first, list(async_sampler))
            with patch('utils.lectura_bloques.precargar', side_effect=OSError('fallo de disco')):
                with self.assertRaises(OSError):
                    list(async_sampler)
            ds.datasets[1].idx = np.array([2, 0])
            with self.assertRaises(ValueError):
                MezclaBloques(ds)

if __name__ == '__main__':
    unittest.main()

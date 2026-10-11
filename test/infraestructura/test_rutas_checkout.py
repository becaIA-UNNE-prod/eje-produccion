"""Defaults de máscaras desde un checkout reubicado, sin datos operativos."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RutasCheckoutTest(unittest.TestCase):
    def test_checkout_reubicado_alias_y_exports(self):
        codigo = """
import importlib, pathlib, sys
from unittest.mock import patch
import numpy, rasterio, matplotlib.pyplot
root, physical = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root))
pairs = [
    ('src.datos.generar_dataset_cordoba_f16', 'src.03c_generar_dataset_cordoba_f16', 'DIR_MASCARAS'),
    ('src.datos.generar_dataset_cordoba_2017_f16', 'src.03d_generar_dataset_cordoba_2017_f16', 'DIR_MASCARAS'),
    ('src.datos.mascaras_mnc_2018_2019', 'src.01b_mascaras_mnc_2018_2019', 'DIR_SALIDA'),
    ('src.datos.mascaras_nivel3_2017_2018', 'src.01c_mascaras_nivel3_2017_2018', 'DIR_SALIDA'),
    ('src.historico.comparar_con_1718', 'src.12_comparar_con_1718', 'DIR_MASCARAS'),
    ('src.historico.evaluar_contra_mnc', 'src.11_evaluar_contra_mnc', 'DIR_MASCARAS'),
    ('src.historico.verificar_pipeline', 'src.0_verificar_pipeline', 'DIR_MASCARAS'),
]
with patch('rasterio.open', side_effect=AssertionError('raster real')), patch('numpy.load', side_effect=AssertionError('array real')), patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir')):
    for name, wrapper, attr in pairs:
        mod = importlib.import_module(name)
        compat = importlib.import_module(wrapper)
        path = getattr(mod, attr)
        assert path == root / 'mascaras_procesadas', (name, path)
        assert path.resolve() == physical, name
        assert compat.main is mod.main, name
        assert getattr(compat, attr) == path, name
    generator = importlib.import_module(pairs[0][0])
    assert generator.crear_parser().parse_args([]).dir_mascaras == root / 'mascaras_procesadas'
    assert generator.crear_parser().parse_args(['--dir-mascaras', str(physical)]).dir_mascaras == physical
"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkout = base / 'otro_checkout'
            checkout.mkdir()
            for name in ('src', 'utils', 'config', 'model'):
                shutil.copytree(ROOT / name, checkout / name,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            physical = base / 'storage' / 'mascaras'
            physical.mkdir(parents=True)
            (checkout / 'mascaras_procesadas').symlink_to(physical, target_is_directory=True)
            cwd = base / 'otro_cwd'
            cwd.mkdir()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(base / 'mpl'))
            result = subprocess.run([sys.executable, '-B', '-c', codigo, str(checkout), str(physical)],
                                    cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(physical.iterdir()), [])
            self.assertEqual(list(cwd.iterdir()), [])

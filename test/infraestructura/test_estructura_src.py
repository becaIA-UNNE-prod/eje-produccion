"""Auditoría transversal de los imports revisados y consumidores actuales."""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class EstructuraSrcTest(unittest.TestCase):
    def test_imports_de_dominios_y_wrappers_sin_operaciones(self):
        # Los cuerpos de nivel módulo fueron revisados antes de habilitar esta auditoría.
        code = r"""
import importlib, sys, argparse, subprocess, os
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np, torch, rasterio, requests, scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
names = [str(p.relative_to(root).with_suffix('')).replace('/', '.') for p in sorted((root / 'src').rglob('*.py')) if p.name != '__init__.py']
torch_state = torch.get_rng_state().clone(); numpy_state = np.random.get_state()
benchmark = torch.backends.cudnn.benchmark
style = dict(plt.rcParams)
sys.argv = ['prueba', '--argumento-invalido']
blocked = ['argparse.ArgumentParser.parse_args', 'numpy.load', 'numpy.save', 'numpy.savez', 'numpy.savez_compressed',
           'rasterio.open', 'torch.load', 'torch.save', 'torch.manual_seed', 'numpy.random.seed',
           'torch.cuda.is_available', 'torch.cuda.empty_cache', 'subprocess.run', 'subprocess.Popen',
           'requests.sessions.Session.request', 'pathlib.Path.mkdir', 'pathlib.Path.write_text',
           'pathlib.Path.write_bytes', 'os.makedirs', 'os.kill', 'matplotlib.pyplot.subplots']
with ExitStack() as stack:
    for name in blocked:
        stack.enter_context(patch(name, side_effect=AssertionError('Operación al importar: ' + name)))
    for name in names:
        importlib.import_module(name)
assert torch.equal(torch_state, torch.get_rng_state())
actual = np.random.get_state()
assert numpy_state[0] == actual[0] and np.array_equal(numpy_state[1], actual[1]) and numpy_state[2:] == actual[2:]
assert benchmark == torch.backends.cudnn.benchmark
assert style == dict(plt.rcParams)
assert str(root / 'src') not in sys.path
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_implementaciones_no_importan_wrappers(self):
        old = {'src.' + p.stem for p in (ROOT / 'src').glob('*.py') if p.name != '__init__.py'}
        violations = []
        for folder in ['src', 'utils', 'scripts']:
            for path in (ROOT / folder).rglob('*.py'):
                if path.parent == ROOT / 'src':
                    continue  # wrappers compatibles
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in old:
                                violations.append((str(path.relative_to(ROOT)), node.lineno, alias.name))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in old:
                            violations.append((str(path.relative_to(ROOT)), node.lineno, node.module))
        self.assertEqual(violations, [])


if __name__ == '__main__':
    unittest.main()

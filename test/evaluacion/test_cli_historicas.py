"""Contratos de importación de los comandos históricos de figuras."""
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULOS = (
    "src.4_generar_metricas",
    "src.5_generar_matriz_confusion",
    "src.6_generar_comparacion_visual",
)


class TestCLIHistoricas(unittest.TestCase):
    def test_importar_no_parsea_ni_lee_datos_ni_carga_modelos(self):
        codigo = """
import importlib, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import numpy, torch, matplotlib.pyplot
sys.argv = ['comando', '--opcion-inexistente']
with patch('numpy.load', side_effect=AssertionError('lectura de datos')), \\
     patch('torch.load', side_effect=AssertionError('lectura de pesos')), \\
     patch('torch.cuda.is_available', side_effect=AssertionError('consulta GPU')), \\
     patch('matplotlib.pyplot.subplots', side_effect=AssertionError('figura')):
    for nombre in ('src.4_generar_metricas', 'src.5_generar_matriz_confusion',
                   'src.6_generar_comparacion_visual'):
        importlib.import_module(nombre)
"""
        with tempfile.TemporaryDirectory() as temporal:
            resultado = subprocess.run(
                [sys.executable, "-B", "-c", codigo, str(ROOT)],
                cwd=temporal, capture_output=True, text=True,
            )
            self.assertEqual(resultado.returncode, 0, resultado.stderr)
            self.assertEqual(list(Path(temporal).iterdir()), [])

    def test_argumentos_historicos(self):
        for nombre in MODULOS[1:]:
            modulo = importlib.import_module(nombre)
            args = modulo.crear_parser().parse_args(
                ["--exp", "/tmp/exp", "--tile", "20JML", "--dir_train", "/tmp/train"]
            )
            self.assertEqual((args.n_clases, args.prefijo, args.modelo),
                             (5, "dataset_", "unet"))
            args = modulo.crear_parser().parse_args(
                ["--exp", "/tmp/exp", "--tile", "20JML", "--dir_train", "/tmp/train",
                 "--n_clases", "6", "--prefijo", "dataset_T", "--modelo", "utae"]
            )
            self.assertEqual((args.n_clases, args.prefijo, args.modelo),
                             (6, "dataset_T", "utae"))

    def test_curvas_desde_main(self):
        import numpy as np
        modulo = importlib.import_module(MODULOS[0])
        with tempfile.TemporaryDirectory() as temporal:
            exp = Path(temporal)
            np.save(exp / "historial.npy", [[1, .5, .6, .7, .8], [2, .4, .7, .6, .9]])
            modulo.main(["--exp", temporal])
            self.assertTrue((exp / "metricas.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual({p.name for p in exp.iterdir()}, {"historial.npy", "metricas.png"})

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
    "src.7_visualizar_inferencia",
    "src.9_comparar_inferencia_etiqueta",
    "src.10_metricas_test",
)


class TestCLIHistoricas(unittest.TestCase):
    def test_importar_no_parsea_ni_lee_datos_ni_carga_modelos(self):
        codigo = """
import importlib, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import numpy, torch, rasterio, matplotlib.pyplot
sys.argv = ['comando', '--opcion-inexistente']
with (
    patch('numpy.load', side_effect=AssertionError('lectura de datos')),
    patch('rasterio.open', side_effect=AssertionError('lectura raster')),
    patch('torch.load', side_effect=AssertionError('lectura de pesos')),
    patch('torch.cuda.is_available', side_effect=AssertionError('consulta GPU')),
    patch('matplotlib.pyplot.subplots', side_effect=AssertionError('figura')),
):
    for nombre in ('src.4_generar_metricas', 'src.5_generar_matriz_confusion',
                   'src.6_generar_comparacion_visual', 'src.7_visualizar_inferencia',
                   'src.9_comparar_inferencia_etiqueta', 'src.10_metricas_test'):
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
        for nombre in MODULOS[1:3]:
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

    def test_argumentos_raster_y_tiles_historicos(self):
        visual = importlib.import_module(MODULOS[3])
        args = visual.crear_parser().parse_args(["/tmp/pred.tif"])
        self.assertEqual((args.tif, args.max_lado, args.salida), ("/tmp/pred.tif", 2048, None))
        comparacion = importlib.import_module(MODULOS[4])
        args = comparacion.crear_parser().parse_args([])
        self.assertEqual((args.tile, args.modelo, args.max_lado),
                         ("20JML", "fcrtransunet", 2048))
        metricas = importlib.import_module(MODULOS[5])
        args = metricas.crear_parser().parse_args([])
        self.assertEqual(args.tiles, ["20HLJ", "20HLK"])

    def test_metricas_repetidas_no_acumulan_y_cuentan_prediccion_cero(self):
        import contextlib
        import io
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        modulo = importlib.import_module(MODULOS[5])
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            pred = np.array([[2, 0, 3], [1, 4, 2]], dtype=np.uint8)
            etiqueta = np.array([[10, 10, 11], [12, 16, 255]], dtype=np.uint8)
            for nombre, mapa in (("prediccion_A.tif", pred), ("etiqueta_mnc_A_10m.tif", etiqueta)):
                with rasterio.open(base / nombre, "w", driver="GTiff", width=3, height=2,
                                   count=1, dtype="uint8", crs="EPSG:32720",
                                   transform=from_origin(100, 200, 10, 10)) as dst:
                    dst.write(mapa, 1)
            argv = ["--modelo", "unet", "--tiles", "A", "--dir-pred", temporal,
                    "--dir_mascaras", temporal]
            args = modulo.crear_parser().parse_args(argv)
            confusion = modulo.confusion_tile("A", args)
            self.assertEqual(confusion.sum(), 5)
            self.assertEqual(confusion[2, 0], 1)
            self.assertEqual(confusion[2, 2], 1)
            salidas = []
            for _ in range(2):
                salida = io.StringIO()
                with contextlib.redirect_stdout(salida):
                    modulo.main(argv)
                salidas.append(salida.getvalue())
            self.assertEqual(*salidas)
            self.assertIn("F1 macro:", salidas[0])

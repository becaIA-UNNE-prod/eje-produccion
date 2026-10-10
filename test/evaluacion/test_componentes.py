import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluacion.metricas import metricas_confusion


class ComponentesTest(unittest.TestCase):
    def test_imports_sin_comandos_ni_figuras(self):
        code = (
            "import sys; sys.argv = ['prueba', '--argumento-invalido']; "
            "import src.evaluacion.metricas; import src.inferencia.raster; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'postprocesar_multicampana' not in sys.modules; "
            "assert 'preparar_evaluar_extension' not in sys.modules"
        )
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)

    def test_imports_anteriores_son_las_mismas_funciones(self):
        sys.path.insert(0, str(ROOT / 'src'))
        post = importlib.import_module('postprocesar_multicampana')
        ext = importlib.import_module('preparar_evaluar_extension')
        from utils.multicampana import metricas_confusion as anterior
        from src.evaluacion.metricas import confusion_mnc, confusion_remapeada
        from src.inferencia.raster import inferir_tile, normalizar
        self.assertIs(anterior, metricas_confusion)
        self.assertIs(post.metricas_confusion, metricas_confusion)
        self.assertIs(post.inferir_tile, inferir_tile)
        self.assertIs(post.normalizar, normalizar)
        self.assertIs(post.evaluar, confusion_mnc)
        self.assertIs(ext.confusion, confusion_remapeada)

    def test_referencia_nodata_y_prediccion_nodata(self):
        cm = np.zeros((5, 5), dtype=np.int64)
        cm[0, 1] = 100  # No participa en las metricas.
        cm[2, 0] = 1    # Error sobre referencia valida.
        cm[2, 2] = 1
        result = metricas_confusion(cm)
        self.assertEqual(result['pixeles'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['f1_macro'], 2 / 3)
        self.assertEqual(result['miou'], .5)
        empty = metricas_confusion(np.zeros((5, 5), dtype=np.int64))
        self.assertEqual(empty['pixeles'], 0)
        self.assertEqual(empty['f1_macro'], 0)
        self.assertEqual(empty['miou'], 0)


if __name__ == '__main__':
    unittest.main()

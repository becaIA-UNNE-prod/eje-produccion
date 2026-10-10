import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.datos.campanas import CATALOGO, cargar_config, meses, composites, campanas_preparacion


class CampanasTest(unittest.TestCase):
    def test_dicmay_y_lista_independiente(self):
        self.assertEqual(meses('2019_2020'), ['201912', '202001', '202002', '202003', '202004', '202005'])
        modified = meses('2019_2020')
        modified.clear()
        self.assertEqual(len(meses('2019_2020')), 6)
        self.assertEqual([c[0] for c in campanas_preparacion()], CATALOGO['preparacion_train_val'])

    def test_rutas_distintas_para_extension_y_train(self):
        base = Path('/tmp/campanas_fixture')
        regular = composites('2019_2020', '20HNJ', base=base)
        extension = composites('2019_2020', '20HNJ', extension=True, base=base)
        self.assertEqual(regular[0], base / 'composites_dicmay_2019_2020/20HNJ_201912_median.tif')
        self.assertEqual(extension[0], base / 'composites_2019_2020_scl/20HNJ_201912_median.tif')

    def test_rechaza_ventana_bandas_y_rutas_incompatibles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'campanas.json'
            for kind in ['meses', 'bandas', 'ruta']:
                cfg = copy.deepcopy(CATALOGO)
                if kind == 'meses':
                    cfg['campanas']['2019_2020']['meses'][0] = '201910'
                elif kind == 'bandas':
                    cfg['bandas'].reverse()
                else:
                    cfg['campanas']['2019_2020']['fuente_mnc'] = '../fuente.tif'
                path.write_text(json.dumps(cfg))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    cargar_config(path)

    def test_evaluador_no_importa_otros_comandos(self):
        code = (
            "import sys; sys.argv = ['prueba', '--argumento-invalido']; "
            "import src.evaluar_test_actual; "
            "assert 'postprocesar_multicampana' not in sys.modules; "
            "assert 'preparar_evaluar_extension' not in sys.modules"
        )
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()

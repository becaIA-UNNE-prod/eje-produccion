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

    def test_protocolo_extension_conserva_alcance_y_no_comparte_listas(self):
        from src.datos.campanas import protocolo, NORMALIZACION
        scope = protocolo('extension_este_original')
        self.assertEqual(scope['campanas'], ['2018_2019', '2019_2020', '2020_2021', '2021_2022'])
        self.assertEqual(scope['tiles'], ['20HNJ', '20HNH'])
        self.assertEqual(scope['campanas_exclusion'], ['2018_2019', '2019_2020', '2020_2021'])
        scope['tiles'].clear()
        self.assertEqual(protocolo('extension_este_original')['tiles'], ['20HNJ', '20HNH'])
        self.assertEqual(NORMALIZACION, 10000)
        self.assertIsNone(CATALOGO['campanas']['2021_2022']['dataset'])

    def test_rechaza_catalogo_ambiguo_o_fuera_de_base(self):
        cases = [
            ('ventana', lambda c: c.update(ventana='octubre-abril')),
            ('dataset', lambda c: c['campanas']['2019_2020'].update(dataset='../otro')),
            ('patron', lambda c: c['campanas']['2019_2020'].update(patron_mascara='../{}.tif')),
            ('leyenda', lambda c: c['campanas']['2019_2020'].update(leyenda='otra')),
            ('duplicada', lambda c: c['preparacion_train_val'].append('2019_2020')),
            ('sin_dataset', lambda c: c['preparacion_train_val'].append('2021_2022')),
            ('campana', lambda c: c['protocolos']['extension_este_original']['campanas'].append('2099_2100')),
            ('tiles', lambda c: c['protocolos']['extension_este_original']['tiles'].append('20HNJ')),
            ('exclusion', lambda c: c['protocolos']['extension_este_original']['campanas_exclusion'].append('2021_2022')),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalogo.json'
            for name, change in cases:
                cfg = copy.deepcopy(CATALOGO)
                change(cfg)
                path.write_text(json.dumps(cfg))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    cargar_config(path)

    def test_evaluador_no_importa_otros_comandos(self):
        code = (
            "import sys; sys.argv = ['prueba', '--argumento-invalido']; "
            "import src.evaluacion.test_actual; import src.evaluar_test_actual; "
            "assert 'postprocesar_multicampana' not in sys.modules; "
            "assert 'preparar_evaluar_extension' not in sys.modules"
        )
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()

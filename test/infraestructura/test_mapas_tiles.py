import json
from pathlib import Path
import shutil
import tempfile
import unittest
from utils.mapas_tiles import generar_mapa
ROOT = Path(__file__).resolve().parents[2]

class MapasTilesTests(unittest.TestCase):
    def test_split_actual_y_campanas(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = generar_mapa(ROOT/'config', tmp)
            self.assertEqual(m['conteos'], dict(train=13, val=2, test=2))
            features=json.loads((Path(tmp)/'tiles.geojson').read_text())['features']
            bytile={f['properties']['tile']: f['properties'] for f in features}
            self.assertEqual({t for t,p in bytile.items() if p['rol']=='test'}, {'20HNJ','20HNH'})
            self.assertEqual(bytile['20HLK']['rol'], 'inactivo')
            self.assertEqual(len(bytile['20HNJ']['campanas_usadas']),4)
            self.assertEqual(len(bytile['20HMJ']['campanas_usadas']),3)
            self.assertNotIn('__DATOS__',(Path(tmp)/'mapa_tiles.html').read_text())

    def test_cambio_de_split_y_faltantes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'config'; shutil.copytree(ROOT/'config',config)
            path=config/'split_extension_este.json'; split=json.loads(path.read_text())
            split['train'].remove('20HMJ');split['val'].append('20HMJ');path.write_text(json.dumps(split))
            self.assertEqual(generar_mapa(config,root/'mapa')['conteos']['val'],3)
            split['test']=['DESCONOCIDO'];path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError,'Faltan contornos'):
                generar_mapa(config,root/'fallo')
            self.assertFalse((root/'fallo').exists())

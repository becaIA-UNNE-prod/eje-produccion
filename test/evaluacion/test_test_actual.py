import unittest,json,tempfile,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from utils.test_actual import cargar_config,preflight,CONFIG
class TestActualTests(unittest.TestCase):
 def test_split_activo_y_no_mezcla_particiones(self):
  cfg=cargar_config();self.assertEqual(cfg['test'],['20HNJ','20HNH']);self.assertNotIn('20HLK',sum((cfg[k] for k in ['train','val','test']),[]))
  with tempfile.TemporaryDirectory() as tmp:
   cfg['test']=[cfg['train'][0]];p=Path(tmp)/'cfg.json';p.write_text(json.dumps(cfg))
   with self.assertRaises(ValueError):cargar_config(p)
 def preparar_fixture(self, directory):
  cfg=cargar_config();cfg['preparacion_base']=str(directory)
  (directory/'protocolo.json').write_text(json.dumps({'split_original':{role:cfg[role] for role in ['train','val']}}))
  (directory/'preparacion_verificada.json').write_text(json.dumps({'tiles':cfg['test'],'campanas':cfg['campanas']}))
  (directory/'mascaras').mkdir()
  for year in cfg['campanas']:
   for tile in cfg['test']:
    for scope in ['completa','independiente']:
     (directory/'mascaras'/f'{year}_{tile}_{scope}.tif').write_bytes(b'fixture: preflight comprueba existencia, no contenido raster')
  return cfg
 def test_preflight_acepta_preparacion_fixture(self):
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp);cfg=self.preparar_fixture(directory)
   self.assertEqual(preflight(cfg),directory)
   self.assertEqual(preflight(cfg,[cfg['campanas'][0]]),directory)
 def test_preflight_rechaza_exclusiones_obsoletas(self):
  for role in ['train','val']:
   with self.subTest(role=role),tempfile.TemporaryDirectory() as tmp:
    cfg=self.preparar_fixture(Path(tmp));cfg[role]=['20HLK']
    with self.assertRaisesRegex(ValueError,'TRAIN/VAL'):preflight(cfg)
 def test_preflight_rechaza_test_o_campana_no_preparados(self):
  for modo in ['test','campana']:
   with self.subTest(modo=modo),tempfile.TemporaryDirectory() as tmp:
    cfg=self.preparar_fixture(Path(tmp))
    if modo=='test':cfg['test']=['20HLK']
    else:cfg['campanas']=['2099_2100']
    with self.assertRaisesRegex(ValueError,'TEST distinto' if modo=='test' else 'Campana sin preparacion'):preflight(cfg)
 def test_preflight_exige_ambos_criterios_y_metadatos(self):
  for missing in ['protocolo.json','preparacion_verificada.json','completa','independiente']:
   with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
    directory=Path(tmp);cfg=self.preparar_fixture(directory)
    ruta=directory/missing if missing.endswith('.json') else directory/'mascaras'/f"{cfg['campanas'][0]}_{cfg['test'][0]}_{missing}.tif"
    ruta.unlink()
    with self.assertRaises(FileNotFoundError):preflight(cfg)
 def test_comprobacion_operativa_explicita_con_fixture_y_faltante(self):
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp);cfg=self.preparar_fixture(directory);config=directory/'config.json';config.write_text(json.dumps(cfg))
   command=[sys.executable,'-B',str(ROOT/'scripts/diagnostico/verificar_preparacion_test_actual.py'),'--split-config',str(config)]
   p=subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=15)
   self.assertEqual(p.returncode,0,p.stderr);self.assertIn('PREPARACION TEST ACTUAL OK',p.stdout)
   (directory/'protocolo.json').unlink()
   p=subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=15)
   self.assertNotEqual(p.returncode,0);self.assertIn('FileNotFoundError',p.stderr)
 def test_entrenador_no_busca_arrays_del_test_antiguo_ni_nuevo(self):
  cfg=cargar_config()
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp);cfg=self.preparar_fixture(directory);config=directory/'split_fixture.json';config.write_text(json.dumps(cfg));x=np.ones((1,24,256,256),dtype=np.float16);y=np.ones((1,256,256),dtype=np.uint8)
   for tile in cfg['train']+cfg['val']:
    np.save(directory/f'dataset_{tile}_X.npy',x);np.save(directory/f'dataset_{tile}_Y.npy',y)
   p=subprocess.run([sys.executable,'-m','src.entrenamiento.entrenar_cordoba','--modelo','unet','--campana','2018_2019',tmp,'--verificar-datos','--split-config',str(config)],cwd=ROOT,capture_output=True,text=True,timeout=60)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertIn("TEST vigente: ['20HNJ', '20HNH']",p.stdout);self.assertIn('VERIFICACION OK',p.stdout)
if __name__=='__main__':unittest.main()

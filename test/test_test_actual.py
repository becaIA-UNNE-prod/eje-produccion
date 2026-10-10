import unittest,json,tempfile,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.test_actual import cargar_config,preflight,CONFIG
class TestActualTests(unittest.TestCase):
 def test_split_activo_y_no_mezcla_particiones(self):
  cfg=cargar_config();self.assertEqual(cfg['test'],['20HNJ','20HNH']);self.assertNotIn('20HLK',sum((cfg[k] for k in ['train','val','test']),[]))
  with tempfile.TemporaryDirectory() as tmp:
   cfg['test']=[cfg['train'][0]];p=Path(tmp)/'cfg.json';p.write_text(json.dumps(cfg))
   with self.assertRaises(ValueError):cargar_config(p)
 def test_preflight_rechaza_exclusiones_obsoletas(self):
  cfg=cargar_config();preflight(cfg);cfg['val']=['20HLK']
  with self.assertRaises(ValueError):preflight(cfg)
 def test_entrenador_no_busca_arrays_del_test_antiguo_ni_nuevo(self):
  cfg=cargar_config()
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp);x=np.ones((1,24,256,256),dtype=np.float16);y=np.ones((1,256,256),dtype=np.uint8)
   for tile in cfg['train']+cfg['val']:
    np.save(directory/f'dataset_{tile}_X.npy',x);np.save(directory/f'dataset_{tile}_Y.npy',y)
   p=subprocess.run([sys.executable,'src/1_train_cordoba.py','--modelo','unet','--campana','2018_2019',tmp,'--verificar-datos','--split-config',str(CONFIG)],cwd=ROOT,capture_output=True,text=True,timeout=60)
   self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertIn("TEST vigente: ['20HNJ', '20HNH']",p.stdout);self.assertIn('VERIFICACION OK',p.stdout)
if __name__=='__main__':unittest.main()

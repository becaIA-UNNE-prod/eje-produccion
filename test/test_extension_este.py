import unittest,tempfile,sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preparar_evaluar_extension import legend,check_codes,confusion
class ExtensionTests(unittest.TestCase):
 def test_codigo20_exclusivo_2021(self):
  lut,allowed,_=legend('2020_2021');check_codes([20],allowed);self.assertEqual(int(lut[20]),0)
  for year in ['2018_2019','2019_2020','2021_2022']:
   with self.assertRaises(ValueError):check_codes([20],legend(year)[1])
 def test_pred0_es_error_y_gt0_se_ignora(self):
  with tempfile.TemporaryDirectory() as tmp:
   profile=dict(driver='GTiff',width=2,height=2,count=1,dtype='uint8',crs='EPSG:32720',transform=from_origin(500000,6200000,10,10),nodata=0)
   paths=[Path(tmp)/'p.tif',Path(tmp)/'m.tif']
   for path,data in zip(paths,[np.array([[0,2],[4,1]],dtype=np.uint8),np.array([[2,2],[0,1]],dtype=np.uint8)]):
    with rasterio.open(path,'w',**profile) as dst:dst.write(data,1)
   cm=confusion(*paths);self.assertEqual(int(cm.sum()),3);self.assertEqual(cm[2,0],1);self.assertEqual(cm[2,2],1);self.assertEqual(cm[1,1],1)
 def test_solapamiento_no_excluye_toda_la_grilla(self):
  geometry={'type':'Polygon','coordinates':[[[0,0],[.9,0],[.9,2],[0,2],[0,0]]]}
  excluded=rasterize([(geometry,1)],out_shape=(2,2),transform=from_origin(0,2,1,1),all_touched=True)
  np.testing.assert_array_equal(excluded,[[1,0],[1,0]])
if __name__=='__main__':unittest.main()

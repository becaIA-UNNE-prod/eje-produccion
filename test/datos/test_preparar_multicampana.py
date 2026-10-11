"""Coordinación simulada; rasters pequeños y estado de shell solo en /tmp."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.datos import preparar_multicampana as prep
from src.datos import mascaras_mnc_2020_2021 as masks

ROOT=Path(__file__).resolve().parents[2]


class PrepararMulticampanaTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name)
        self.masc=self.base/'mascaras'; self.masc.mkdir()
        self.logs=self.base/'logs'
        self.campanas=[]
        for name,leyenda in [('2018_2019','1819'),('2019_2020','2020'),('2020_2021','2021')]:
            comp=self.base/name/'comp'; comp.mkdir(parents=True)
            datos=self.base/name/'datos'; datos.mkdir()
            patron='etiqueta_'+leyenda+'_{}.tif'
            self.campanas.append((name,comp,['202012'],leyenda,patron,datos))
            self.raster(comp/'A_202012_median.tif')
            self.raster(self.masc/patron.format('A'),mask=True)

    def raster(self,path,mask=False,**overrides):
        meta=dict(driver='GTiff',height=2,width=2,count=1 if mask else 4,
                  dtype='uint8' if mask else 'uint16',crs='EPSG:32720',
                  transform=from_origin(0,20,10,10))
        meta.update(overrides)
        with rasterio.open(path,'w',**meta) as dst:
            dst.write(np.ones((meta['count'],meta['height'],meta['width']),dtype=meta['dtype']))
            if not mask:
                dst.descriptions=prep.BANDAS

    def test_grillas_legacy_y_mascara2021_requerida(self):
        with patch.multiple(prep,TILES=['A'],MASCARAS=self.masc):
            legacy=self.campanas[1][1]/'A_202012_median.tif'
            self.raster(legacy,dtype='float32')
            with rasterio.open(legacy,'r+') as dst:
                dst.descriptions=(None,)*4
            prep.comprobar_grillas(self.campanas)
            mascara=self.masc/self.campanas[2][4].format('A')
            mascara.unlink()
            prep.comprobar_grillas(self.campanas)
            with self.assertRaises(rasterio.errors.RasterioIOError):
                prep.comprobar_grillas(self.campanas,exigir_2021=True)
            self.raster(mascara,mask=True,width=3)
            with self.assertRaisesRegex(ValueError,'grilla incompatible'):
                prep.comprobar_grillas(self.campanas)
            self.raster(mascara,mask=True)
            with rasterio.open(self.campanas[0][1]/'A_202012_median.tif','r+') as dst:
                dst.descriptions=('B04','B03','B02','B08')
            with self.assertRaisesRegex(ValueError,'orden de bandas'):
                prep.comprobar_grillas(self.campanas)

    def test_procedencia_incompleta_obsoleta_y_codigo20(self):
        campana=self.campanas[2]
        name,comp,meses,leyenda,patron,directory=campana
        with patch.multiple(prep,TILES=['A'],MASCARAS=self.masc):
            prep.comprobar_procedencia(campana,True)
            (directory/'dataset_A_X.npy').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError,'salida incompleta'):
                prep.comprobar_procedencia(campana,True)
            (directory/'dataset_A_Y.npy').write_bytes(b'fixture')
            metadata={'leyenda':leyenda,'meses':meses,
                      'composites':[str((comp/f'A_{m}_median.tif').resolve()) for m in meses],
                      'composites_mtime_ns':[(comp/f'A_{m}_median.tif').stat().st_mtime_ns for m in meses],
                      'mascara':str((self.masc/patron.format('A')).resolve()),
                      'mascara_mtime_ns':(self.masc/patron.format('A')).stat().st_mtime_ns,
                      'bandas':list(prep.BANDAS),'normalizacion':10000,'leyenda_version':1,
                      'codigos_ignorados':[20]}
            ruta=directory/'dataset_A_procedencia.json'
            ruta.write_text(json.dumps(metadata))
            before=ruta.read_bytes()
            prep.comprobar_procedencia(campana,True)
            self.assertEqual(before,ruta.read_bytes())
            with self.assertRaisesRegex(ValueError,'procedencia distinta'):
                prep.comprobar_procedencia(campana,False)
            metadata['composites_mtime_ns']=[0]
            ruta.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'procedencia distinta'):
                prep.comprobar_procedencia(campana,True)

    def ejecutar_fixture(self,free=211*1024**3,fallos=None):
        manager=Mock()
        with patch.multiple(prep,TILES=['A'],CAMPANAS=self.campanas,MASCARAS=self.masc,LOGS=self.logs,Y3=self.base), \
             patch.object(masks,'verificar_fuente',return_value={'10':1,'11':1,'16':1}) as source, \
             patch.object(masks,'generar') as generar, \
             patch.object(prep,'comprobar_grillas') as grillas, \
             patch.object(prep.shutil,'disk_usage',return_value=SimpleNamespace(free=free)), \
             patch.object(prep,'auditar_rasters',return_value=fallos or []) as audit, \
             patch.object(prep,'comprobar_procedencia') as provenance, \
             patch.object(prep.subprocess,'run') as run, \
             patch.object(prep,'validar_campanas',return_value=([('2018_2019','fixture')],{'fixture':True},np.array([0,1,1,1,1]))) as validate, \
             patch.object(prep,'pesos_desde_conteos',return_value=[0.,1.,1.,1.,1.]), \
             patch.object(prep.time,'strftime',return_value='FECHA_FIXTURE'):
            for name,mock in [('fuente',source),('grillas',grillas),('mascara',generar),('auditar',audit),('procedencia',provenance),('ejecutar',run),('validar',validate)]:
                manager.attach_mock(mock,name)
            if free<210*1024**3 or fallos:
                with self.assertRaises(RuntimeError):
                    prep.preparar(3,True)
                run.assert_not_called()
                generar.assert_not_called()
                validate.assert_not_called()
                return
            prep.preparar(3,True)
            operations=[call[0] for call in manager.mock_calls]
            self.assertEqual(operations,['fuente','grillas','auditar','mascara','grillas','procedencia','ejecutar','procedencia','ejecutar','validar'])
            grillas.assert_any_call(self.campanas,exigir_2021=True)
            commands=[call.args[0] for call in run.call_args_list]
            self.assertEqual([c[c.index('--leyenda')+1] for c in commands],['2021','1819'])
            for command in commands:
                self.assertEqual(command[:4],[sys.executable,'-u','-m','src.datos.generar_dataset_cordoba_f16'])
                self.assertIn('--estricto',command)
                self.assertEqual(command[command.index('--workers')+1],'3')
            self.assertIn('--ignorar-codigo20',commands[0])
            self.assertNotIn('--ignorar-codigo20',commands[1])
            self.assertTrue(all(call.kwargs=={'check':True,'cwd':ROOT} for call in run.call_args_list))
        report=json.loads((self.logs/'auditoria.json').read_text())
        self.assertEqual(report['estado'],'completo')
        self.assertEqual(report['split'],prep.SPLIT)
        self.assertEqual(report['conteos_train'],[0,1,1,1,1])
        self.assertEqual(report['pesos'],[0.,1.,1.,1.,1.])
        self.assertEqual(json.loads((self.logs/'labels_2021.json').read_text())['codigos_ignorados'],[20])
        self.assertFalse(list(self.logs.glob('*.tmp')))

    def test_preparacion_orden_comandos_y_auditoria(self):
        self.ejecutar_fixture()

    def test_reserva_o_corrupcion_impiden_generacion(self):
        self.ejecutar_fixture(free=209*1024**3)
        self.assertFalse(self.logs.exists())
        self.ejecutar_fixture(fallos=['corrupto.tif'])
        self.assertFalse((self.logs/'labels_2021.json').exists())

    def test_cli_solo_verificar_sin_generar(self):
        with patch.object(masks,'verificar_fuente') as source, patch.object(prep,'comprobar_grillas') as grids, patch.object(prep,'preparar',side_effect=AssertionError('No generar')), patch('pathlib.Path.mkdir',side_effect=AssertionError('No escribir')):
            prep.main(['--solo-verificar','--ignorar-codigo20'])
            source.assert_called_once_with(masks.MNC,True)
            grids.assert_called_once_with(prep.CAMPANAS)
        with self.assertRaises(SystemExit) as error:
            prep.main(['--workers','0'])
        self.assertEqual(error.exception.code,2)

    def test_imports_exports_y_ayuda_desde_otro_cwd(self):
        code='''import sys,importlib
from unittest.mock import patch
task_root=sys.argv[1]; sys.path.insert(0,task_root)
import src.datos.campanas, utils.multicampana, utils.integridad_raster
sys.argv=['prueba','--invalido']
with patch('rasterio.open',side_effect=AssertionError('raster')), patch('numpy.load',side_effect=AssertionError('arrays')), patch('subprocess.run',side_effect=AssertionError('comando')), patch('pathlib.Path.mkdir',side_effect=AssertionError('salida')), patch('argparse.ArgumentParser.parse_args',side_effect=AssertionError('parser')):
    a=importlib.import_module('src.datos.preparar_multicampana')
    b=importlib.import_module('src.preparar_multicampana')
    for name in ['preparar','comprobar_grillas','comprobar_procedencia','json_atomico','crear_parser','main']:
        assert getattr(a,name) is getattr(b,name)
    assert a.ROOT==__import__('pathlib').Path(task_root)
'''
        r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=self.base,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        for prefix,cwd in [([str(ROOT/'src/preparar_multicampana.py')],self.base),(['-m','src.datos.preparar_multicampana'],ROOT)]:
            r=subprocess.run([sys.executable,'-B',*prefix,'--help'],cwd=cwd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_shell_estado_exitcode_y_lock_en_replica_temporal(self):
        import fcntl
        # Copy only the two shell entries; relocate their status path to /tmp.
        replica=self.base/'replica'
        logs=replica/'logs'
        logs.mkdir(parents=True)
        for name in ['scripts/preparacion/preparar_multicampana.sh','src/preparar_multicampana.sh']:
            path=replica/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text((ROOT/name).read_text().replace('/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana',str(logs)))
        fake=self.base/'python_simulado'
        fake.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
if sys.argv[1]=='-c':
    (Path(os.environ['FIXTURE_LOGS'])/'preparacion.exitcode').unlink(missing_ok=True)
else:
    Path(os.environ['FIXTURE_COMANDO']).write_text(json.dumps({'cwd':str(Path.cwd()),'args':sys.argv[1:],'exitcode_previo':(Path(os.environ['FIXTURE_LOGS'])/'preparacion.exitcode').exists()}))
    raise SystemExit(int(os.environ['FIXTURE_RC']))
''')
        fake.chmod(0o755)
        command_record=self.base/'comando.json'
        env=dict(os.environ,PY=str(fake),FIXTURE_LOGS=str(logs),FIXTURE_COMANDO=str(command_record))
        for entry in ['scripts/preparacion/preparar_multicampana.sh','src/preparar_multicampana.sh']:
            for rc in [0,7]:
                env['FIXTURE_RC']=str(rc)
                (logs/'preparacion.exitcode').write_text('ANTERIOR')
                r=subprocess.run(['bash',str(replica/entry),'--solo-verificar','--workers','2'],cwd=self.base,env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,rc,r.stderr)
                self.assertEqual((logs/'preparacion.exitcode').read_text(),str(rc)+'\n')
                self.assertTrue((logs/'preparacion.pid').read_text().strip().isdigit())
                record=json.loads(command_record.read_text())
                self.assertEqual(record['cwd'],str(replica))
                self.assertEqual(record['args'],['-u','-m','src.datos.preparar_multicampana','--solo-verificar','--workers','2'])
                self.assertFalse(record['exitcode_previo'])
            command_record.unlink()
            with open(logs/'preparacion.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                pid=(logs/'preparacion.pid').read_bytes()
                r=subprocess.run(['bash',str(replica/entry)],cwd=self.base,env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,1,r.stderr)
                self.assertIn('Ya hay una preparacion activa',r.stdout)
                self.assertFalse(command_record.exists())
                self.assertEqual((logs/'preparacion.pid').read_bytes(),pid)

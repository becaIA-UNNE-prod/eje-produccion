import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cache_lectura import firma, cargar_cache
from utils.multicampana import validar_dataset, cargar_campanas
spec = importlib.util.spec_from_file_location('optimizar', ROOT/'src/optimizar_lectura_multicampana.py')
optimizar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizar)


class CacheYCheckpointTest(unittest.TestCase):
    def test_copia_validacion_y_rechazo_de_cache_modificada(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, target = root/'dataset_A', root/'cache/dataset_A'
            x = np.ones((2, 4, 4, 4), dtype=np.float16)
            y = np.ones((2, 4, 4), dtype=np.uint8)
            records = {}
            for suffix, array in [('_X.npy',x),('_Y.npy',y)]:
                np.save(str(source)+suffix, array)
                records[suffix] = optimizar.copiar_verificado(Path(str(source)+suffix), Path(str(target)+suffix), 0)
                self.assertEqual(records[suffix], optimizar.copiar_verificado(Path(str(source)+suffix),Path(str(target)+suffix),0))
            manifest = root/'cache.json'
            manifest.write_text(json.dumps({'estado':'verificado','rutas':{str(source):{'destino':str(target),'archivos':records}}}))
            routes = cargar_cache(manifest)
            split = {'train':['A'], 'val':[], 'test':[]}
            original = validar_dataset(root, split, 4,4)
            cached = validar_dataset(root, split,4,4,cache=routes)
            self.assertEqual(original[0],cached[0])
            self.assertTrue(np.array_equal(original[1],cached[1]))
            # Modificar destino invalida el manifiesto antes de usar el dataset.
            with Path(str(target)+'_X.npy').open('ab') as file:
                file.write(b'x')
            with self.assertRaises(ValueError): cargar_cache(manifest)

    def _bloque_entrenador(self, attr):
        tree = ast.parse((ROOT/'src/1_train_cordoba.py').read_text())
        function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='entrenar')
        block = next(node for node in function.body if isinstance(node,ast.If)
                     and isinstance(node.test,ast.Attribute) and node.test.attr==attr)
        return compile(ast.Module(body=[block],type_ignores=[]), '<bloque_entrenador>', 'exec')

    def test_init_pesos_entrenador_carga_archivo(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)/'weights.pth'
            original = torch.nn.Linear(2,1)
            with torch.no_grad():
                original.weight.fill_(7); original.bias.fill_(3)
            torch.save(original.state_dict(),path)
            model = torch.nn.Linear(2,1)
            exec(self._bloque_entrenador('init_pesos'), {'ARGS':types.SimpleNamespace(init_pesos=path),
                 'model':model, 'torch':torch,'device':torch.device('cpu')})
            self.assertTrue(torch.equal(model.weight,original.weight))
            self.assertTrue(torch.equal(model.bias,original.bias))

    def test_reanudar_recupera_epoca_modelo_y_optimizador(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(directory)
            original = torch.nn.Linear(2,1)
            optim = torch.optim.Adam(original.parameters(),lr=.003)
            loss = original(torch.ones(2,2)).sum(); loss.backward(); optim.step()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
            scaler = torch.amp.GradScaler('cpu',enabled=False)
            campaigns = [('2018_2019','/data')]
            path = directory/'last.pth'
            torch.save({'args':{'modelo':'unet','campana':campaigns},'epoch':7,
                        'model':original.state_dict(),'optimizer':optim.state_dict(),
                        'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                        'historial':[[7,1,2,3,4]],'mejor':.25,'sin_mejora':2,
                        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'cuda_rng':None},path)
            torch.save(original.state_dict(),directory/'best_model.pth')
            model = torch.nn.Linear(2,1)
            optimizer = torch.optim.Adam(model.parameters(),lr=.01)
            env = {'ARGS':types.SimpleNamespace(reanudar=path,modelo='unet',campana=campaigns),
                   'DIR_EXP':str(directory),'model':model,'optimizer':optimizer,
                   'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                   'scaler':scaler,'device':torch.device('cpu'),'torch':torch,'np':np,'Path':Path}
            exec(self._bloque_entrenador('reanudar'),env)
            self.assertEqual(env['inicio_epoch'],7)
            self.assertEqual(env['sin_mejora'],2)
            self.assertEqual(optimizer.param_groups[0]['lr'],.003)
            self.assertTrue(torch.equal(model.weight,original.weight))
            self.assertTrue(torch.equal(next(iter(optimizer.state.values()))['exp_avg'],next(iter(optim.state.values()))['exp_avg']))

if __name__ == '__main__': unittest.main()

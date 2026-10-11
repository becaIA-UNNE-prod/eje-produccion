"""Cuatro shells históricos en réplicas temporales: sin operaciones reales."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['pipeline_dicmay', 'pipeline_dicmay_1718', 'evaluar_dicmay', 'preparar_ano_no_visto']
Y3 = '/mnt/yacy_3/prod_bis/ferreyra'
Y1 = '/mnt/yacy_1/prod/ferreyra'
FAKE_PY = r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FIXTURE_TRACE'], 'a') as out:
    out.write(json.dumps({'args': args, 'cwd': str(Path.cwd())}) + '\n')
module = args[args.index('-m') + 1]
rest = args[args.index('-m') + 2:]
scenario = os.environ.get('FIXTURE_SCENARIO', '')
def value(flag): return rest[rest.index(flag) + 1]
def touch(path, text='fixture'):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
if module == 'src.datos.descargar_stac' and 'buscar' in rest:
    touch(Path(value('--out')) / 'productos.json', '[]')
if module == 'src.datos.composites_scl' and 'seleccionar' in rest:
    touch(Path(value('--base')) / 'seleccion_scl.json', '[]')
if scenario == 'reintento' and module == 'src.datos.descargar_stac' and 'descargar' in rest:
    marker = Path(os.environ['FIXTURE_TRACE'] + '.retry')
    if not marker.exists():
        marker.touch(); print('SIMULADO'); raise SystemExit(6)
if module == 'src.entrenamiento.entrenar_cordoba':
    if scenario == 'entrenamiento': raise SystemExit(7)
    touch(Path(value('--dir-exp')) / 'modelo_final.pth')
if module == 'src.inferencia.inferir_tile':
    model = value('--modelo')
    suffix = '' if model == 'unet' else '_' + model
    touch(Path(value('--dir-salida')) / ('prediccion_' + rest[0] + suffix + '.tif'))
    if scenario == 'inferencia': raise SystemExit(5)
if scenario == 'evaluacion' and module in ['src.historico.evaluar_contra_mnc', 'src.historico.evaluar_ano_no_visto']:
    raise SystemExit(7)
if scenario == 'composite' and module == 'src.datos.composites_cog' and '2018_2019' in value('--base'):
    raise SystemExit(6)
print('SIMULADO')
"""
FAKE_WAIT = r"""import json, os, sys
from pathlib import Path
command = Path(sys.argv[0]).name
scenario = os.environ.get('FIXTURE_SCENARIO', '')
with open(os.environ['FIXTURE_WAIT'], 'a') as out:
    out.write(json.dumps([command] + sys.argv[1:]) + '\n')
if command == 'pgrep':
    marker = Path(os.environ['FIXTURE_WAIT'] + '.pgrep')
    if scenario == 'descarga_activa' and not marker.exists():
        marker.touch(); raise SystemExit(0)
    raise SystemExit(1)
if scenario == 'fuente_espera':
    source = Path(os.environ['FIXTURE_VOL1']) / 'MNC_INTA/2021_2022/MNC_verano2022.tif'
    source.parent.mkdir(parents=True, exist_ok=True); source.touch()
"""


def run_flow(name, entry='canonical', scenario='', baseline=None):
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        repo = base / 'repo con espacios'
        vol3 = base / 'vol3'; vol3.mkdir()
        vol1 = base / 'vol1'; vol1.mkdir()
        out = base / 'salida con espacios'; out.mkdir()
        exp = base / 'experimento con espacios'; exp.mkdir()
        (out / 'tiles.txt').write_text('20AAA\n')
        (repo / 'config').mkdir(parents=True)
        (repo / 'config/leyenda_mnc_2021_2022.json').write_text('{}')
        (vol3 / 'tiles.txt').write_text('20AAA\n')
        (vol3 / 'logs_multicampana').mkdir()
        (vol3 / 'logs_dicmay').mkdir()
        (vol3 / 'logs_dicmay/pipeline.log').write_text('fallo' if scenario == 'pid_fallo' else 'LISTO')
        if scenario != 'falta_composite':
            (vol3 / 'composites_2018_2019_scl').mkdir()
        if scenario not in ['fuente_espera', 'fuente_falta']:
            source = vol1 / 'MNC_INTA/2021_2022/MNC_verano2022.tif'
            source.parent.mkdir(parents=True); source.touch()
        if scenario == 'cache':
            for relative in ['sentinel2_cordoba_2020_05/productos.json', 'sentinel2_cordoba_2017_2018_dicmay/productos_v2_maxnodata100.json', 'sentinel2_cordoba_2021_2022/productos.json']:
                path = vol3 / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('[]')
            for model in ['unet', 'fcrtransunet']:
                for prefix in ['exp_cordoba_dicmay_', 'exp_cordoba_dicmay_con1718_']:
                    checkpoint = vol3 / 'dataset' / (prefix + model + '_f16') / 'modelo_final.pth'
                    checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.touch()
                suffix = '' if model == 'unet' else '_' + model
                for pred_dir in ['predicciones_2018_2019', 'predicciones_2018_2019_con1718', 'predicciones_2019_2020_dicmay', 'predicciones_2019_2020_dicmay_con1718']:
                    for tile in ['20AAA', '20HLJ', '20HLK', '20HKJ', '20HMH']:
                        path = vol3 / pred_dir / ('prediccion_' + tile + suffix + '.tif')
                        path.parent.mkdir(parents=True, exist_ok=True); path.touch()
        def relocate(body):
            return body.replace(Y3, str(vol3)).replace(Y1, str(vol1))
        if baseline is not None:
            actual = repo / 'src' / (name + '.sh')
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.write_text(relocate((baseline / 'src' / (name + '.sh')).read_text()))
        else:
            for relative in ['src/' + name + '.sh', 'scripts/historico/' + name + '.sh']:
                path = repo / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relocate((ROOT / relative).read_text())); path.chmod(0o755)
            actual = repo / ('src' if entry == 'wrapper' else 'scripts/historico') / (name + '.sh')
        helper = repo/'scripts/lib/corridas.sh'
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text((ROOT/'scripts/lib/corridas.sh').read_text())
        binpath = base / 'bin'; binpath.mkdir()
        fake = binpath / 'python_simulado'; fake.write_text('#!' + sys.executable + '\n' + FAKE_PY); fake.chmod(0o755)
        for command in ['sleep', 'pgrep']:
            path = binpath / command; path.write_text('#!' + sys.executable + '\n' + FAKE_WAIT); path.chmod(0o755)
        date = binpath / 'date'; date.write_text('#!/bin/sh\nprintf "%s\\n" FECHA_FIXTURE\n'); date.chmod(0o755)
        trace = base / 'trace.jsonl'; waits = base / 'wait.jsonl'
        hook = base / 'bash_env'
        hook.write_text('kill() { if [[ "$1" == "-0" ]]; then if [[ ! -f "$FIXTURE_WAIT.kill" ]]; then touch "$FIXTURE_WAIT.kill"; return 0; fi; return 1; fi; return 2; }\n')
        env = dict(os.environ, PY=str(fake), PATH=str(binpath) + os.pathsep + os.environ['PATH'], BASH_ENV=str(hook),
                   FIXTURE_TRACE=str(trace), FIXTURE_WAIT=str(waits), FIXTURE_SCENARIO=scenario, FIXTURE_VOL1=str(vol1))
        if scenario in ['pid_espera', 'pid_fallo']:
            env['ESPERAR_PID'] = '123456789'
        else:
            env.pop('ESPERAR_PID', None)
        env.pop('MODELOS', None)
        env.update(CORDOBA_RUN_DIR=str(out), CORDOBA_RUNTIME_ROOT=str(base/'runtime'))
        command = ['bash', str(actual)]
        if name == 'preparar_ano_no_visto':
            if scenario != 'argumentos_faltantes': command.extend([str(out), str(exp), 'argumento-extra'])
        gpu_wait = False
        if scenario == 'lock':
            with open(out / 'preparacion.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True, timeout=15)
        elif scenario == 'gpu_lock':
            with open(vol3 / 'logs_multicampana/inferencia.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                process = subprocess.Popen(command, cwd=base, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if trace.exists() and 'componer' in trace.read_text(): break
                        if process.poll() is not None: raise AssertionError('salió antes del lock GPU')
                        time.sleep(.01)
                    else: raise AssertionError('no llegó a composición')
                    time.sleep(.05)
                    assert process.poll() is None
                    assert 'src.historico.evaluar_ano_no_visto' not in trace.read_text()
                    gpu_wait = True
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                stdout, stderr = process.communicate(timeout=15)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        else:
            result = subprocess.run(command, cwd=base, env=env, capture_output=True, text=True, timeout=15)
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        for event in events: assert event['cwd'] == str(repo), event
        # El composite de 2018/2019 corre en paralelo con descarga; conservar su evento aparte.
        background = []; ordered = []
        for event in events:
            if name == 'pipeline_dicmay' and 'src.datos.composites_cog' in event['args'] and str(vol3 / 'sentinel2_cordoba_2018_2019') in event['args']:
                background.append(event)
            else: ordered.append(event)
        state = {}
        for volume in [vol3, vol1, out]:
            for path in volume.rglob('*'):
                key = str(path.relative_to(base))
                if path.is_symlink(): state[key] = {'link': os.readlink(path)}
                elif path.is_file():
                    value = path.read_text()
                    if path.suffix == '.pid':
                        assert value.strip().isdigit(); value = 'PID_FIXTURE\n'
                    state[key] = value
        value = {'rc': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'events': ordered,
                 'background': background, 'waits': waits.read_text() if waits.exists() else '', 'state': state, 'gpu_wait': gpu_wait}
        return json.loads(json.dumps(value).replace(str(base), 'FIXTURE_ROOT'))


def args(result): return [event['args'] for event in result['events']]


class ShellsHistoricosTest(unittest.TestCase):
    def test_cuatro_wrappers_equivalentes_y_ejecutables(self):
        for name in NAMES:
            with self.subTest(name=name):
                for folder in ['src', 'scripts/historico']:
                    self.assertTrue(os.access(ROOT / folder / (name + '.sh'), os.X_OK))
                actual = run_flow(name)
                self.assertEqual(actual, run_flow(name, entry='wrapper'))
                self.assertEqual(actual['rc'], 0, actual['stderr'])
                self.assertTrue(actual['events'])
                if name == 'preparar_ano_no_visto':
                    self.assertEqual(actual['state']['salida con espacios/preparacion.exitcode'], '0\n')
                    self.assertIn('experimento con espacios', str(args(actual)[-1]))

    def test_pipelines_cache_reintentos_y_fallos(self):
        for name in NAMES[:2]:
            with self.subTest(name=name):
                cache = run_flow(name, scenario='cache')
                self.assertEqual(cache['rc'], 0)
                self.assertFalse(any('buscar' in a or 'src.entrenamiento.entrenar_cordoba' in a or 'src.inferencia.inferir_tile' in a for a in args(cache)))
                retry = run_flow(name, scenario='reintento')
                self.assertEqual(retry['rc'], 0)
                self.assertGreaterEqual(sum('descargar' in a for a in args(retry)), 2)
                failed = run_flow(name, scenario='inferencia')
                self.assertEqual(failed['rc'], 1)
                self.assertFalse(any(key.endswith('.tif') and '/predicciones_' in key for key in failed['state']))
                training = run_flow(name, scenario='entrenamiento')
                self.assertEqual(training['rc'], 7)
                self.assertFalse(any('src.inferencia.inferir_tile' in a for a in args(training)))
        self.assertEqual(run_flow('pipeline_dicmay', scenario='composite')['rc'], 6)
        self.assertEqual(run_flow('pipeline_dicmay_1718', scenario='falta_composite')['rc'], 1)
        waiting = run_flow('pipeline_dicmay_1718', scenario='descarga_activa')
        self.assertEqual(waiting['rc'], 0)
        self.assertIn('60', waiting['waits'])

    def test_evaluacion_espera_pid_y_fallos(self):
        wait = run_flow('evaluar_dicmay', scenario='pid_espera')
        self.assertEqual(wait['rc'], 0)
        self.assertIn('300', wait['waits'])
        failed = run_flow('evaluar_dicmay', scenario='pid_fallo')
        self.assertEqual(failed['rc'], 1)
        self.assertEqual(failed['events'], [])
        self.assertEqual(run_flow('evaluar_dicmay', scenario='evaluacion')['rc'], 7)
        cache = run_flow('evaluar_dicmay', scenario='cache')
        self.assertEqual(cache['rc'], 0)
        self.assertEqual(len(args(cache)), 2)

    def test_temporal_locks_esperas_trap_y_argumentos(self):
        locked = run_flow('preparar_ano_no_visto', scenario='lock')
        self.assertEqual(locked['rc'], 1)
        self.assertEqual(locked['events'], [])
        self.assertNotIn('salida con espacios/preparacion.pid', locked['state'])
        for scenario, code in [('fuente_espera', 0), ('fuente_falta', 1), ('evaluacion', 7), ('gpu_lock', 0)]:
            with self.subTest(scenario=scenario):
                actual = run_flow('preparar_ano_no_visto', scenario=scenario)
                self.assertEqual(actual['rc'], code)
                self.assertEqual(actual['state']['salida con espacios/preparacion.exitcode'], str(code) + '\n')
                self.assertEqual(actual['state']['salida con espacios/fin.txt'], 'FECHA_FIXTURE\n')
                if scenario.startswith('fuente'): self.assertIn('30', actual['waits'])
                if scenario == 'gpu_lock': self.assertTrue(actual['gpu_wait'])
        missing = run_flow('preparar_ano_no_visto', scenario='argumentos_faltantes')
        self.assertNotEqual(missing['rc'], 0)
        self.assertEqual(missing['events'], [])


if __name__ == '__main__':
    unittest.main()

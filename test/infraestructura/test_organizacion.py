"""Verifica el registro de corridas y la salida de figuras fuera de la raíz."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class OrganizacionTests(unittest.TestCase):
    def test_rechaza_corridas_dentro_del_repo(self):
        base = ROOT / 'salida_prohibida_test'
        result = self.run_logged(base, 'print("no ejecutar")')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('fuera del repositorio', result.stderr)
        self.assertFalse(base.exists())

    def run_logged(self, base, code):
        with tempfile.TemporaryDirectory() as runtime:
            return subprocess.run([
                sys.executable, str(ROOT / 'scripts/ejecutar_con_log.py'),
                '--nombre', 'prueba', '--base', str(base), '--', sys.executable, '-c', code,
            ], cwd=ROOT, env=dict(os.environ, CORDOBA_RUNTIME_ROOT=runtime), capture_output=True, text=True)


    def test_log_y_procedencia_de_corrida(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_logged(tmp, 'import os; print(os.environ["CORDOBA_RUN_DIR"])')
            self.assertEqual(result.returncode, 0, result.stderr)
            run, = Path(tmp).iterdir()
            self.assertEqual((run / 'logs/coordinador.log').read_text().strip(), str(run))
            record = json.loads((run / 'run.json').read_text())
            self.assertEqual(record['estado'], 'finalizado')
            self.assertEqual(record['exitcode'], 0)
            self.assertEqual(len(record['split_config_sha256']), 64)
            self.assertIn('git_status', record)
            self.assertEqual(json.loads((run / 'estado.json').read_text())['exitcode'], 0)
            self.assertEqual((run / 'configuracion/split_extension_este.json').read_bytes(),
                             (ROOT / 'config/split_extension_este.json').read_bytes())
            self.assertTrue((run / 'mapas/mapa_tiles.html').is_file())
            published = ROOT / record['mapa_repo']
            self.assertEqual(published.read_bytes(), (run / 'mapas/mapa_tiles.html').read_bytes())
            self.assertFalse((run / 'predicciones').exists())
            self.assertTrue((run / 'logs/workers').is_dir())

    def test_fallo_conserva_log_y_exitcode(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_logged(tmp, 'import sys; print("evidencia", file=sys.stderr); sys.exit(7)')
            self.assertEqual(result.returncode, 7, result.stderr)
            run, = Path(tmp).iterdir()
            self.assertEqual((run / 'exitcode').read_text().strip(), '7')
            self.assertIn('evidencia', (run / 'logs/coordinador.log').read_text())
            self.assertEqual(json.loads((run / 'run.json').read_text())['estado'], 'fallo')

    def test_visualizacion_respeta_carpeta_de_corrida(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'mapa.tif'
            with rasterio.open(source, 'w', driver='GTiff', width=2, height=2,
                               count=1, dtype='uint8', crs='EPSG:32720',
                               transform=from_origin(500000, 6200000, 10, 10)) as dst:
                dst.write(np.array([[1, 2], [3, 4]], dtype=np.uint8), 1)
            run = folder / 'corrida'
            result = subprocess.run([
                sys.executable, str(ROOT / 'src/7_visualizar_inferencia.py'), str(source),
            ], cwd=folder, env=dict(os.environ, CORDOBA_RUN_DIR=str(run)),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run / 'figuras/mapa.png').is_file())
            self.assertFalse(source.with_suffix('.png').exists())
            anterior = (run / 'figuras/mapa.png').read_bytes()
            result = subprocess.run([
                sys.executable, '-m', 'src.figuras.visualizar_inferencia', str(source),
            ], cwd=ROOT, env=dict(os.environ, CORDOBA_RUN_DIR=str(run)),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((run / 'figuras/mapa.png').read_bytes(), anterior)



class RunsSeparadosTests(unittest.TestCase):
    def test_rutas_pesadas_reintentos_y_registro_del_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runs = base / 'runs'
            output = base / 'pesado'
            argv = [sys.executable, str(ROOT / 'scripts/ejecutar_con_log.py'),
                    '--base', str(runs), '--nombre', 'evaluar', '--etapa', 'evaluacion',
                    '--salida', f'predicciones={output}', '--', sys.executable, '-c',
                    'import os; from utils.corridas import registrar_rutas; '
                    'registrar_rutas(os.environ["CORDOBA_RUN_DIR"], salidas=[("metricas", os.environ["DESTINO"])])']
            env = dict(os.environ, CORDOBA_RUNTIME_ROOT=str(base / 'runtime'), DESTINO=str(base / 'metricas'))
            first = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            parent, = runs.iterdir()
            second = subprocess.run(argv[:2] + ['--padre', str(parent)] + argv[2:], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            children = [p for p in runs.iterdir() if p != parent]
            self.assertEqual(len(children), 1)
            record = json.loads((children[0] / 'run.json').read_text())
            self.assertEqual(record['padre'], str(parent))
            self.assertEqual(record['salidas']['predicciones'], str(output))
            self.assertEqual(record['salidas']['metricas'], str(base / 'metricas'))
            self.assertFalse(output.exists())
            self.assertFalse((base / 'metricas').exists())

    def test_prevision_suma_salidas_en_el_mismo_disco(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            import shutil
            free = shutil.disk_usage(base).free / 2**30
            argv = [sys.executable, str(ROOT / 'scripts/ejecutar_con_log.py'), '--nombre', 'capacidad',
                    '--base', str(base / 'runs'), '--reserva-gib', '0',
                    '--salida', f'a={base / "a"}', '--salida', f'b={base / "b"}',
                    '--prevision-gib', f'a={free * .6}', '--prevision-gib', f'b={free * .6}',
                    '--', sys.executable, '-c', 'raise AssertionError("no ejecutar")']
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Espacio insuficiente', result.stderr)
            self.assertFalse((base / 'runs').exists())


class ShellAutomaticoTests(unittest.TestCase):
    def test_shell_directo_crea_runs_distintos_y_captura_fallos(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            repo = base / 'repo'
            for name in ['scripts/ejecutar_con_log.py', 'scripts/registrar_rutas_corrida.py',
                         'scripts/lib/corridas.sh', 'utils/rutas.py',
                         'utils/corridas.py', 'utils/mapas_tiles.py', 'utils/mapa_tiles.html',
                         'config/geometria_tiles.json', 'config/campanas.json', 'config/almacenamiento.json', 'config/split_extension_este.json']:
                target = repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            script = repo / 'scripts/evaluacion/inferencia_multicampana.sh'
            script.parent.mkdir(parents=True)
            script.write_text((ROOT / 'scripts/evaluacion/inferencia_multicampana.sh').read_text().replace(
                '/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana', str(base / 'locks_historicos')))
            subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                            'commit', '--allow-empty', '-qm', 'fixture'], check=True, capture_output=True)
            fake = base / 'python_ml'
            fake.write_text('#!' + sys.executable + '\nimport os; os.fstat(9); print("ML SIMULADO"); raise SystemExit(7)\n')
            fake.chmod(0o755)
            out = base / 'resultado'; out.mkdir()
            exp = base / 'experimento'; exp.mkdir()
            runs = base / 'runs'
            env = dict(os.environ, PY=str(fake), CORDOBA_PY_REGISTRO=sys.executable,
                       CORDOBA_RUNS_ROOT=str(runs), CORDOBA_RUNTIME_ROOT=str(base / 'runtime'))
            env.pop('CORDOBA_RUN_DIR', None)
            for attempt in range(2):
                result = subprocess.run(['bash', str(script), str(exp), str(out)], cwd=base,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(len(list(runs.iterdir())), 2)
            for run in runs.iterdir():
                record = json.loads((run / 'run.json').read_text())
                self.assertEqual(record['salidas']['resultados'], str(out))
                self.assertEqual(record['salidas']['experimento'], str(exp))
                self.assertEqual(record['exitcode'], 7)
                self.assertIn('ML SIMULADO', (run / 'logs/coordinador.log').read_text())
                self.assertFalse((run / 'predicciones').exists())
            self.assertTrue((base / 'runtime/locks/inferencia.lock').exists())


class DestinosExplicitosTests(unittest.TestCase):
    def test_herramientas_historicas_exigen_destinos_en_corridas_livianas(self):
        from unittest.mock import patch
        import importlib
        modules = [('src.inferencia.inferir_tile', [], '--dir-salida'),
                   ('src.figuras.visualizar_inferencia', ['mapa.tif'], '--salida'),
                   ('src.historico.comparar_modelos', [], '--salida_dir'),
                   ('src.historico.comparar_inferencia_etiqueta', [], '--dir-salida'),
                   ('src.historico.metricas_test', [], '--dir-pred')]
        import contextlib, io
        with patch.dict(os.environ, CORDOBA_REGISTRO_LIVIANO='1'):
            for name, argv, flag in modules:
                with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()) as error:
                    parser = importlib.import_module(name).crear_parser()
                    with self.assertRaises(SystemExit) as result:
                        parser.parse_args(argv)
                    self.assertEqual(result.exception.code, 2)
                    self.assertIn(flag, error.getvalue())


if __name__ == "__main__":
    unittest.main()

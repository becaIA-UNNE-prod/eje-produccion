"""Verifica el registro de corridas y la salida de figuras fuera de la raíz."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OrganizacionTests(unittest.TestCase):
    def test_rechaza_corridas_dentro_del_repo(self):
        base = ROOT / 'salida_prohibida_test'
        result = self.run_logged(base, 'print("no ejecutar")')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('fuera del repositorio', result.stderr)
        self.assertFalse(base.exists())

    def run_logged(self, base, code):
        return subprocess.run([
            sys.executable, str(ROOT / 'scripts/ejecutar_con_log.py'),
            '--nombre', 'prueba', '--base', str(base), '--', sys.executable, '-c', code,
        ], cwd=ROOT, capture_output=True, text=True)

    def test_log_y_procedencia_de_corrida(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_logged(tmp, 'import os; print(os.environ["CORDOBA_RUN_DIR"])')
            self.assertEqual(result.returncode, 0, result.stderr)
            run, = Path(tmp).iterdir()
            self.assertEqual((run / 'logs/stdout.log').read_text().strip(), str(run))
            record = json.loads((run / 'run.json').read_text())
            self.assertEqual(record['estado'], 'finalizado')
            self.assertEqual(record['exitcode'], 0)
            self.assertEqual(len(record['split_config_sha256']), 64)
            self.assertIn('git_status', record)

    def test_fallo_conserva_log_y_exitcode(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_logged(tmp, 'import sys; print("evidencia", file=sys.stderr); sys.exit(7)')
            self.assertEqual(result.returncode, 7, result.stderr)
            run, = Path(tmp).iterdir()
            self.assertEqual((run / 'exitcode').read_text().strip(), '7')
            self.assertIn('evidencia', (run / 'logs/stdout.log').read_text())
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


if __name__ == '__main__':
    unittest.main()

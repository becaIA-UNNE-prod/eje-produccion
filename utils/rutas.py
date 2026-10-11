"""Rutas de salida de las herramientas históricas de octubre–abril.

Los evaluadores de TEST vigente mantienen sus rutas explícitas en /mnt.
CORDOBA_RUN_DIR permite agrupar cada ejecución manual en su propia carpeta.
"""
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_STORAGE = json.loads((ROOT / 'config/almacenamiento.json').read_text())
STORAGE_ROOT = Path(os.environ.get('CORDOBA_STORAGE_ROOT', CONFIG_STORAGE['storage_root'])).expanduser()
RUNS_ROOT = Path(os.environ.get('CORDOBA_RUNS_ROOT', CONFIG_STORAGE['runs_root'])).expanduser()
if not STORAGE_ROOT.is_absolute() or not RUNS_ROOT.is_absolute():
    raise ValueError('CORDOBA_STORAGE_ROOT y CORDOBA_RUNS_ROOT requieren rutas absolutas fuera del repositorio')
RUNTIME_ROOT = Path(os.environ.get('CORDOBA_RUNTIME_ROOT', CONFIG_STORAGE.get('runtime_root', str(RUNS_ROOT.parent.parent / 'runtime')))).expanduser()
if not RUNTIME_ROOT.is_absolute() or RUNTIME_ROOT.resolve() == ROOT or ROOT in RUNTIME_ROOT.resolve().parents:
    raise ValueError('CORDOBA_RUNTIME_ROOT requiere una ruta absoluta fuera del repositorio')
DATA_ROOT = STORAGE_ROOT / 'data'
ARCHIVE_ROOT = STORAGE_ROOT / 'artifacts/archive'
MASKS_ROOT = DATA_ROOT / 'mascaras'
REFERENCES_ROOT = DATA_ROOT / 'referencias'
RUN_DIR = Path(os.environ.get('CORDOBA_RUN_DIR', RUNS_ROOT / 'legacy_octabr')).expanduser()
if not RUN_DIR.is_absolute():
    RUN_DIR = ROOT / RUN_DIR
if RUN_DIR.resolve() == ROOT or ROOT in RUN_DIR.resolve().parents:
    raise ValueError('CORDOBA_RUN_DIR debe estar fuera del repositorio; usar un disco /mnt/yacy_*')
LEGACY_PREDICTIONS = RUN_DIR / 'predicciones'
LEGACY_FIGURES = RUN_DIR / 'figuras'

# Ubicaciones existentes acreditadas: conservarlas fuera del namespace nuevo.
OPERATIVAS = CONFIG_STORAGE['rutas_operativas']
MULTICAMPANA_ROOT = Path(OPERATIVAS['multicampana'])
EXTENSION_ROOT = Path(OPERATIVAS['extension_este'])
MNC_ROOT = Path(OPERATIVAS['mnc'])
if not all(p.is_absolute() for p in (MULTICAMPANA_ROOT, EXTENSION_ROOT, MNC_ROOT)):
    raise ValueError('Las rutas operativas requieren ubicaciones absolutas')

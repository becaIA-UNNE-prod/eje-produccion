"""Comprueba explícitamente la preparación acreditada, sin inferir ni escribir."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.test_actual import CONFIG, cargar_config, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Comprobación operativa de metadatos y presencia de máscaras del TEST vigente.',
        epilog='Requiere la preparación configurada. No audita bloques raster ni checkpoints; no genera salidas.',
    )
    parser.add_argument('--split-config', type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    cfg = cargar_config(args.split_config)
    base = preflight(cfg)
    print(f"PREPARACION TEST ACTUAL OK: {base}; tiles={cfg['test']}; campañas={cfg['campanas']}")
    return base


if __name__ == '__main__':
    main()

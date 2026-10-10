"""Indexa tiles, fechas y bandas JP2; no audita la integridad de las imágenes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.rutas import RUN_DIR


def mapear_estructura_sentinel(base_path):
    base_dir = Path(base_path)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Directorio inexistente: {base_dir}")
    estructura = {}
    total = 0
    for tile in sorted(base_dir.iterdir()):
        if not tile.is_dir():
            continue
        estructura[tile.name] = {}
        for fecha in sorted(tile.iterdir()):
            if fecha.is_dir():
                bandas = sorted(p.stem.split('_')[-1] for p in fecha.glob('*.jp2'))
                estructura[tile.name][fecha.name] = bandas
                total += len(bandas)
    return estructura, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Carpeta tile/fecha/*.jp2')
    parser.add_argument('--salida', type=Path, default=RUN_DIR/'inventarios/estructura_sentinel2.json')
    args = parser.parse_args()
    try:
        datos, total = mapear_estructura_sentinel(args.base)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    salida = args.salida.resolve()
    if salida == ROOT or ROOT in salida.parents:
        parser.error('La salida debe estar físicamente fuera del repositorio')
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(datos, ensure_ascii=False, indent=4)+'\n')
    print(f'{len(datos)} tiles, {total} bandas JP2; inventario: {salida}')


if __name__ == '__main__':
    main()

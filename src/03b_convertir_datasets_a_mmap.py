"""
Convierte datasets .npz existentes (comprimidos -> hay que cargarlos enteros
en RAM para leer cualquier parche) a pares de archivos .npy sueltos, sin
comprimir, que se pueden abrir con mmap_mode="r" (utils/tile_dataset.py).

Por que: np.load sobre un .npz comprimido descomprime el array COMPLETO en
RAM apenas se accede a una clave -- no existe forma de leer "un pedacito".
Separando X e Y en archivos .npy individuales (formato binario simple, sin
zip ni compresion), numpy puede memory-mapear el archivo: el sistema
operativo trae a RAM solo las paginas que efectivamente se leen. Mismos
datos, mismos valores en los mismos parches -- no se pierde informacion,
solo cambia el contenedor en disco.

Nota de espacio: al no comprimir, el resultado ocupa mas disco que el .npz
de origen (cuanto mas, depende de cuanto comprimia esa banda/tile puntual).
Verificar espacio libre antes de correr sobre datasets grandes.

Uso:
    python3 src/03b_convertir_datasets_a_mmap.py /mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16
    python3 src/03b_convertir_datasets_a_mmap.py /mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16 --borrar-original
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir_datasets", help="Carpeta con los .npz a convertir")
parser.add_argument("--borrar-original", action="store_true",
                     help="Borra el .npz de origen despues de verificar que la conversion es identica")
args = parser.parse_args()

dir_datasets = Path(args.dir_datasets)
archivos = sorted(dir_datasets.glob("*.npz"))
if not archivos:
    sys.exit(f"No hay .npz en {dir_datasets}")

for ruta_npz in archivos:
    base = ruta_npz.with_suffix("")
    ruta_x = base.parent / f"{base.name}_X.npy"
    ruta_y = base.parent / f"{base.name}_Y.npy"

    if ruta_x.exists() and ruta_y.exists():
        print(f"{ruta_npz.name}: ya convertido, omitiendo")
        continue

    print(f"{ruta_npz.name}: convirtiendo...")
    with np.load(ruta_npz) as data:
        X = data["X"]
        Y = data["Y"]
        if len(X) != len(Y):
            print(f"  ERROR: X tiene {len(X)} parches pero Y tiene {len(Y)}, se omite")
            continue

        # Algunos .npz (ej. train_cordoba_f16) traen un flag "remap_aplicado"
        # que 1_train_cordoba.py verificaba al cargar, para no entrenar por
        # error con un dataset sin remapear a 6 clases. El .npy suelto no
        # tiene donde guardar ese flag, asi que la validacion se hace aca,
        # una vez, al convertir -- si esto pasa, el .npy resultante ya quedo
        # verificado y el chequeo en tiempo de entrenamiento deja de hacer falta.
        if "remap_aplicado" in data and not bool(data["remap_aplicado"]):
            print(f"  ERROR: remap_aplicado=0 en {ruta_npz.name}, se omite (dataset sin remapear)")
            continue

        tmp_x = ruta_x.with_suffix(ruta_x.suffix + ".tmp")
        tmp_y = ruta_y.with_suffix(ruta_y.suffix + ".tmp")
        with open(tmp_x, "wb") as f:
            np.save(f, X)
        with open(tmp_y, "wb") as f:
            np.save(f, Y)

    # Verificacion: releer lo escrito con mmap y comparar contra el original
    # antes de dar la conversion por buena (y antes de tocar el .npz de origen).
    X_verif = np.load(tmp_x, mmap_mode="r")
    Y_verif = np.load(tmp_y, mmap_mode="r")
    if not (np.array_equal(X_verif, X) and np.array_equal(Y_verif, Y)):
        print(f"  ERROR: la verificacion post-escritura no coincide con el original, se descarta")
        tmp_x.unlink(missing_ok=True)
        tmp_y.unlink(missing_ok=True)
        continue
    del X_verif, Y_verif

    os.replace(tmp_x, ruta_x)
    os.replace(tmp_y, ruta_y)
    print(f"  OK: {len(X)} parches -> {ruta_x.name} / {ruta_y.name}")

    if args.borrar_original:
        ruta_npz.unlink()
        print(f"  borrado {ruta_npz.name}")

print("\nListo.")

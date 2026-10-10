"""
Inferencia de un modelo entrenado con 1_train_cordoba.py sobre un tile completo.

Default: campana 2019-2020 (oct-abr), pesos en exp_cordoba_[<modelo>_]f16.
Campana 2018-2019 (dic-may, ver 02c_composites_cog.py) con un modelo
entrenado en esa misma ventana:
    python3 src/2_inferencia.py 20HLJ --modelo unet \
        --dir-composites /mnt/yacy_3/prod_bis/ferreyra/composites_2018_2019 \
        --meses 201812 201901 201902 201903 201904 201905 \
        --dir-exp /mnt/yacy_3/prod_bis/ferreyra/dataset/exp_cordoba_dicmay_unet_f16 \
        --dir-salida /mnt/yacy_3/prod_bis/ferreyra/predicciones_2018_2019
Los meses tienen que ser la misma ventana del ciclo (en el mismo orden) que la
usada para entrenar los pesos: la posicion del canal es la unica senal de mes.
"""
import argparse
import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import sys

sys.path.append(os.path.abspath("."))
from model import SimpleUNet, TransUNet, UTAE, FCRTransUNet
from utils.rutas import LEGACY_PREDICTIONS

def generar_mapa_clasificacion():
    # Configuración alineada con el pipeline de entrenamiento vigente (1_train_cordoba.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("tile", nargs="?", default="20HMK")
    parser.add_argument("--modelo", choices=["unet", "transunet", "utae", "fcrtransunet"],
                         default="unet",
                         help="unet=SimpleUNet (baseline), transunet=TransUNet, "
                              "utae=UTAE, fcrtransunet=FCRTransUNet (ver model/README.md) "
                              "-- debe coincidir con el modelo usado para entrenar los "
                              "pesos de exp_cordoba_*_f16")
    parser.add_argument("--dir-composites", default=None,
                         help="dir con <TILE>_<AAAAMM>_median.tif (default: composites_2019_2020)")
    parser.add_argument("--meses", nargs="*", default=None,
                         help="meses AAAAMM en el orden de entrenamiento (default: 201910..202004)")
    parser.add_argument("--dir-exp", default=None,
                         help="dir con best_model.pth (default: exp_cordoba_[<modelo>_]f16)")
    parser.add_argument("--dir-salida", default=str(LEGACY_PREDICTIONS),
                         help="dir donde escribir prediccion_<TILE>[_<modelo>].tif")
    args = parser.parse_args()

    TILE = args.tile
    DIR_COMPOSITES = args.dir_composites or "/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020"
    # exp_cordoba_f16 (unet) se deja con el nombre historico; las demas variantes
    # usan el dir separado que ya escribe 1_train_cordoba.py --modelo <variante>.
    DIR_EXP = "/mnt/yacy_1/prod/ferreyra/dataset/exp_cordoba_f16" if args.modelo == "unet" \
              else f"/mnt/yacy_1/prod/ferreyra/dataset/exp_cordoba_{args.modelo}_f16"
    DIR_EXP = args.dir_exp or DIR_EXP
    RUTA_PESOS = f"{DIR_EXP}/best_model.pth"
    sufijo = "" if args.modelo == "unet" else f"_{args.modelo}"
    os.makedirs(args.dir_salida, exist_ok=True)
    RUTA_SALIDA = os.path.join(args.dir_salida, f"prediccion_{TILE}{sufijo}.tif")

    # 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani (sin Sorgo, mismo esquema que 1_train_cordoba.py)
    NUM_CLASSES = 5
    SIZE = 512  # Ventana de inferencia

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferencia en: {device}")

    # 2. Meses usados para entrenar exp_cordoba_f16 (dataset_, campaña 2019-2020 MNC INTA):
    # ver bash_history del generador de datasets_mnc.log. No autodetectar: mismo
    # motivo que antes, composites_2019_2020 puede tener más meses que estos.
    MESES_COMUNES = args.meses or ["201910", "201911", "201912", "202001", "202002", "202003", "202004"]
    archivos_mensuales = [f"{TILE}_{mes}_median.tif" for mes in MESES_COMUNES]
    faltantes = [f for f in archivos_mensuales if not os.path.exists(os.path.join(DIR_COMPOSITES, f))]
    if faltantes:
        raise FileNotFoundError(f"Faltan composites para {TILE} en {DIR_COMPOSITES}: {faltantes}")

    # 4 bandas por cada mes procesado
    IN_CHANNELS = len(archivos_mensuales) * 4
    print(f"Usando {IN_CHANNELS} canales ({len(archivos_mensuales)} meses).")

    # 3. Cargar el modelo entrenado
    if args.modelo == "unet":
        model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    elif args.modelo == "transunet":
        model = TransUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    elif args.modelo == "utae":
        model = UTAE(IN_CHANNELS, NUM_CLASSES, T=len(archivos_mensuales)).to(device)
    else:
        model = FCRTransUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(f"No se encontraron los pesos en {RUTA_PESOS}")

    model.load_state_dict(torch.load(RUTA_PESOS, map_location=device))
    model.eval()

    # 4. Metadatos espaciales del primer composite (misma grilla que la
    # mascara MNC; asi no hace falta tener mascara para tiles/campanas sin etiquetas)
    with rasterio.open(os.path.join(DIR_COMPOSITES, archivos_mensuales[0])) as src_ref:
        meta = src_ref.meta.copy()
        alto, ancho = src_ref.height, src_ref.width
        # Forzamos la salida a ser de un solo canal, tipo uint8 (suficiente para NUM_CLASSES)
        meta.update(dtype=rasterio.uint8, count=1, nodata=0, driver="GTiff", compress="deflate")

    print(f"Dimensiones del mapa a predecir: {ancho} x {alto} píxeles.")
    print("Prediciendo por bloques y escribiendo en disco...")

    # 5. Bucle de Inferencia
    with rasterio.open(RUTA_SALIDA, 'w', **meta) as dst:
        with torch.no_grad():
            for y in range(0, alto, SIZE):
                for x in range(0, ancho, SIZE):
                    w = min(SIZE, ancho - x)
                    h = min(SIZE, alto - y)
                    ventana = Window(x, y, w, h)

                    parche_x_temporal = []

                    # Leer de los composites mensuales
                    for archivo_mes in archivos_mensuales:
                        ruta_mes = os.path.join(DIR_COMPOSITES, archivo_mes)
                        with rasterio.open(ruta_mes) as src_mes:
                            # Lee todas las bandas de ese mes para esa ventana
                            parche = src_mes.read(window=ventana)
                            parche_x_temporal.append(parche)

                    # Apilar y normalizar
                    parche_x = np.concatenate(parche_x_temporal, axis=0).astype(np.float32)

                    # Debe coincidir con la normalización usada en TileDatasetMmap (1_train_cordoba.py)
                    parche_x = parche_x / 10000.0

                    # Las ventanas de borde (ancho/alto no multiplo de SIZE, p.ej.
                    # 10980 % 512 = 228) quedan con H/W no multiplo de 16 -- al
                    # bajar 4 niveles con MaxPool2d(2) eso desalinea el tamaño del
                    # ConvTranspose2d contra el skip connection y explota el
                    # torch.cat. Se rellena hasta el multiplo de 16 mas cercano
                    # (padding "edge", no afecta la prediccion real) y se recorta
                    # el resultado antes de escribir.
                    h_real, w_real = parche_x.shape[1], parche_x.shape[2]
                    pad_h = (-h_real) % 16
                    pad_w = (-w_real) % 16
                    if pad_h or pad_w:
                        parche_x = np.pad(parche_x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

                    # (Batch, Channels, Height, Width)
                    tensor_x = torch.from_numpy(parche_x).unsqueeze(0).to(device)

                    # Predicción
                    salida = model(tensor_x)
                    prediccion = torch.argmax(salida, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    if pad_h or pad_w:
                        prediccion = prediccion[:h_real, :w_real]
                    # Pixeles sin dato en todos los meses (fuera de la huella
                    # de las escenas) -> NoData en vez de una clase inventada
                    sin_dato = (parche_x[:, :h_real, :w_real] == 0).all(axis=0)
                    prediccion[sin_dato] = 0

                    # Escribir en TIF
                    dst.write(prediccion, 1, window=ventana)

    print(f"Inferencia completada. Mapa guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    generar_mapa_clasificacion()

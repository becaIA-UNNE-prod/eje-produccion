import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import sys

sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet

def generar_mapa_clasificacion():
    # Configuración alineada con el pipeline de entrenamiento vigente (1_train_cordoba.py)
    TILE = sys.argv[1] if len(sys.argv) > 1 else "20HMK"
    DIR_COMPOSITES = "/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020"
    RUTA_MASCARA = f"./mascaras_procesadas/etiqueta_mnc_{TILE}_10m.tif"
    RUTA_PESOS = "/mnt/yacy_1/prod/ferreyra/dataset/exp_cordoba_f16/best_model.pth"
    RUTA_SALIDA = f"./prediccion_{TILE}.tif"

    # 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani (sin Sorgo, mismo esquema que 1_train_cordoba.py)
    NUM_CLASSES = 5
    SIZE = 512  # Ventana de inferencia

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferencia en: {device}")

    # 2. Meses usados para entrenar exp_cordoba_f16 (dataset_, campaña 2019-2020 MNC INTA):
    # ver bash_history del generador de datasets_mnc.log. No autodetectar: mismo
    # motivo que antes, composites_2019_2020 puede tener más meses que estos.
    MESES_COMUNES = ["201910", "201911", "201912", "202001", "202002", "202003", "202004"]
    archivos_mensuales = [f"{TILE}_{mes}_median.tif" for mes in MESES_COMUNES]
    faltantes = [f for f in archivos_mensuales if not os.path.exists(os.path.join(DIR_COMPOSITES, f))]
    if faltantes:
        raise FileNotFoundError(f"Faltan composites para {TILE} en {DIR_COMPOSITES}: {faltantes}")

    # 4 bandas por cada mes procesado
    IN_CHANNELS = len(archivos_mensuales) * 4
    print(f"Usando {IN_CHANNELS} canales ({len(archivos_mensuales)} meses).")

    # 3. Cargar el modelo entrenado
    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(f"No se encontraron los pesos en {RUTA_PESOS}")

    model.load_state_dict(torch.load(RUTA_PESOS, map_location=device))
    model.eval()

    # 4. Usar la máscara original solo para metadatos espaciales
    with rasterio.open(RUTA_MASCARA) as src_ref:
        meta = src_ref.meta.copy()
        alto, ancho = src_ref.height, src_ref.width
        # Forzamos la salida a ser de un solo canal, tipo uint8 (suficiente para NUM_CLASSES)
        meta.update(dtype=rasterio.uint8, count=1, nodata=0)

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

                    # (Batch, Channels, Height, Width)
                    tensor_x = torch.from_numpy(parche_x).unsqueeze(0).to(device)

                    # Predicción
                    salida = model(tensor_x)
                    prediccion = torch.argmax(salida, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    # Escribir en TIF
                    dst.write(prediccion, 1, window=ventana)

    print(f"Inferencia completada. Mapa guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    generar_mapa_clasificacion()

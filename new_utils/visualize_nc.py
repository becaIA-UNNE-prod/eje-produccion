"""Transforma un nc file en un jpg para visualizar fácilmente la imagen satelital"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Nombre exacto del archivo .nc que quieres convertir
INPUT_FILE = "images_tests/texas_radio_15km.nc" 

# Calidad de la imagen de salida (DPI)
# 100 = web/pantalla normal
# 300 = alta calidad para impresión/zoom
CALIDAD_DPI = 150 

def generar_jpg_desde_nc(filename):
    if not os.path.exists(filename):
        print(f"❌ Error: No encuentro el archivo '{filename}'")
        return

    print(f"📂 Leyendo archivo: {filename}...")
    
    # 1. Abrir el NetCDF
    try:
        ds = xr.open_dataset(filename)
    except Exception as e:
        print(f"Error al abrir el NetCDF. Asegúrate de tener 'netCDF4' instalado.\nDetalle: {e}")
        return

    # 2. Verificar que existan las bandas necesarias
    bandas_necesarias = ['red', 'green', 'blue']
    for b in bandas_necesarias:
        if b not in ds:
            print(f"❌ Error: El archivo no contiene la banda '{b}'.")
            return

    print("🎨 Procesando canales RGB y ajustando contraste...")

    # 3. Crear stack RGB
    # Concatenamos las 3 variables en una nueva dimensión 'band'
    rgb = xr.concat([ds['red'], ds['green'], ds['blue']], dim='band')

    # 4. Ajuste de Contraste (Percentiles)
    # Las imágenes satelitales crudas suelen verse oscuras. 
    # Cortamos el 2% más oscuro y el 2% más brillante para estirar el histograma.
    vmin, vmax = np.nanpercentile(rgb, [2, 98])
    rgb_stretched = ((rgb - vmin) / (vmax - vmin)).clip(0, 1)

    # 5. Configurar visualización
    plt.figure(figsize=(10, 10))
    
    # Transponer a (Y, X, Banda) para que matplotlib lo entienda
    # y plotear ocultando la barra de color
    rgb_stretched.transpose('y', 'x', 'band').plot.imshow(add_colorbar=False)
    
    # Quitar ejes y bordes
    plt.axis('off')
    plt.title(f"Vista previa: {filename}", fontsize=10)

    # 6. Guardar
    output_filename = filename.replace(".nc", ".jpg")
    print(f"💾 Guardando imagen como: {output_filename}")
    
    plt.savefig(
        output_filename, 
        dpi=CALIDAD_DPI, 
        bbox_inches='tight', 
        pad_inches=0.1
    )
    plt.close() # Cierra la figura para liberar memoria
    print("✅ ¡Listo!")

# Ejecutar
generar_jpg_desde_nc(INPUT_FILE)

import pystac_client
import stackstac
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DE RADIO
# ==========================================

# 1. Define el CENTRO de tu análisis (Latitud, Longitud)
# Usando el centro aproximado de tu ROI anterior (Texas Panhandle)
CENTRO_LAT = 34.165
CENTRO_LON = -101.835

# 2. Define el RADIO en Kilómetros
# Si pones 5, cubrirá 5km a la redonda (un cuadro de 10x10km aprox)
RADIO_KM = 15  

# 3. Carpeta en donde guardar
OUTPUT_DIR = "images_tests"

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
ASSET_NAMES = ["blue", "green", "red", "nir"]
FECHA_RANGO = "2024-05-01/2024-05-31"

def calcular_bbox_desde_radio(lat_centro, lon_centro, radio_km):
    """
    Convierte un punto central y un radio en km a un Bounding Box [min_x, min_y, max_x, max_y].
    Considera la curvatura de la tierra para la longitud.
    """
    # Aproximación: 1 grado de latitud ~= 111.32 km
    lat_degree_dist = 111.32
    
    # Aproximación: 1 grado de longitud cambia según la latitud
    # 111.32 * cos(latitud en radianes)
    lon_degree_dist = 111.32 * math.cos(math.radians(lat_centro))

    lat_delta = radio_km / lat_degree_dist
    lon_delta = radio_km / lon_degree_dist

    min_y = lat_centro - lat_delta
    max_y = lat_centro + lat_delta
    min_x = lon_centro - lon_delta
    max_x = lon_centro + lon_delta

    bbox = [min_x, min_y, max_x, max_y]
    
    print(f"\n--- CÁLCULO DE ÁREA ---")
    print(f"Centro: {lat_centro}, {lon_centro}")
    print(f"Radio: {radio_km} km")
    print(f"Ancho aprox (Lon): {lon_delta*2:.4f} grados")
    print(f"Alto aprox (Lat): {lat_delta*2:.4f} grados")
    print(f"BBOX Resultante: {bbox}")
    print("-----------------------\n")
    
    return bbox

def get_sentinel_data(bbox):
    print("Conectando con STAC API...")
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=FECHA_RANGO,
        query={"eo:cloud_cover": {"lt": 5}}, # Menos de 5% de nubes
        sortby=[{"field": "properties.datetime", "direction": "desc"}]
    )

    items = list(search.items())
    if not items:
        print("❌ No se encontraron imágenes en este rango/área.")
        return None

    # Tomamos la imagen más reciente
    selected_item = items[0]
    print(f"✅ Imagen encontrada: {selected_item.datetime}")
    print(f"   ID: {selected_item.id}")
    
    # Stackstac lazy loading
    # Nota: Si el radio es muy grande (>50km), esto consumirá mucha RAM al hacer .compute()
    data = stackstac.stack(
        selected_item, 
        assets=ASSET_NAMES, 
        bounds_latlon=bbox, 
        epsg=32614  # UTM Zone 14N (Texas)
    )
    return data

def process_and_save(stack, radio_info):
    print("Descargando y procesando píxeles en memoria (esto puede tardar si el área es grande)...")
    
    # 1. Seleccionar primer tiempo y descargar
    da = stack.isel(time=0).compute()

    # 2. Asignar nombres de bandas
    da = da.assign_coords(band=ASSET_NAMES)

    # 3. Convertir a Dataset
    dataset = da.to_dataset(dim='band')

    # 4. LIMPIEZA TOTAL DE ATRIBUTOS (Para evitar errores de NetCDF)
    dataset.attrs = {}
    for var in dataset.data_vars:
        dataset[var].attrs = {}
    for coord in dataset.coords:
        dataset.coords[coord].attrs = {}

    # Eliminar coordenadas extrañas de STAC
    spatial_coords = ['x', 'y']
    coords_to_drop = [c for c in dataset.coords if c not in spatial_coords]
    dataset = dataset.drop_vars(coords_to_drop)

    # 5. Guardar
    # El nombre del archivo incluye el radio para diferenciarlos
    output_filename = f"texas_radio_{radio_info}km.nc"
    dataset.to_netcdf(f'{OUTPUT_DIR}/{output_filename}')
    print(f"💾 NetCDF guardado: {output_filename}")

    return dataset

def visualizar(dataset, radio_info):
    print("Generando previsualización...")
    rgb = xr.concat([dataset['red'], dataset['green'], dataset['blue']], dim='band')

    # Ajuste de contraste (Percentiles 2% y 98%)
    vmin, vmax = np.nanpercentile(rgb, [2, 98])
    rgb_stretched = ((rgb - vmin) / (vmax - vmin)).clip(0, 1)

    plt.figure(figsize=(10, 10))
    rgb_stretched.transpose('y', 'x', 'band').plot.imshow()
    
    plt.title(f"Sentinel-2: Centro Texas + Radio {radio_info}km", fontsize=12)
    plt.axis('off')
    plt.show()

# ==========================================
# EJECUCIÓN
# ==========================================

# 1. Calcular la caja basada en el radio
bbox_calculado = calcular_bbox_desde_radio(CENTRO_LAT, CENTRO_LON, RADIO_KM)

# 2. Obtener datos (Lazy)
stack_data = get_sentinel_data(bbox_calculado)

# 3. Procesar, Guardar y Visualizar
if stack_data is not None:
    ds_final = process_and_save(stack_data, RADIO_KM)
    visualizar(ds_final, RADIO_KM)

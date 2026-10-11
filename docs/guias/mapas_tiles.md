# Mapas de tiles

Abrir `artifacts/maps/index.html` en el navegador. El acceso **Split vigente**
apunta a `vigente/mapa_tiles.html`: TRAIN verde, VAL amarillo, TEST rojo.
Los tiles históricos no asignados están ocultos y se activan con «inactivo».
Filtrar por campaña para distinguir TRAIN/VAL (tres campañas) del TEST (cuatro).
Seleccionar un tile muestra campaña, CRS, dimensiones y raster de origen.
El mapa funciona sin Internet; admite zoom, desplazamiento y descarga GeoJSON.

`artifacts/maps/` es un directorio real del repositorio, versionado con Git,
tanto en floryacy como en el mirror. Incluye `vigente/` y `corridas/<ID>/`.
El índice enlaza ambos. Solo se versionan HTML, GeoJSON y JSON pequeños.
Después de generar mapas nuevos, agregarlos al commit y publicarlo para que
lleguen a otros clones con `git pull`.

Regenerar el mapa después de cambiar el split:

```bash
~/s4a-venv/bin/python scripts/diagnostico/generar_mapa_tiles.py
```

La generación usa `config/geometria_tiles.json`, sin leer rasters. Para incorporar
tiles nuevos, actualizar explícitamente ese catálogo en floryacy y regenerar:

```bash
~/s4a-venv/bin/python scripts/diagnostico/actualizar_geometria_tiles.py --salida config/geometria_tiles.json
~/s4a-venv/bin/python scripts/diagnostico/generar_mapa_tiles.py
```

La actualización abre solo las cabeceras del composite del primer mes de cada
campaña; falla ante fuentes faltantes. No mueve ni lee los píxeles. El catálogo
es metadato pequeño versionable. No certifica que todos los meses compartan grilla.

Cada nueva corrida del lanzador guarda `mapas/mapa_tiles.html`, `tiles.geojson`
y `mapa.json`, generados desde su copia de `configuracion/`. No se regeneran las
corridas históricas. La firma SHA256 identifica el split y el catálogo usados.
**Es el split configurado al inicio**, no prueba de que un comando arbitrario
haya procesado todos sus tiles. Consultar comando, entradas y salidas de `run.json`.
Los contornos son las grillas completas de los composites: no muestran parches
válidos, cobertura de clases ni máscaras de exclusión científica del TEST.

Cada corrida publica también su mapa en `artifacts/maps/corridas/<ID>/`, dentro
del repo, y actualiza el índice. La copia en la carpeta de registros de /mnt se
conserva como evidencia. Si cambia el split, regenerar «vigente»; los mapas de
corridas conservan su configuración original. Los mapas generados quedan
pendientes de `git add`, commit y push; ejecutar una corrida no crea commits.

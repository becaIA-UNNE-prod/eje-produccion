# Inferencia raster

raster.py expone normalizar e inferir_tile. Recibe un modelo ya construido,
fuentes mensuales ordenadas, destino y dispositivo. No carga checkpoints,
elige campañas ni define el split. Conserva bloques de 512 por defecto,
normalización /10000 con paso por float16, tratamiento NoData, padding de
bordes y publicación del TIFF solo al completar la inferencia.

Los consumidores nuevos importan desde src.inferencia.raster.
postprocesar_multicampana.py reexporta los nombres anteriores.

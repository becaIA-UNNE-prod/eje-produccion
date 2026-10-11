# Inferencia raster

raster.py expone normalizar e inferir_tile. Recibe un modelo ya construido,
fuentes mensuales ordenadas, destino y dispositivo. No carga checkpoints,
elige campañas ni define el split. Conserva bloques de 512 por defecto,
normalización /10000 con paso por float16, tratamiento NoData, padding de
bordes y publicación del TIFF solo al completar la inferencia.

Los consumidores nuevos importan desde src.inferencia.raster.
src/historico/postprocesar_multicampana.py y el wrapper anterior reexportan
los nombres compatibles; el dispatcher vigente carga su evaluador al ejecutar.

## Comando de inferencia de tile

`python -m src.inferencia.inferir_tile TILE --modelo unet` desde la raíz
conserva el comando 2. Expone crear_parser, ejecutar_inferencia(args), main(argv)
y el nombre anterior generar_mapa_clasificacion(argv). La ruta src/2_inferencia.py
delega y reexporta; funciona por ruta absoluta desde otro directorio.
Importar no parsea argumentos, carga pesos, abre rasters ni consulta GPU.

Default: siete meses oct–abr de 2019/2020, 28 canales y cinco clases.
--meses conserva orden y calcula canales como cuatro por mes; --dir-composites,
--dir-exp y --dir-salida permiten usar otras entradas explícitas. Los pesos
deben corresponder a esos canales/meses. Soporta U-Net, TransUNet, UTAE (T=meses)
y FCRTransUNet. Salida prediccion_TILE.tif para U-Net y sufijo de modelo para
las demás. --dir-salida prevalece sobre LEGACY_PREDICTIONS/CORDOBA_RUN_DIR.

Este comando mantiene normalización float32 /10000, bloques512, padding edge
hasta múltiplos de16, recorte de bordes y NoData cuando todos los canales son0.
Escribe directamente el TIFF y admite CPU. No se sustituye por raster.inferir_tile:
ese componente usa paso por float16, lotes, otras verificaciones y publicación
atómica. No cambiar esas diferencias como efecto de organizar archivos.
El comando no aplica máscaras ni genera métricas del TEST canónico; para ello
usar src.evaluacion.test_actual y la guía de TEST vigente.

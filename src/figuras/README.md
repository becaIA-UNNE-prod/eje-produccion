# Figuras reutilizables

- comunes.py: colores, nombres, matrices de confusión y curvas de entrenamiento.
- test_actual.py: comparación de un modelo y comparación entre modelos del TEST.
  Recibe explícitamente la ruta RGB, máscaras y predicciones.
- historico.py: comparación espacial histórica; recibe máscaras y split explícitos.

Estos módulos no importan scripts de src, no eligen checkpoints, no hacen
inferencia ni calculan métricas científicas sobre las imágenes reducidas.
Los scripts conservan wrappers/imports compatibles y coordinan las salidas.
Las gráficas de resumen ligadas a cada protocolo todavía están en los comandos.

## Comandos generales

- `python -m src.figuras.curvas_entrenamiento --exp EXP`: curvas desde historial.npy.
- `python -m src.figuras.visualizar_inferencia MAPA.tif --salida FIGURA.png`:
  visualiza clases 0–4 con reducción nearest y la paleta existente.

Las rutas anteriores 4 y 7 conservan compatibilidad. Estos comandos no definen
el TEST ni eligen modelos; las curvas y figuras son pasos vigentes del proyecto.
La representación raster conserva su salida por defecto en la corrida, según
utils.rutas. Explicitar --salida permite reutilizarla en otra corrida.

## Comparación visual de parches NPZ

`python -m src.figuras.comparacion_parches --exp EXP --tile TILE --dir_train DIR`
conserva la entrada anterior 6. Usa best_model.pth, modelos de 28 canales y
parches de 256; forma el mosaico con los primeros floor(sqrt(N))² parches,
ignorando el resto. Con dataset_T conserva el remapeo anterior, independiente
del argumento de clases. El mapa de errores compara solo etiquetas >1.
Salida: comparacion_TILE.png en EXP. Representa parches, no la grilla completa;
no equivale a las figuras del protocolo TEST de mapas y 24 canales.

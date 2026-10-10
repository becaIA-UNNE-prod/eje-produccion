# Entrenamiento

`calcular_pesos.py` cuenta clases TRAIN y emite pesos sugeridos y PESOS_ARGS.
Desde la raíz: `python -m src.entrenamiento.calcular_pesos --dirs DIR [DIR ...]`.
Con --dirs usa TRAIN del split común y rechaza faltantes. Sin --dirs conserva
las rutas y tiles históricos, omitiendo faltantes. No selecciona checkpoints.
`src/0b_calcular_pesos.py` reexporta las funciones y conserva la CLI anterior.
El entrenador permanece en `src/1_train_cordoba.py` hasta su propio traslado.

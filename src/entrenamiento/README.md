# Entrenamiento

`calcular_pesos.py` cuenta clases TRAIN y emite pesos sugeridos y PESOS_ARGS.
Desde la raíz: `python -m src.entrenamiento.calcular_pesos --dirs DIR [DIR ...]`.
Con --dirs usa TRAIN del split común y rechaza faltantes. Sin --dirs conserva
las rutas y tiles históricos, omitiendo faltantes. No selecciona checkpoints.
`src/0b_calcular_pesos.py` reexporta las funciones y conserva la CLI anterior.
El entrenador está en `src/entrenamiento/entrenar_cordoba.py`.
Entrada: `python -m src.entrenamiento.entrenar_cordoba` desde la raíz.
El shell scripts/entrenamiento/entrenar_multicampana_unet.sh EXP conserva
las tres campañas y el control de estado/bloqueo. Las rutas anteriores delegan.
--verificar-datos valida arrays sin entrenar. Parámetros, modos históricos,
semillas y selección de checkpoint con VAL permanecen intactos.

`finetune_manfredi.py` especializa el checkpoint original en 20HMK, con cinco
clases, VAL 0.15 y seed42. Ejecutar `python -m src.entrenamiento.finetune_manfredi`
desde la raíz; `--help` describe la entrada sin entrenar. Conserva configuración
fija, loaders, pérdida CE/Dice, optimizador y salidas externas originales.
Importarlo no crea DIR_EXP ni fija semillas. La entrada 1b reexporta y delega.

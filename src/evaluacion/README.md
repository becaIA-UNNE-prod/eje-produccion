# Métricas de evaluación

metricas.py contiene tres operaciones reutilizables:

- confusion_remapeada(pred, mask): máscaras ya expresadas en clases 0..4,
  como las del TEST vigente.
- confusion_mnc(pred_path, mask_path, remap, year): etiquetas crudas MNC,
  con remapeo y verificación histórica de códigos 2020/2021.
- metricas_confusion(cm): accuracy, precision, recall, F1 e IoU a partir
  de una matriz de cinco clases; filas=referencia, columnas=predicción.

Referencia 0 se ignora. Predicción 0 donde la referencia es válida cuenta
como error. F1 macro y mIoU promedian las clases con referencia presente.
Las funciones no eligen split, modelos ni checkpoints, y no generan figuras.
No convertir una máscara ya remapeada como si fueran códigos crudos MNC.

Se conservan imports anteriores desde utils.multicampana y los scripts.
Los protocolos y las figuras siguen en sus comandos hasta el siguiente paso.

## Matriz de confusión de parches NPZ

`python -m src.evaluacion.matriz_confusion_parches --exp EXP --tile TILE --dir_train DIR`
conserva la entrada anterior 5. Usa best_model.pth y modelos de 28 canales,
procesa parches en lotes de 16 y permite los esquemas anteriores de clases.
Referencia 0 se ignora; predicción 0 cuenta como error aunque no aparezca como
columna de la figura. dataset_T se remapea únicamente al pedir seis clases.
Salida: matriz_confusion_TILE.png dentro de EXP. No evalúa mapas TEST vigentes.

## TEST vigente de mapas

`python -m src.evaluacion.test_actual --exp EXP --salida CORRIDA [--verificar]`
es la entrada canónica de mapas para 20HNJ/20HNH, 24 canales y cinco clases.
Mantiene selección previa de checkpoint con VAL, preflight, firmas y métricas
completas/independientes. Consultar docs/guias/test_actual.md para precondiciones.
La ruta src/evaluar_test_actual.py conserva compatibilidad. --verificar exige
checkpoints y preparación compatibles; omitirlo ejecuta inferencia y escribe reportes.

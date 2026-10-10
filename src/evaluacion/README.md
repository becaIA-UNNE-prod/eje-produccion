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

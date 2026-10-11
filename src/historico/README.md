# Protocolos específicos anteriores

Estos módulos conservan ventanas, splits, remapeos y salidas anteriores.
No redefinen TEST vigente 20HNJ/20HNH ni eligen pesos con TEST.

- `python -m src.historico.postprocesar_multicampana --exp EXP --salida CORRIDA`
  conserva la evaluación de mapas completos de tres campañas, comparación con
  modelos base, TRAIN/VAL descriptivos y split anterior de utils.multicampana.
  --verificar realiza su preflight de configuración, checkpoints y grillas.
  No equivale al preflight del TEST vigente. --base mantiene su ruta original.
- `python -m src.historico.evaluar_ano_no_visto --exp EXP --salida CORRIDA`
  conserva 2021/2022, tiles20HLK/20HLJ/20HKJ/20HMH, checksum MNC fijo,
  checkpoint congelado, máscaras crudas compartidas y comparación de tres modelos.
  Actualiza LEGENDS['2021_2022'] en el mismo diccionario usado por las figuras.

Ambos tienen crear_parser, main(argv) y una función de ejecución con args.
Importar no abre rasters/pesos, escribe máscaras ni consulta GPU. ROOT sigue
siendo la raíz del repo, no src. Sus entradas anteriores Python reexportan;
el main importado del wrapper postprocesar_multicampana conserva flujo anterior,
pero su CLI deriva a TEST vigente por defecto y acepta --historico.

Los datos/resultados mantienen ubicaciones acreditadas externas; no ejecutar
estos protocolos para validar traslados. Las figuras de resumen y comparación
están en src.figuras.historico; la inferencia reusable sigue en src.inferencia.raster.

`python -m src.historico.verificar_pipeline --etapa ETAPA --tiles_train TILE`
conserva la auditoría anterior 2019/2020, octubre–abril, NPZ, 28 canales y
seis clases. No sustituye el preflight vigente. La entrada 0 reexporta/delega;
importar no audita entradas. Revisar rutas originales antes de ejecutarlo.

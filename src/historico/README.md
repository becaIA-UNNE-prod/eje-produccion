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

## Comparaciones anteriores de modelos y mapas

Desde la raíz, ejecutar con `python -m`:

- `src.historico.comparar_modelos`: curvas de cuatro experimentos y mosaico de
  los primeros floor(sqrt(N))² parches NPZ de 256, modelos de 28 canales y cinco
  clases por defecto. Tile original 20HLJ, prefijo dataset_, calendario oct–abr.
  Conserva `--salida_dir` y los nombres comparacion_curvas/comparacion_mapa.
- `src.historico.comparar_inferencia_etiqueta`: mapa raster y etiqueta MNC cruda,
  remapeo al vuelo 0–4, nearest y límite2048. Defaults 20JML/fcrtransunet.
  Aciertos/errores y accuracy de la figura solo sobre target>1 (cultivos).
- `src.historico.metricas_test`: confusión por bloques1024 de mapas completos,
  defaults 20HLJ/20HLK/fcrtransunet; precisión, recall, F1, IoU y stdout. Ignora
  referencia0 y cuenta predicción0 en confusión y recall/F1/IoU; conserva la
  accuracy impresa calculada con el subbloque1–4, que excluye predicción0.

No definen ni evalúan el TEST vigente. Los mapas de 9/10 proceden de inferencia
previa y no ejecutan modelos; el 8 usa parches NPZ y checkpoints originales.
Las entradas 8/9/10 en src reexportan y delegan. Todos tienen crear_parser/main;
importar no lee datos/pesos, consulta GPU, crea salidas ni procesa argumentos.
Salidas explícitas prevalecen sobre CORDOBA_RUN_DIR y el destino legacy
establecido en utils.rutas. No lanzar inferencias reales para probar traslados.

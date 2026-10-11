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

## Evaluación MNC y comparación de variantes

`python -m src.historico.evaluar_contra_mnc --leyenda 1819|2020 --dir-pred PRED
--dir-salida SALIDA` conserva TRAIN/VAL y TEST anterior20HLK/20HLJ. Remapea
MNC según campaña, exige grillas iguales y evalúa solo referencia>0 y pred>0.
Guarda confusiones NPY, matrices PNG, metricas.csv y resumen.json por tile/rol.
No equivale al TEST vigente ni a los comandos que cuentan pred0 como error.

`python -m src.historico.comparar_con_1718 --comparacion 1718|scl` consume
esos resúmenes y curvas originales. 1718 compara base2019/2020 con +2017/2018
Nivel3; scl compara composites02c/02d, sin curvas de entrenamiento. Conserva
rutas externas, grupos/tiles, factor8 nearest, acuerdo sobre ref>0/pred>0,
blanco sin dato, barras/kappa/curvas/mapas y CSV delta segunda menos primera.
--dir-salida explícita prevalece sobre el default de la variante en Y3.

Ambos tienen crear_parser/ejecución/main, wrappers compatibles y exports.
Importar no evalúa datos, escribe resultados ni cambia rcParams de Matplotlib.
La matriz del 11 y helpers lectura/RGB/estilo del 12 viven en src.figuras;
las figuras ligadas a VARIANTES/EVALS siguen con la coordinación histórica.
El estilo se aplica en cada gráfico y se restaura al salir.

## Shells de coordinación anteriores

Implementaciones en `scripts/historico/`, desde la raíz:

- `bash scripts/historico/pipeline_dicmay.sh`: preparación2019/2020,
  entrenamiento base e inferencia2018/2019. Conserva MODELOS y PY.
- `bash scripts/historico/pipeline_dicmay_1718.sh`: incorpora TRAIN2017/2018
  Nivel3, espera descargas anteriores, prepara SCL/arrays/pesos, entrena,
  infiere y compara las campañas anteriores.
- `bash scripts/historico/evaluar_dicmay.sh`: inferencia2019/2020 en VAL/TEST
  anterior y evaluación MNC de ambas campañas. ESPERAR_PID espera al proceso
  y exige LISTO en el log antes de continuar.
- `bash scripts/historico/preparar_ano_no_visto.sh SALIDA EXP`: prepara2021/2022
  y evalúa pesos congelados. SALIDA debe existir; conserva posiciones1/2,
  lock preparación fd8, PID/trap/exitcode/fin, espera MNC y lock GPU fd9.

Las rutas antiguas en src son wrappers ejecutables. No ejecutar estos flujos
para comprobar un traslado: pueden descargar, entrenar e inferir. Las pruebas
usan réplicas en /tmp, comandos simulados y locks de archivos temporales.

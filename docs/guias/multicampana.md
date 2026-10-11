# Preparación y entrenamiento multicampaña

Referencia: `floryacy:~/cordoba_ia_unne`. Entorno: `~/s4a-venv/bin/python`.
Almacenamiento y lanzamientos: [CONTRIBUTING](../../CONTRIBUTING.md).

## Procedimiento para correr el flujo

1. Conectarse con `ssh floryacy` y hacer `cd ~/cordoba_ia_unne`.
   Leer CONTRIBUTING, revisar `git status --short`, almacenamiento y split.
   Usar `~/s4a-venv/bin/python`; conservar los cambios pendientes.
2. Identificar las entradas existentes de las campañas de la tabla siguiente:
   fuentes, composites, máscaras y arrays. Revisar sus procedencias y grillas.
   No descargar ni reconstruir datos ya acreditados por el solo hecho de correr
   el proyecto. Si faltan entradas, revisar los comandos de preparación de la
   campaña correspondiente y sus salidas antes de ejecutarlos.
3. Si las entradas y arrays existen, ejecutar las verificaciones de la sección
   «Verificar antes de preparar o entrenar». Revisar también los JSON de
   procedencia; estas verificaciones no sustituyen su comparación. Si faltan
   entradas, completar la preparación del paso 4 y volver a verificar antes de
   entrenar.
4. Si hay que preparar TRAIN/VAL, revisar `scripts/preparacion/preparar_multicampana.sh` y
   ejecutar esa preparación con log externo según CONTRIBUTING. El proceso
   audita composites, prepara máscaras MNC2021 y genera arrays; requiere
   espacio y no entrena. Reutiliza salidas solo con procedencia compatible.
5. Crear un directorio nuevo de experimento en `/mnt/yacy_*`, verificar recursos
   y ejecutar el shell de entrenamiento de la sección «Entrenar y evaluar».
   Seleccionar el checkpoint con VAL; el cierre multicampaña evalúa TEST vigente.
6. Para evaluar un checkpoint existente sin entrenar, seguir
   [TEST vigente](test_actual.md): primero `--verificar`, luego evaluación en
   una corrida nueva. Usar la preparación acreditada 20HNJ/20HNH; no sustituirla
   por los arrays históricos de 17 tiles.
7. Revisar código de salida, logs, protocolo, métricas y figuras en la corrida
   externa. Distinguir mapas completos del criterio independiente principal.
   Registrar decisiones duraderas en documentación y conservar los artefactos
   en los discos operativos.

### Orden de etapas y alternativas

El orden responde a dependencias de datos, no a la numeración de scripts:

| Etapa | Entrada necesaria | Resultado o siguiente paso |
|---|---|---|
| Fuentes | Tiles, campaña, fechas, bandas y destino externo | Escenas y productos.json |
| Composites | Escenas compatibles y método de la campaña | Seis mosaicos de cuatro bandas por tile para dic–may |
| Máscaras | Referencia de la campaña y grilla Sentinel/composite | Etiquetas crudas alineadas |
| Arrays TRAIN/VAL | Composites, máscaras, leyenda y meses | X/Y y procedencia; verificar antes de entrenar |
| Entrenamiento | Arrays verificados y split vigente | Checkpoint seleccionado con VAL |
| Evaluación | Checkpoint congelado y preparación TEST acreditada | Mapas y métricas completa/independiente |
| Figuras | Predicciones y protocolo de evaluación | Reporte de resultados |

Si ya hay composites, máscaras o arrays acreditados, retomar en la etapa que
falta tras verificar compatibilidad. Evaluar un checkpoint existente no requiere
volver a preparar arrays ni entrenar. TEST de mapas requiere su preparación
específica; no se genera a partir de los arrays TRAIN/VAL.

### Secuencia de fuentes y composites SCL

Para una campaña que realmente necesite preparación, elegir antes tiles, fechas,
meses diciembre–mayo, fuentes y salidas absolutas en /mnt según configuración.
Desde la raíz, usar `~/s4a-venv/bin/python -m` con el módulo de cada fila:

| Paso | Módulo y modo | Parámetros principales / evidencia |
|---|---|---|
| 1. Buscar candidatos | src.datos.descargar_stac buscar | --inicio, --fin, --out, --tiles-file; productos.json |
| 2. Descargar SCL | src.datos.descargar_stac descargar | --out, --bandas SCL |
| 3. Verificar SCL | src.datos.descargar_stac verificar | --out, --bandas SCL; lectura de ventana |
| 4. Seleccionar escenas | src.datos.composites_scl seleccionar | --base, --tiles, --meses; seleccion_scl.json |
| 5. Descargar bandas elegidas | src.datos.descargar_stac descargar | --out, --seleccion |
| 6. Verificar bandas | src.datos.descargar_stac verificar | --out, --seleccion; lectura de ventana |
| 7. Auditar bloques | utils.integridad_raster, invocado por coordinación | Lectura completa; revisar inventario y fallos |
| 8. Componer | src.datos.composites_scl componer | --base, --out, --seleccion, --tiles; TIFF y fechas_usadas.json |
| 9. Auditar composites/grillas | Preparación multicampaña o extensión | Integridad, bandas, meses y alineación de máscaras |

Consultar --help antes de construir el comando concreto. La política
--max-nodata 100 de los shells SCL permite escenas parciales y se conserva;
el default general de STAC sigue siendo 50. No cambiar umbrales por el traslado.
Los wrappers scripts/preparacion/rehacer_2018_2019_scl.sh y scripts/preparacion/descargar_2020_2021_scl.sh
coordinan sus campañas específicas. Revisar sus rutas, locks, logs y qué etapas
incluyen antes de ejecutarlos; conservar procedencia y no relanzar por refactor.
La extensión sigue su [guía específica](test_actual.md): su shell de preparación
incluye evaluación. La auditoría exhaustiva no está incluida en el modo
«verificar» de descarga, que solo lee ventanas de 64×64.

### Otros métodos de composites

Son alternativas de preparación, no etapas posteriores al método SCL:
composites_mensuales calcula la mediana JP2 original; composites_2019_2020
elige la primera fecha JP2 válida y conserva siete meses oct–abr y float32;
composites_cog rellena píxeles sin dato en orden de nubosidad, sin máscara SCL.
Conservar especialmente el procesamiento acreditado 2019/2020: diciembre–abril
proviene del método original y mayo del mosaico COG. Estos módulos no deben
sustituirse por SCL para uniformar el flujo. Sus rutas numeradas siguen compatibles.

### Entradas específicas de máscaras

Para revisar argumentos, usar `~/s4a-venv/bin/python -m` seguido de:
`src.datos.mascaras_mnc_2018_2019`, `src.datos.mascaras_nivel3_2017_2018`
o `src.datos.mascaras_mnc_2020_2021`, con `--help`.
Son alternativas por fuente/campaña, no tres pasos obligatorios consecutivos.
Nivel3 requiere `--ref` con su grilla COG. MNC2021 admite
`--solo-verificar` y exige la política explícita para el código 20 desconocido.
Las rutas numeradas anteriores siguen funcionando.
`src.datos.moldear_mascaras` conserva la prueba original fija sobre JP2;
no es el preparador multicampaña y no debe ejecutarse como paso del flujo vigente.

### Generar arrays después de composites y máscaras

El generador de las campañas TRAIN/VAL es
`src.datos.generar_dataset_cordoba_f16`. La coordinación
`src.datos.preparar_multicampana` lo llama con meses, leyenda, rutas y modo estricto;
preferir el wrapper de preparación para conservar auditorías y procedencia.
Para revisar la entrada específica: `~/s4a-venv/bin/python -m
src.datos.generar_dataset_cordoba_f16 --help` (escribir el comando en una línea).

La forma de un comando explícito, sustituyendo rutas/tiles/meses por los de la
campaña acreditada, es:

```bash
~/s4a-venv/bin/python -m src.datos.generar_dataset_cordoba_f16 \
  --tiles TILE_TRAIN_VAL --meses MES1 MES2 MES3 MES4 MES5 MES6 \
  --leyenda LEYENDA_CAMPANA --dir-composites /mnt/RUTA_COMPOSITES \
  --dir-mascaras /mnt/RUTA_MASCARAS --dir-salida /mnt/RUTA_ARRAYS \
  --workers 4 --estricto
```

Son marcadores, no un comando listo para lanzar. Leyendas: 1819, 2020, 2021
o nivel3; meses en AAAAMM. --ignorar-codigo20 solo aplica a 2021. El default
sin argumentos conserva siete meses oct–abr y no representa el flujo dic–may.
No usarlo por accidente para preparar las campañas de la tabla siguiente.
Las verificaciones posteriores acreditan meses, canales y grillas; comparar
además los JSON de procedencia antes de reutilizar pares existentes.

`src.datos.generar_arrays_originales` conserva el flujo original 0–8 y su
calendario, y `src.datos.generar_dataset_cordoba_2017_f16` conserva oct–abr,
sufijo _2017 y su remapeo. Son alternativas para reproducir esos contratos;
no pasos adicionales necesarios para el TRAIN/VAL dic–may ni el TEST de mapas.

## Entradas

Base existente: `/mnt/yacy_3/prod_bis/ferreyra/`.

| Campaña | Composites bajo la base | Dataset bajo `dataset/` |
|---|---|---|
| 2018/2019 | `composites_2018_2019_scl` | `train_cordoba_dicmay_1819_scl_f16` |
| 2019/2020 | `composites_dicmay_2019_2020` | `train_cordoba_dicmay_f16` |
| 2020/2021 | `composites_2020_2021_scl` | `train_cordoba_dicmay_2021_f16` |

Cada campaña preparada contiene 17 tiles y seis meses diciembre–mayo.
Por mes se usan B02/B03/B04/B08, en ese orden: 24 canales. Parches 256×256,
paso 256; X float16 normalizado /10000, Y uint8 remapeado; carga por mmap.
Las máscaras compartidas viven en la ruta configurada de `/mnt/yacy_1`.
Los JSON de procedencia acreditan entradas, meses, leyenda y normalización.
No reutilizar la variante 1819 anterior a SCL como equivalente a la acreditada.

El split proviene de [configuración](../../config/split_extension_este.json).
El entrenador vigente carga solo TRAIN/VAL en modo multicampaña; los dos tiles
antiguos 20HLK/20HLJ pueden existir en disco sin participar en TEST vigente.
2017/2018 no se agrega automáticamente. La preparación de 17 tiles por campaña
no significa que el entrenador requiera arrays del TEST nuevo.

## Etiquetas y calidad

Esquema: 0=NoData, 1=Fondo, 2=Maíz, 3=Soja, 4=Maní. Consultar
[leyendas](../referencias/leyenda_mnc.md) y el remapeo por campaña en el código.
En 2018/2019 el código crudo 0 es Maíz; en 2019/2020 se excluye como NoData.
La excepción explícita `--ignorar-codigo20` excluye únicamente el código 20
no documentado de 2020/2021. No extenderla a otras campañas o códigos.

2018/2019 y 2020/2021 usan composites SCL. El procesamiento original de
2019/2020 difiere; sus meses diciembre–abril y mayo también difieren en dtype
y descripciones de bandas. Mantener ese contexto al comparar resultados.
Las grillas completas pueden incluir provincias vecinas: no existe un recorte
provincial automático. La [auditoría histórica](../historico/geografia_20261009.md)
explica el alcance y los límites de su verificación por muestras.

## Verificar antes de preparar o entrenar

Desde la raíz del repo:

```bash
~/s4a-venv/bin/python -m src.datos.preparar_multicampana --solo-verificar --ignorar-codigo20
~/s4a-venv/bin/python -m src.entrenamiento.entrenar_cordoba --verificar-datos \
  --campana 2018_2019 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_1819_scl_f16 \
  --campana 2019_2020 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16 \
  --campana 2020_2021 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_2021_f16
```

`--solo-verificar` comprueba entradas y grillas; comprobar por separado los JSON
cuando cambien rutas resueltas o procedencia. `--verificar-datos` valida arrays
sin entrenar. Para regenerar datos, revisar primero `scripts/preparacion/preparar_multicampana.sh`:
reutiliza pares solo con procedencia compatible y aborta ante incompatibilidades.
Consultar auditoría y códigos de salida bajo `logs_multicampana/`; no interpretar
un log antiguo como estado actual. No regenerar lo ya acreditado para probar documentación.

## Entrenar y evaluar

`scripts/entrenamiento/entrenar_multicampana_unet.sh DIRECTORIO_EXPERIMENTO` ejecuta U-Net con
las tres campañas. Crear previamente un directorio nuevo en `/mnt/yacy_*`,
comprobar recursos/procesos y lanzar con log externo según CONTRIBUTING.
El wrapper requiere ese directorio existente para escribir PID y estado.

`--campana` es repetible; `--dir-exp` selecciona la salida.
Los pesos de clase se calculan con TRAIN y el mejor checkpoint se elige con VAL.
`--reanudar` usa un checkpoint completo; `--init-pesos` comienza una nueva
optimización desde pesos, y no equivale a reanudar. Revisar `--help` para opciones
de lectura por bloques y caches; un manifiesto de cache depende de sus rutas reales.
No mover ni eliminar caches acreditadas sin revisar consumidores y procedencia.

El cierre evalúa [TEST vigente](test_actual.md). `--test-historico` reproduce el
flujo anterior explícitamente. Las métricas de parches y mapas completos no son
intercambiables. Los detalles de las corridas de octubre quedan en el
[registro histórico](../historico/experimentos_20261006_10.md).

El entrenador está en src/entrenamiento/entrenar_cordoba.py. Las entradas
anteriores Python y shell de src conservan compatibilidad.

## Ejecutar la coordinación de preparación

Para verificar fuentes y grillas sin generar datos, desde la raíz:

```bash
~/s4a-venv/bin/python -m src.datos.preparar_multicampana \
  --solo-verificar --ignorar-codigo20
```

No comprueba cada JSON de procedencia ni valida arrays X/Y; completar las
verificaciones de esta guía antes de entrenar. Para preparar entradas realmente
faltantes, revisar primero rutas, procedencia y recursos, y usar el shell:

```bash
bash scripts/preparacion/preparar_multicampana.sh --workers 4 --ignorar-codigo20
```

Guardar stdout/stderr en una corrida externa con el lanzador de CONTRIBUTING.
El shell conserva su estado operativo en
/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana: lock exclusivo, PID y exitcode.
Si se invoca el shell con --solo-verificar, también escribe ese estado; para
verificación sin esos archivos, usar la entrada Python de arriba.

La secuencia interna, con los composites ya disponibles, sigue siendo:

1. Verificar los códigos de la fuente MNC2021 y las grillas de las campañas.
2. Exigir la reserva de 210 GiB y auditar completamente los composites.
3. Generar/verificar las máscaras MNC2021 y volver a comprobar alineación.
4. Comprobar procedencia y generar arrays 2020/2021, luego 2018/2019.
5. Validar todas las campañas, incluida 2019/2020 existente, calcular pesos
   con TRAIN y registrar auditoria.json y campanas.json.
6. Ejecutar después la verificación de arrays y, si corresponde, iniciar el
   entrenamiento con su shell separado.

No incorpora automáticamente 2017/2018 o 2021/2022. Las entradas antiguas
src/preparar_multicampana.py y src/preparar_multicampana.sh siguen compatibles.

## Shells específicos: elegir el flujo correspondiente

Desde la raíz del repositorio, con PY apuntando al entorno operativo:

| Entrada en scripts/preparacion/ | Secuencia y alcance |
|---|---|
| descargar_2020_2021_scl.sh | Buscar si falta productos.json → descargar SCL → seleccionar → descargar/verificar bandas → componer; conserva los tiles de tiles.txt y seis meses dic–may |
| rehacer_2018_2019_scl.sh | Descargar SCL → seleccionar → descargar/verificar bandas → componer → inferir modelos base → evaluar contra MNC1819 → gráficos; usa productos.json existente y MODELOS |
| descargar_extension_este.sh DIRECTORIO_LOGS | Cuatro campañas 2018/2019–2021/2022 para 20HNJ/20HNH; buscar si falta → SCL/verificación → selección → bandas/verificación; no compone |
| preparar_extension_este.sh DIRECTORIO_RESULTADOS | Auditar bloques → componer cuatro campañas → preparar máscaras → esperar lock GPU → evaluar U-Net congelada |

La descarga de extensión precede a su preparación cuando faltan fuentes. Revisar
la preparación acreditada antes de ejecutar: estas entradas reutilizan destinos
operativos existentes; no crean automáticamente una corrida científica nueva.
Guardar logs externos según CONTRIBUTING. Los directorios requeridos por los
shells de extensión deben estar en /mnt; locks, PID, exitcodes y reservas de
100 GiB se conservan donde los scripts ya los registraban.

Rehacer 2018/2019 es un flujo mixto de modelos base anterior al TEST canónico.
No ejecutarlo como si solo preparara datos ni sustituirlo por evaluación vigente.
Para evaluar checkpoints existentes en TEST seguir la guía correspondiente.
El shell de preparación de extensión también evalúa; sus comandos Python
de preparación y evaluación están separados como se describe abajo. Las cuatro rutas antiguas en src siguen compatibles.

### Comandos separados de extensión

El shell preparar_extension_este.sh llama a
`python -m src.datos.preparar_extension --salida CORRIDA` para auditar composites
y generar máscaras, y después del lock GPU llama a
`python -m src.evaluacion.evaluar_extension --salida CORRIDA` para evaluar la
U-Net congelada. Sustituir CORRIDA por un destino externo acreditado.
Mantener el mismo directorio entre etapas: contiene protocolo, modelo congelado
y preparación. La entrada src/preparar_evaluar_extension.py conserva --modo
preparar/evaluar. Separar los módulos no cambia el protocolo específico ni
lo convierte en el evaluador canónico del TEST; consultar test_actual.md.

### Inferencia de un tile con entradas explícitas

El comando general está en `src.inferencia.inferir_tile` y se ejecuta por -m.
La entrada src/2_inferencia.py sigue compatible. Los pipelines anteriores y
rehacer_2018_2019_scl.sh usan el módulo nuevo con sus mismos argumentos.
Revisar `python -m src.inferencia.inferir_tile --help`; --meses debe tener
la ventana/orden del entrenamiento y --dir-exp un checkpoint compatible.
Su default conserva siete meses oct–abr/28 canales y no representa el flujo
multicampaña dic–may ni la evaluación vigente de TEST. Para TEST usar
src.evaluacion.test_actual; el comando de tile solo produce un mapa.

### Coordinación de evaluación y protocolos anteriores

`bash scripts/evaluacion/inferencia_multicampana.sh EXP SALIDA` conserva lock
GPU y archivos de estado. SALIDA debe existir; la entrada antigua en src delega.
El shell ejecuta por -m src.evaluacion.postprocesar_multicampana, que deriva
al TEST vigente por defecto. Para argumentos/preflight vigente ver test_actual.md.
El dispatcher Python conserva --historico para el flujo anterior; la implementación
está en src.historico.postprocesar_multicampana. No confundir sus mapas completos
20HLK/20HLJ con las máscaras independientes 20HNJ/20HNH del TEST canónico.
El protocolo temporal anterior 2021/2022 está en src.historico.evaluar_ano_no_visto;
su shell está en scripts/historico/preparar_ano_no_visto.sh y llama al módulo por -m;
la entrada anterior src/preparar_ano_no_visto.sh delega y conserva argumentos.

## Ubicaciones del código y compatibilidad

El [mapa del código](../../src/INDICE.md) enumera implementaciones actuales,
wrappers y contratos. Entradas Python por -m desde la raíz; shells en scripts
por responsabilidad. La raíz de src conserva compatibilidad. Los cuatro shells
de experimentos anteriores están en scripts/historico y no forman parte de
una secuencia obligatoria del TEST vigente. Configurar meses/rutas por campaña
no implica cambiar automáticamente los defaults de otros protocolos.

## Registros de nuevas corridas

Los shells crean automáticamente registros en `artifacts/runs/` de `/mnt/yacy_3`.
Los logs del coordinador y de los pasos se guardan juntos allí; la configuración,
el estado y las rutas de salidas quedan en el manifiesto. Los experimentos y
composites pesados conservan sus ubicaciones; nuevos destinos se eligen según
espacio disponible. Para comandos Python usar el lanzador y declarar sus rutas.
Consultar CONTRIBUTING.md para `--padre`, previsión de espacio y reserva por disco.

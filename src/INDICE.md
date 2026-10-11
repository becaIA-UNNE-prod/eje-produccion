# Mapa del código y entradas de trabajo

Este índice describe los comandos y dependencias existentes antes de reorganizar
`src/`. La referencia es `floryacy:~/cordoba_ia_unne`. Leer primero
[CONTRIBUTING](../CONTRIBUTING.md), la guía de
[multicampaña](../docs/guias/multicampana.md) y la de
[TEST vigente](../docs/guias/test_actual.md).

El TEST vigente es **20HNJ/20HNH**, diciembre–mayo, 24 canales, definido en
`config/split_extension_este.json`. Los números y sufijos de los archivos no
constituyen una secuencia única ni indican por sí solos si un componente sigue
en uso. Este índice no autoriza regenerar datos o relanzar corridas.

## Entradas del flujo vigente

Ejecutar desde la raíz con `~/s4a-venv/bin/python`. Consultar las guías para
argumentos, almacenamiento y precondiciones. Las salidas voluminosas deben vivir
en `/mnt/yacy_*`; no cambiar rutas acreditadas solo para uniformar nombres.

| Necesidad | Entrada | Alcance y precauciones |
|---|---|---|
| Verificar o preparar TRAIN/VAL multicampaña | `datos/preparar_multicampana.py`, shell `../scripts/preparacion/preparar_multicampana.sh` | `--solo-verificar` comprueba entradas y grillas; la procedencia se revisa por separado. La preparación conserva el conjunto histórico de 17 tiles: no define el TEST vigente. |
| Verificar arrays sin entrenar | `1_train_cordoba.py --verificar-datos --campana ...` | Valida los datos multicampaña; no requiere arrays de los nuevos tiles TEST. |
| Entrenar | `1_train_cordoba.py`, wrapper `entrenar_multicampana_unet.sh` | El wrapper selecciona U-Net y tres campañas. Requiere un directorio de experimento existente. El cierre multicampaña evalúa TEST vigente salvo modo histórico explícito. |
| Evaluar checkpoints congelados | `evaluar_test_actual.py` | Entrada canónica del TEST vigente. `--verificar` hace preflight sin inferencia, pero necesita checkpoints compatibles. |
| Ejecutar inferencia/evaluación con lock y estado | `../scripts/evaluacion/inferencia_multicampana.sh` | Invoca por -m el dispatcher src.evaluacion.postprocesar_multicampana; deriva al TEST vigente y conserva lock/PID/exitcode. Entrada anterior compatible. |
| Regenerar figuras de un reporte vigente | `regenerar_figuras_test_actual.py` | Usa protocolo y predicciones existentes; no genera inferencias nuevas. |

## Inventario por responsabilidad

Se incluyen los 49 archivos ejecutables versionados de `src/` presentes al crear
este índice. Los grupos describen su función actual; no implican moverlos todavía.

### Datos y componentes usados por preparación

| Archivos | Función |
|---|---|
| `00_descargar_stac.py` | Búsqueda, descarga y verificación Sentinel por STAC. |
| `01_moldear_mascaras.py` | Reproyección de máscaras del flujo original. |
| `01b_mascaras_mnc_2018_2019.py` | Máscaras MNC 2018/2019 y su remapeo. |
| `01c_mascaras_nivel3_2017_2018.py` | Máscaras Nivel3 2017/2018; importa REMAP_NIVEL3 de `datos/generar_dataset_cordoba_f16.py`. |
| `01d_mascaras_mnc_2020_2021.py` | Máscaras MNC 2020/2021 y verificación de códigos. Usado por preparación multicampaña. |
| `02_composites_mensuales.py`, `02b_composites_2019_2020.py` | Composites del procesamiento original y de 2019/2020. No equivalen automáticamente al método SCL. |
| `02c_composites_cog.py` | Composites desde entradas COG. |
| `02d_composites_scl.py` | Selección de escenas y composición SCL; también usado por recuperación. |
| `03_generar_datasets_npz.py`, `03b_convertir_datasets_a_mmap.py` | Generación original de pares NPY (nombre antiguo NPZ) y conversión de NPZ existentes a mmap. |
| `03c_generar_dataset_cordoba_f16.py` | Generador parametrizable float16; sigue siendo usado por preparación multicampaña. Sus defaults son del flujo original. |
| `03d_generar_dataset_cordoba_2017_f16.py` | Generación de arrays 2017/2018 de la ventana histórica. |
| `0_verificar_pipeline.py` | Verificación del pipeline original; no sustituye el preflight vigente. |
| `0b_calcular_pesos.py` | Cálculo de pesos de clase sobre TRAIN. |
| `preparar_multicampana.py`, `preparar_multicampana.sh` | Auditoría y coordinación de preparación multicampaña. |

### Preparación de la extensión este

| Archivos | Función |
|---|---|
| `descargar_extension_este.sh` | Descarga y selección de fuentes para los tiles de la extensión. |
| `preparar_evaluar_extension.py` | Wrapper compatible: datos/preparar_extension.py y evaluacion/evaluar_extension.py; conserva --modo y exports. El evaluador vigente no depende de él. |
| `preparar_extension_este.sh` | Audita fuentes, compone, prepara máscaras **y luego evalúa una U-Net congelada**. No es una entrada de solo preparación ni sustituye el evaluador canónico. |

Esta preparación todavía contiene tiles, campañas y rutas explícitos en código.
Cambiar el JSON del split no parametriza automáticamente esos comandos. No
relanzarlos sin revisar salidas y metadatos acreditados.

### Entrenamiento, inferencia, evaluación y figuras

| Archivos | Función |
|---|---|
| `1_train_cordoba.py`, `entrenar_multicampana_unet.sh` | Entrenador y wrapper vigente. El entrenador conserva también opciones históricas. |
| `1b_finetune_manfredi.py` | Fine-tuning específico de Manfredi; no es la entrada multicampaña. |
| `2_inferencia.py` | Wrapper de inferencia/inferir_tile.py; CLI y exports compatibles, usado por pipelines anteriores. |
| `4_generar_metricas.py`, `5_generar_matriz_confusion.py`, `6_generar_comparacion_visual.py`, `7_visualizar_inferencia.py` | Métricas y visualizaciones de los flujos originales. |
| `8_comparar_modelos.py`, `9_comparar_inferencia_etiqueta.py`, `10_metricas_test.py` | Comparación y métricas históricas. No adaptar cambiando solo nombres de tiles para evaluar TEST vigente. |
| `11_evaluar_contra_mnc.py`, `12_comparar_con_1718.py` | Evaluación contra MNC y comparación con experimentos 2017/2018. |
| `postprocesar_multicampana.py` | Wrapper con exports perezosos y main importado histórico. CLI deriva al dispatcher evaluacion/postprocesar_multicampana: TEST vigente por defecto, --historico explícito. Implementación anterior en historico; resumen en figuras/historico. |
| `evaluar_test_actual.py`, `inferencia_multicampana.sh` | Entradas compatibles de evaluacion/test_actual.py y scripts/evaluacion/inferencia_multicampana.sh. |
| `regenerar_figuras_test_actual.py` | Figuras desde predicciones acreditadas existentes. |

### Pipelines de campañas y protocolos anteriores

| Archivos | Función |
|---|---|
| `pipeline_dicmay.sh`, `pipeline_dicmay_1718.sh` | Pipelines de experimentos dic–may de una campaña y de la incorporación 2017/2018. |
| `evaluar_dicmay.sh` | Evaluación de esos experimentos con los tiles anteriores de TEST y VAL. |
| `evaluar_ano_no_visto.py`, `preparar_ano_no_visto.sh` | Python compatible de historico/evaluar_ano_no_visto.py; shell llama por -m. Protocolo temporal 2021/2022 con cuatro tiles anteriores; no es TEST vigente 20HNJ/20HNH. |
| `descargar_2020_2021_scl.sh`, `rehacer_2018_2019_scl.sh` | Preparación específica de fuentes/campañas; revisar reutilización, rutas y salidas antes de ejecutar. |

Histórico describe aquí un protocolo o configuración anterior, no código
descartable. Algunos componentes siguen sirviendo para reproducir o preparar datos.

### Mantenimiento operativo

| Archivos | Función |
|---|---|
| `recuperar_multicampana.py` | Reconstrucción de composites corruptos y continuación de preparación. Consume código de preparación y composición. |
| `mejorar_calidad_extension.py`, `recuperar_extension_este.sh` | Reparación específica de dos composites de la extensión; el wrapper continúa con preparación y evaluación. |
| `optimizar_lectura_multicampana.py`, `optimizar_lectura_multicampana.sh` | Preparación de caches y gestión de un entrenamiento existente. Puede detener/reanudar procesos; no es un verificador de solo lectura. |

## Componentes compartidos extraídos

- [Inferencia raster](inferencia/README.md): normalización e inferencia por bloques.
- [Métricas](evaluacion/README.md): matrices para etiquetas crudas o remapeadas y métricas comunes.

Los evaluadores y el entrenador importan estos módulos directamente. Las rutas
anteriores reexportan funciones para conservar compatibilidad. Las [leyendas](datos/README.md) y las [figuras compartidas](figuras/README.md)
también tienen módulos independientes. La coordinación y algunas gráficas
de resumen específicas del protocolo todavía permanecen en los scripts.

## Dependencias que condicionan la reorganización

El siguiente mapa conserva las relaciones entre comandos y las dependencias de
figuras/configuración que todavía quedan. Inferencia y métricas ya se importan
desde los módulos compartidos indicados arriba:

```text
1_train_cordoba.py
  └─ subprocess → src.evaluacion.test_actual (cierre multicampaña vigente)

scripts/evaluacion/inferencia_multicampana.sh
  └─ src.evaluacion.postprocesar_multicampana (dispatcher)
       ├─ por defecto → src.evaluacion.test_actual.main
       └─ --historico → src.historico.postprocesar_multicampana.main

evaluar_test_actual.py y regenerar_figuras_test_actual.py
  └─ módulos de datos, figuras, inferencia, evaluación y soporte
     (ya no importan preparar_evaluar_extension ni postprocesar_multicampana)

preparar_multicampana.py
  ├─ src.datos.campanas (catálogo de configuración)
  ├─ importa src.datos.mascaras_mnc_2020_2021
  └─ ejecuta generadores mediante subprocess

recuperar_multicampana.py
  ├─ importa src.datos.preparar_multicampana al ejecutar
  └─ importa src.datos.composites_scl al ejecutar
```

También hay consumidores en shell, documentación y `test/` que fijan nombres o
rutas. Las pruebas cargan algunos scripts con `importlib` o agregando `src` a
`sys.path`. Antes de cualquier traslado hay que buscar todos esos consumidores.

Las arquitecturas están en [model/](../model/README.md), con un archivo por
modelo y bloques compartidos. utils/model.py conserva imports anteriores.

El código compartido de modelos, datos, inferencia, métricas y figuras tiene
módulos propios; algunas variantes históricas siguen dentro de sus scripts.
El entrenador y la regeneración de figuras separan parser y ejecución en
main(); importarlos no procesa argumentos, fija semillas ni escribe salidas.
Los comandos históricos de curvas, matriz de confusión y comparación visual
(4, 5, 6, 7, 9 y 10) también exponen crear_parser() y main(argv=None), sin leer datos ni
cargar modelos al importar. Conservan sus canales, remapeos y cálculos originales.
Los comandos de pesos y conversión mmap también separan crear_parser() y
main(argv=None): importarlos no cuenta etiquetas ni convierte o borra archivos.
Conservan el modo histórico de pesos, TRAIN con --dirs y las verificaciones de
conversión. mejorar_calidad_extension.py también separa parser, main y mantenimiento
por compuesto con rutas explícitas; importarlo no descarga ni reemplaza archivos.
Sus respaldos y el rechazo de cobertura empeorada se verifican con fixtures y mocks.
Los tres efectos operativos confirmados al importar quedaron resueltos; esto no
constituye una auditoría completa de todos los imports. Otros ya protegen su ejecución con
__main__, aunque no todos exponen una función main().
Por eso no se deben importar indiscriminadamente para hacer un inventario.

## Secuencia de reorganización propuesta

1. Conservar este mapa y los contratos científicos como punto de partida.
2. Extraer inferencia, leyendas, métricas y figuras compartidas de los scripts,
   con imports estables y sin cambiar resultados ni protocolos. Mantener las
   entradas existentes durante la transición.
3. Separar argumentos y ejecución en `main()` de la implementación reutilizable.
4. Agrupar módulos por dominio y separar wrappers, mantenimiento y protocolos
   históricos; actualizar consumidores y conservar compatibilidad donde haga falta.
5. Centralizar campañas, ventanas y fuentes que hoy están repetidas, preservando
   las diferencias de leyenda, normalización y procedencia.

Cada paso requiere un diff acotado y las pruebas pertinentes. Un movimiento de
archivos no justifica relanzar entrenamiento o inferencia completos ni cambiar
dependencias del entorno operativo.

## Estructura objetivo y plan de traslado

Este es el diseño propuesto para los próximos pasos; las rutas de las tablas
anteriores siguen siendo las actuales hasta migrar cada grupo. Cada traslado
se entrega por separado, conservando contratos y consumidores. Los nombres de
módulos nuevos describen su responsabilidad y no llevan numeración de pipeline.

```text
model/                        arquitecturas y bloques (ya organizado)
src/
├── datos/                    fuentes, máscaras, composites, arrays y preparación
├── entrenamiento/            entrenador y cálculo de pesos TRAIN
├── inferencia/               inferencia raster y comando de tile
├── evaluacion/               métricas y protocolo TEST vigente
├── figuras/                  figuras compartidas y regeneración vigente
├── mantenimiento/            recuperación, calidad y optimización operativa
├── historico/                comandos de protocolos anteriores o específicos
└── soporte/                  operaciones comunes sin protocolo científico
scripts/
├── preparacion/              wrappers de preparación por campaña/extensión
├── entrenamiento/            wrapper multicampaña
├── evaluacion/               wrapper de inferencia/evaluación vigente
├── mantenimiento/            wrappers de recuperación y optimización
└── historico/                pipelines y evaluación de protocolos anteriores
```

Cada dominio contiene implementación importable y, cuando haga falta, un módulo
de entrada con `crear_parser()` y `main(argv=None)`. No crear un segundo árbol
de implementaciones en `cli/`. Mantener los módulos compartidos existentes.
`utils/` conserva compatibilidad; su reorganización completa queda fuera de este
paso. Las arquitecturas permanecen en `model/`.

### Destinos de Python

Los orígenes siguientes son relativos a la raíz actual de `src/`; los destinos
son relativos a `src/`. Son destinos previstos, no archivos creados todavía.

| Origen | Destino previsto |
|---|---|
| `00_descargar_stac.py` | `datos/descargar_stac.py` |
| `01_moldear_mascaras.py` | `datos/moldear_mascaras.py` |
| `01b_mascaras_mnc_2018_2019.py` | `datos/mascaras_mnc_2018_2019.py` |
| `01c_mascaras_nivel3_2017_2018.py` | `datos/mascaras_nivel3_2017_2018.py` |
| `01d_mascaras_mnc_2020_2021.py` | `datos/mascaras_mnc_2020_2021.py` |
| `02_composites_mensuales.py` | `datos/composites_mensuales.py` |
| `02b_composites_2019_2020.py` | `datos/composites_2019_2020.py` |
| `02c_composites_cog.py` | `datos/composites_cog.py` |
| `02d_composites_scl.py` | `datos/composites_scl.py` |
| `03_generar_datasets_npz.py` | `datos/generar_arrays_originales.py` |
| `03b_convertir_datasets_a_mmap.py` | `datos/convertir_datasets_a_mmap.py` |
| `03c_generar_dataset_cordoba_f16.py` | `datos/generar_dataset_cordoba_f16.py` |
| `03d_generar_dataset_cordoba_2017_f16.py` | `datos/generar_dataset_cordoba_2017_f16.py` |
| `preparar_multicampana.py` | `datos/preparar_multicampana.py` |
| `0b_calcular_pesos.py` | `entrenamiento/calcular_pesos.py` |
| `1_train_cordoba.py` | `entrenamiento/entrenar_cordoba.py` |
| `2_inferencia.py` | `inferencia/inferir_tile.py` |
| `evaluar_test_actual.py` | `evaluacion/test_actual.py` |
| `regenerar_figuras_test_actual.py` | `figuras/regenerar_test_actual.py` |
| `mejorar_calidad_extension.py` | `mantenimiento/mejorar_calidad_extension.py` |
| `recuperar_multicampana.py` | `mantenimiento/recuperar_multicampana.py` |
| `optimizar_lectura_multicampana.py` | `mantenimiento/optimizar_lectura_multicampana.py` |
| `0_verificar_pipeline.py` | `historico/verificar_pipeline.py` |
| `1b_finetune_manfredi.py` | `entrenamiento/finetune_manfredi.py` |
| `4_generar_metricas.py` | `figuras/curvas_entrenamiento.py` |
| `5_generar_matriz_confusion.py` | `evaluacion/matriz_confusion_parches.py` |
| `6_generar_comparacion_visual.py` | `figuras/comparacion_parches.py` |
| `7_visualizar_inferencia.py` | `figuras/visualizar_inferencia.py` |
| `8_comparar_modelos.py` | `historico/comparar_modelos.py` |
| `9_comparar_inferencia_etiqueta.py` | `historico/comparar_inferencia_etiqueta.py` |
| `10_metricas_test.py` | `historico/metricas_test.py` |
| `11_evaluar_contra_mnc.py` | `historico/evaluar_contra_mnc.py` |
| `12_comparar_con_1718.py` | `historico/comparar_con_1718.py` |
| `evaluar_ano_no_visto.py` | `historico/evaluar_ano_no_visto.py` |

La extensión y el postprocesador ya están separados; se conservan estas entradas:

- `preparar_evaluar_extension.py`: preparación en `datos/preparar_extension.py`
  y evaluación específica en `evaluacion/evaluar_extension.py`. Conservar su
  orquestación CLI mediante compatibilidad; no equiparar este protocolo al
  evaluador canónico ni duplicar helpers ya extraídos.
- `postprocesar_multicampana.py`: despacho implementado en evaluacion hacia
  `evaluacion/test_actual.py`; protocolo anterior en historico/postprocesar_multicampana.py.
  Resumen gráfico en figuras/historico, exports compatibles y consumidor temporal
  evaluar_ano_no_visto también trasladado a historico.

Curvas y visualización general pertenecen a `figuras/`. Los comandos 5 y 6
se ubicarán por responsabilidad en evaluación/figuras, describiendo su contrato
actual de NPZ, parches y 28 canales; no se consideran evaluadores del TEST
vigente de 24 canales. Los protocolos concretos anteriores permanecen en
`historico/`. Clasificar por responsabilidad y contrato, no por numeración.
Las variantes de preparación permanecen en `datos/` con campañas explícitas,
sin consolidar meses, leyendas o métodos diferentes.


### Destinos de shell

Conservar los nombres de archivo al migrar cada wrapper:

| Directorio objetivo | Archivos actuales en `src/` |
|---|---|
| `scripts/preparacion/` | `preparar_multicampana.sh`, `descargar_extension_este.sh`, `preparar_extension_este.sh`, `descargar_2020_2021_scl.sh`, `rehacer_2018_2019_scl.sh` |
| `scripts/entrenamiento/` | `entrenar_multicampana_unet.sh` |
| `scripts/evaluacion/` | `inferencia_multicampana.sh` |
| `scripts/mantenimiento/` | `recuperar_extension_este.sh`, `optimizar_lectura_multicampana.sh` |
| `scripts/historico/` | `pipeline_dicmay.sh`, `pipeline_dicmay_1718.sh`, `evaluar_dicmay.sh`, `preparar_ano_no_visto.sh` |

`preparar_extension_este.sh` debe seguir documentado como preparación **más
evaluación**; la carpeta no modifica su comportamiento. El wrapper de
recuperación también continúa hacia preparación/evaluación.

### Compatibilidad y orden de implementación

1. Migrar pesos TRAIN y conversión mmap: módulos pequeños ya separables y con
   pruebas sintéticas. Es el primer grupo recomendado.
2. Migrar mantenimiento de calidad y su wrapper; luego recuperación y
   optimización tras revisar sus cargas por ruta y gestión de procesos.
3. Migrar preparación por grupos: máscaras compartidas, STAC/composites,
   generadores y coordinación. Resolver cargas `01c -> 01b`, preparación
   `-> 01d` y recuperación `-> preparación/02d` con imports de módulos.
4. Migrar evaluador vigente/regenerador y entrenador con sus consumidores;
   conservar cierre TEST, lock, estado y comportamiento de wrappers.
5. Separar comandos mixtos y migrar históricos; actualizar el índice para
   mostrar ubicaciones reales y retirar este plan cuando esté implementado.

Al mover una entrada con consumidores, dejar en la ruta anterior un wrapper
delgado que delegue a la implementación única. Preservar los exports importados
por pruebas o comandos, no solamente `main`. Los consumidores del repositorio
se actualizan a las rutas nuevas en el mismo paso; no mantener dos versiones.
El objetivo de la raíz de `src/` es índice, inicialización y esos wrappers de
compatibilidad documentados, sin lógica de negocio duplicada.

Para módulos nuevos, preferir `python -m src.<dominio>.<modulo>` desde la raíz.
Los wrappers anteriores mantienen ejecución directa y resuelven la raíz por
`__file__`, sin depender del directorio actual. Recalcular también la raíz de
shell al cambiar su profundidad. No insertar `src/` globalmente en `sys.path`:
puede colisionar con paquetes de `test/`. Revisar antes de mover scripts que
se relanzan a sí mismos, como optimización de lectura.

En cada grupo, buscar consumidores en código, pruebas, shell, configuración y
documentación; verificar imports, ayuda, salidas sintéticas y rutas. Mantener
relatos históricos con sus rutas fechadas, añadiendo aclaraciones cuando sean
necesarias. No lanzar operaciones ML o mantenimiento para validar un traslado.

## Traslados implementados

| Implementación actual | Entrada anterior compatible |
|---|---|
| `entrenamiento/calcular_pesos.py` | `0b_calcular_pesos.py` |
| `datos/convertir_datasets_a_mmap.py` | `03b_convertir_datasets_a_mmap.py` |

Ejecutar desde la raíz con `python -m src.entrenamiento.calcular_pesos` o
`python -m src.datos.convertir_datasets_a_mmap`. Los archivos anteriores solo
reexportan funciones y delegan su CLI. Sus imports no leen ni escriben datasets.
El pipeline 1718 y las pruebas de cálculo usan las implementaciones nuevas.
Las referencias de los relatos históricos se conservan como rutas de su época.
El resto de la tabla objetivo sigue pendiente.

### Recomposición de imágenes mensuales trasladada

Implementación: `mantenimiento/mejorar_calidad_extension.py` (20HNJ marzo 2019,
20HNH marzo 2021). Shell: `scripts/mantenimiento/recuperar_extension_este.sh`.
Las rutas anteriores de src delegan. El shell conserva lock, PID, exitcode y
continuación hacia preparación/evaluación; invoca el módulo nuevo desde la raíz.

### Reconstrucción de imágenes corruptas trasladada

`mantenimiento/recuperar_multicampana.py` contiene la implementación;
`recuperar_multicampana.py` delega y conserva la entrada anterior.
Ejecutar desde raíz con `python -m src.mantenimiento.recuperar_multicampana`.
Preparación y compositor SCL se cargan por nombre de módulo al ejecutar,
no por ruta al importar; sus propios traslados siguen pendientes.
Los bloqueos, respaldos, invalidación de arrays y reanudación sin entrenamiento
se conservan. main y reparar aceptan dependencias explícitas para pruebas.

### Preparación de cachés y reanudación trasladadas

Python: `mantenimiento/optimizar_lectura_multicampana.py`.
Shell: `scripts/mantenimiento/optimizar_lectura_multicampana.sh`.
Las rutas anteriores delegan; el subproceso --solo-copiar usa -m desde la raíz,
para no depender de la profundidad del archivo trasladado. El shell conserva
lock, PID, exitcode y argumentos. No ejecutar como diagnóstico: puede detener
un entrenamiento y relanzarlo desde un checkpoint. Los destinos físicos de
cachés existentes, reservas de espacio y parámetros de reanudación se conservan.

### Figuras generales trasladadas

`figuras/curvas_entrenamiento.py` lee historial.npy y grafica Train/Val.
`figuras/visualizar_inferencia.py` representa un GeoTIFF de clases 0–4.
No fijan campañas, modelos o split; mantienen sus formatos y salidas anteriores.
Las entradas 4 y 7 reexportan y delegan. El postprocesador usa la entrada por -m.
Esta reorganización facilita reutilización; parametrizar regiones, bandas,
leyendas y protocolos es trabajo adicional y no consecuencia de mover archivos.

### Evaluación y comparación de parches trasladadas

`evaluacion/matriz_confusion_parches.py` y `figuras/comparacion_parches.py`
implementan las entradas 5 y 6, conservadas como wrappers. Ejecutar por -m
con los argumentos anteriores. Estos comandos usan NPZ, best_model.pth y
modelos de 28 canales; no son el evaluador TEST vigente de mapas/24 canales.
Matriz: ignora referencia 0 y cuenta predicción 0 como error; la figura muestra
solo clases no cero con referencia presente. Comparación: mosaico de los primeros
floor(sqrt(N))² parches de 256, y mapa de errores solo sobre etiquetas >1.
Mantener diferencias de remapeo dataset_T entre ambos hasta una revisión propia.

### Evaluador y figuras del TEST vigente trasladados

Implementaciones: `evaluacion/test_actual.py` y `figuras/regenerar_test_actual.py`.
Las rutas anteriores delegan y reexportan funciones. Entrenador y postprocesador
usan el módulo nuevo. El evaluador expone crear_parser y main(argv=None), sin
alterar el protocolo, preflight, locks, checkpoints, firmas o salidas.
TEST 20HNJ/20HNH, 24 canales y cinco clases; criterios completa/independiente
preservados. Regeneración conserva espera de inferencia y figuras.exitcode.

### Entrenador multicampaña trasladado

Implementación: `entrenamiento/entrenar_cordoba.py`.
Shell: `scripts/entrenamiento/entrenar_multicampana_unet.sh`.
Las entradas anteriores reexportan/delegan. Pipelines y pruebas usan -m;
el coordinador de cachés reconoce la entrada anterior y la nueva por módulo.
Conserva todos los modos, datos, semillas, checkpoints y cierre TEST vigente.

### Máscaras trasladadas

Implementaciones en datos/: moldear_mascaras.py, mascaras_mnc_2018_2019.py,
mascaras_nivel3_2017_2018.py y mascaras_mnc_2020_2021.py. Entradas anteriores
compatibles; preparación multicampaña y pipeline 1718 usan los módulos nuevos.
La dependencia real de Nivel3 es el remapeo de 03c, no 01b; se conserva.

### Fuentes STAC y composites trasladados

Implementaciones: datos/descargar_stac.py, datos/composites_mensuales.py,
datos/composites_2019_2020.py, datos/composites_cog.py y datos/composites_scl.py.
Entradas numeradas compatibles; shells y mantenimiento usan los módulos nuevos.
Recuperación carga composites_scl solo al ejecutar. La espera de descargas del
pipeline 1718 reconoce rutas antiguas y nuevas. Shells conservan su ubicación.
No se unifican los cuatro métodos ni sus ventanas o formatos.

### Generadores de arrays trasladados

Implementaciones: datos/generar_arrays_originales.py,
datos/generar_dataset_cordoba_f16.py y datos/generar_dataset_cordoba_2017_f16.py.
Las entradas 03/03c/03d reexportan/delegan. Preparación y pipelines usan -m;
Nivel3 importa directamente el remapeo compartido del generador parametrizable.
Se preservan calendarios, ventanas, remapeos, normalización, tipos, nombres,
procedencia y comportamiento ante faltantes o salidas existentes.
Coordinación de preparación y sus shells quedan para el paso siguiente.

### Coordinación de preparación multicampaña trasladada

Python: datos/preparar_multicampana.py, con crear_parser/main y exports
compatibles. Shell: scripts/preparacion/preparar_multicampana.sh. Las entradas
anteriores delegan. Recuperación importa la implementación nueva solo al
ejecutar y continúa por el nuevo shell; no se ejecuta mantenimiento al importar.
Locks, estado y logs externos conservan sus rutas. No cambia el orden de
operaciones, procedencia, campañas ni separación entre preparación y entrenamiento.
Los restantes wrappers de preparación y el comando mixto de extensión siguen
pendientes de su propio paso.

### Shells de preparación por campaña/extensión trasladados

En scripts/preparacion/: descargar_2020_2021_scl.sh,
rehacer_2018_2019_scl.sh, descargar_extension_este.sh y preparar_extension_este.sh.
Entradas anteriores compatibles; recuperación de extensión usa el shell nuevo.
Raíz recalculada; rutas físicas, campañas, meses, locks, PID, logs y exitcodes
conservados. Descargar 2020/2021 también compone; descargar extensión selecciona
pero no compone. Rehacer 2018/2019 compone, infiere, evalúa y grafica modelos base:
es un flujo mixto anterior, no solo preparación ni el TEST canónico.
Preparar extensión audita, compone, prepara máscaras y evalúa U-Net congelada.
La separación de preparar_evaluar_extension.py está implementada (ver abajo).

### Preparación y evaluación de extensión separadas

Implementaciones únicas: datos/preparar_extension.py (shapes/write/prepare) y
evaluacion/evaluar_extension.py (figure/evaluate), con crear_parser/main.
El wrapper anterior mantiene --modo y exports; el shell usa -m para ambas etapas.
Conserva campañas, tiles, fuentes, máscaras completas/independientes, SHA256,
firmas, GPU, predicciones, métricas y figuras. No sustituye TEST canónico.
La preparación no importa Torch ni Matplotlib; no se agrega src a sys.path.

### Comando de inferencia de tile trasladado

Implementación: inferencia/inferir_tile.py, con crear_parser, ejecutar_inferencia,
main(argv=None) y generar_mapa_clasificacion compatible. Entrada 2 reexporta/delega.
Los cuatro shells consumidores usan -m. Se conservan modelos, defaults oct–abr,
meses explícitos, normalización float32, bloques512, padding/recorte, NoData,
metadatos y nombres de salida. No se sustituye por raster.py, cuyo contrato
de precisión y publicación del TIFF difiere. No evalúa el TEST canónico.

### Postprocesado, coordinación y evaluación temporal separados

Dispatcher: evaluacion/postprocesar_multicampana.py (vigente por defecto,
--historico explícito). Implementaciones anteriores: historico/postprocesar_multicampana.py
e historico/evaluar_ano_no_visto.py, con parser/main separados. Resumen gráfico:
figuras/historico.resumen_multicampana. Mantienen split anterior, leyendas,
calendarios, modelos, firmas, métricas, figuras y salidas; no se convierten a TEST.
Wrapper postprocesar reexporta perezosamente y mantiene main importado histórico.
Shell: scripts/evaluacion/inferencia_multicampana.sh; anterior delega. Mantiene
lock, PID, timestamps, exitcode y --lock-adquirido. Preparar año no visto llama
al módulo nuevo por -m, conservando la preparación temporal previa.

### Verificador anterior y ajuste fino de Manfredi trasladados

Implementaciones: historico/verificar_pipeline.py y entrenamiento/finetune_manfredi.py.
Las entradas 0 y 1b reexportan y delegan; ejecutar por -m desde la raíz.
El verificador conserva 2019/2020, octubre–abril, NPZ, 28 canales y seis clases;
no sustituye el preflight vigente. Manfredi conserva 20HMK, cinco clases,
checkpoint base, VAL 0.15 y seed42. Sus imports no crean carpetas ni fijan
semillas; esas acciones ocurren al ejecutar finetune(). No cambian cálculos.

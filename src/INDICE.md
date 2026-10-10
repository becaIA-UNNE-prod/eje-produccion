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
| Verificar o preparar TRAIN/VAL multicampaña | `preparar_multicampana.py`, wrapper `preparar_multicampana.sh` | `--solo-verificar` comprueba entradas y grillas; la procedencia se revisa por separado. La preparación conserva el conjunto histórico de 17 tiles: no define el TEST vigente. |
| Verificar arrays sin entrenar | `1_train_cordoba.py --verificar-datos --campana ...` | Valida los datos multicampaña; no requiere arrays de los nuevos tiles TEST. |
| Entrenar | `1_train_cordoba.py`, wrapper `entrenar_multicampana_unet.sh` | El wrapper selecciona U-Net y tres campañas. Requiere un directorio de experimento existente. El cierre multicampaña evalúa TEST vigente salvo modo histórico explícito. |
| Evaluar checkpoints congelados | `evaluar_test_actual.py` | Entrada canónica del TEST vigente. `--verificar` hace preflight sin inferencia, pero necesita checkpoints compatibles. |
| Ejecutar inferencia/evaluación con lock y estado | `inferencia_multicampana.sh` | Invoca `postprocesar_multicampana.py`, cuya ejecución por defecto deriva a `evaluar_test_actual.py`. |
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
| `01c_mascaras_nivel3_2017_2018.py` | Máscaras Nivel3 2017/2018; depende por ruta del archivo `01b`. |
| `01d_mascaras_mnc_2020_2021.py` | Máscaras MNC 2020/2021 y verificación de códigos. Usado por preparación multicampaña. |
| `02_composites_mensuales.py`, `02b_composites_2019_2020.py` | Composites del procesamiento original y de 2019/2020. No equivalen automáticamente al método SCL. |
| `02c_composites_cog.py` | Composites desde entradas COG. |
| `02d_composites_scl.py` | Selección de escenas y composición SCL; también usado por recuperación. |
| `03_generar_datasets_npz.py`, `03b_convertir_datasets_a_mmap.py` | Generación NPZ y conversión histórica a arrays mmap. |
| `03c_generar_dataset_cordoba_f16.py` | Generador parametrizable float16; sigue siendo usado por preparación multicampaña. Sus defaults son del flujo original. |
| `03d_generar_dataset_cordoba_2017_f16.py` | Generación de arrays 2017/2018 de la ventana histórica. |
| `0_verificar_pipeline.py` | Verificación del pipeline original; no sustituye el preflight vigente. |
| `0b_calcular_pesos.py` | Cálculo de pesos de clase sobre TRAIN. |
| `preparar_multicampana.py`, `preparar_multicampana.sh` | Auditoría y coordinación de preparación multicampaña. |

### Preparación de la extensión este

| Archivos | Función |
|---|---|
| `descargar_extension_este.sh` | Descarga y selección de fuentes para los tiles de la extensión. |
| `preparar_evaluar_extension.py` | Preparación de máscaras completas/independientes y evaluación específica de la extensión. Expone funciones consumidas por el evaluador vigente. |
| `preparar_extension_este.sh` | Audita fuentes, compone, prepara máscaras **y luego evalúa una U-Net congelada**. No es una entrada de solo preparación ni sustituye el evaluador canónico. |

Esta preparación todavía contiene tiles, campañas y rutas explícitos en código.
Cambiar el JSON del split no parametriza automáticamente esos comandos. No
relanzarlos sin revisar salidas y metadatos acreditados.

### Entrenamiento, inferencia, evaluación y figuras

| Archivos | Función |
|---|---|
| `1_train_cordoba.py`, `entrenar_multicampana_unet.sh` | Entrenador y wrapper vigente. El entrenador conserva también opciones históricas. |
| `1b_finetune_manfredi.py` | Fine-tuning específico de Manfredi; no es la entrada multicampaña. |
| `2_inferencia.py` | Inferencia parametrizable de un tile, usada por pipelines anteriores. |
| `4_generar_metricas.py`, `5_generar_matriz_confusion.py`, `6_generar_comparacion_visual.py`, `7_visualizar_inferencia.py` | Métricas y visualizaciones de los flujos originales. |
| `8_comparar_modelos.py`, `9_comparar_inferencia_etiqueta.py`, `10_metricas_test.py` | Comparación y métricas históricas. No adaptar cambiando solo nombres de tiles para evaluar TEST vigente. |
| `11_evaluar_contra_mnc.py`, `12_comparar_con_1718.py` | Evaluación contra MNC y comparación con experimentos 2017/2018. |
| `postprocesar_multicampana.py` | Compatibilidad para inferencia y métricas; implementación de figuras; conserva un flujo histórico en `main()`. Como comando deriva al TEST vigente por defecto; `--historico` selecciona el anterior. |
| `evaluar_test_actual.py`, `inferencia_multicampana.sh` | Evaluación canónica y wrapper operativo del TEST vigente. |
| `regenerar_figuras_test_actual.py` | Figuras desde predicciones acreditadas existentes. |

### Pipelines de campañas y protocolos anteriores

| Archivos | Función |
|---|---|
| `pipeline_dicmay.sh`, `pipeline_dicmay_1718.sh` | Pipelines de experimentos dic–may de una campaña y de la incorporación 2017/2018. |
| `evaluar_dicmay.sh` | Evaluación de esos experimentos con los tiles anteriores de TEST y VAL. |
| `evaluar_ano_no_visto.py`, `preparar_ano_no_visto.sh` | Protocolo temporal 2021/2022 con los cuatro tiles espaciales anteriores. No es el protocolo vigente de TEST 20HNJ/20HNH. |
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
  └─ subprocess → evaluar_test_actual.py (cierre multicampaña vigente)

inferencia_multicampana.sh
  └─ postprocesar_multicampana.py como comando
       ├─ por defecto → evaluar_test_actual.main
       └─ --historico → su propio main histórico

evaluar_test_actual.py y regenerar_figuras_test_actual.py
  └─ módulos de datos, figuras, inferencia, evaluación y soporte
     (ya no importan preparar_evaluar_extension ni postprocesar_multicampana)

preparar_multicampana.py
  ├─ src.datos.campanas (catálogo de configuración)
  ├─ carga por ruta 01d_mascaras_mnc_2020_2021.py
  └─ ejecuta generadores mediante subprocess

recuperar_multicampana.py
  ├─ carga por ruta preparar_multicampana.py
  └─ carga por ruta 02d_composites_scl.py
```

También hay consumidores en shell, documentación y `test/` que fijan nombres o
rutas. Las pruebas cargan algunos scripts con `importlib` o agregando `src` a
`sys.path`. Antes de cualquier traslado hay que buscar todos esos consumidores.

Las arquitecturas están en [model/](../model/README.md), con un archivo por
modelo y bloques compartidos. utils/model.py conserva imports anteriores.

`utils/` contiene parte del código compartido, pero no todo: las leyendas
y figuras todavía residen en scripts de `src/`. Varios comandos procesan
argumentos al importar; el entrenador además fija semillas en el nivel superior.
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

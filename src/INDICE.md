# Mapa del código y entradas de trabajo

Este índice describe la estructura implementada. Referencia principal:
`floryacy:~/cordoba_ia_unne`; la copia local es un mirror. Leer
[CONTRIBUTING](../CONTRIBUTING.md), [Multicampaña](../docs/guias/multicampana.md)
y [TEST vigente](../docs/guias/test_actual.md) antes de operar.

El TEST vigente es **20HNJ/20HNH**, diciembre–mayo, 24 canales y cinco clases,
definido en config/split_extension_este.json. El checkpoint se elige con VAL;
la métrica principal excluye grillas completas TRAIN/VAL. Los números de los
wrappers no definen una secuencia de ejecución.

## Estructura y responsabilidades

| Ubicación | Responsabilidad y referencia |
|---|---|
| [model/](../model/README.md) | Una arquitectura por archivo y bloques compartidos; utils/model.py conserva imports |
| [datos/](datos/README.md) | Fuentes, máscaras, composites, arrays, leyendas y coordinación de preparación |
| [entrenamiento/](entrenamiento/README.md) | Entrenador, pesos TRAIN y ajuste fino de Manfredi |
| [inferencia/](inferencia/README.md) | Inferencia raster y comando de tile |
| [evaluacion/](evaluacion/README.md) | Métricas, evaluación de parches, extensión y TEST vigente |
| [figuras/](figuras/README.md) | Curvas, visualización, figuras de evaluación y regeneración |
| [mantenimiento/](mantenimiento/README.md) | Recomposición, recuperación, caches y reanudación |
| [historico/](historico/README.md) | Evaluación y comparación con contratos científicos anteriores |
| soporte/archivos.py | JSON atómico y SHA256, sin definir protocolo científico |
| ../scripts/ | Shells por responsabilidad, lanzador con logs y diagnóstico manual |
| ../utils/ | Infraestructura y compatibilidad; no se reorganizó completamente |

La raíz de src contiene el índice, __init__ y entradas compatibles: la
implementación única está en los dominios. No hay otro árbol cli/. Los módulos
se ejecutan por `python -m src.DOMINIO.MODULO` desde la raíz; los wrappers
Python resuelven raíz por __file__ y los shell por su ubicación. No insertar
src globalmente en sys.path: puede colisionar con paquetes de las pruebas.

## Entradas del flujo vigente

Con el entorno `~/s4a-venv/bin/python`, desde la raíz:

| Necesidad | Entrada actual | Contrato |
|---|---|---|
| Preparar/verificar TRAIN/VAL | src.datos.preparar_multicampana; scripts/preparacion/preparar_multicampana.sh | --solo-verificar revisa fuentes/grillas; procedencia y arrays se verifican aparte. Conserva17 tiles de preparación |
| Validar arrays sin entrenar | src.entrenamiento.entrenar_cordoba --verificar-datos --campana ... | TRAIN/VAL del split vigente; TEST de mapas no requiere arrays nuevos |
| Entrenar | src.entrenamiento.entrenar_cordoba; scripts/entrenamiento/entrenar_multicampana_unet.sh | Tres campañas, selección con VAL, cierre multicampaña en TEST vigente |
| Evaluar pesos congelados | src.evaluacion.test_actual --exp EXP --salida CORRIDA | --verificar hace preflight; sin él infiere y escribe métricas completas/independientes |
| Evaluar con lock/estado | scripts/evaluacion/inferencia_multicampana.sh EXP SALIDA | Dispatcher vigente, lock/PID/exitcode; SALIDA debe existir |
| Regenerar figuras | src.figuras.regenerar_test_actual --salida CORRIDA | Usa protocolo y predicciones existentes; conserva figuras.exitcode |

Preparación de extensión: scripts/preparacion/descargar_extension_este.sh
selecciona/descarga cuatro campañas; preparar_extension_este.sh audita,
compone, prepara máscaras y evalúa U-Net congelada. No son entradas
intercambiables con el evaluador canónico. Ver el orden en las guías.
Datos, TIFF, pesos, caches, corridas y logs viven físicamente en /mnt/yacy_*;
conservar las rutas acreditadas y no recrear el dataset20JLL eliminado.

## Compatibilidad Python: entradas y destinos reales

Orígenes y destinos relativos a src/. Todas las rutas siguientes existen.
Los wrappers reexportan funciones requeridas; la implementación no se duplica.

| Entrada anterior compatible | Implementación actual |
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

Dos entradas coordinan contratos distintos:

- preparar_evaluar_extension.py conserva --modo preparar/evaluar y exports de
  datos/preparar_extension.py y evaluacion/evaluar_extension.py.
- postprocesar_multicampana.py conserva main importado histórico y exports
  perezosos de historico/postprocesar_multicampana.py. Su CLI deriva al dispatcher
  evaluacion/postprocesar_multicampana.py: TEST vigente por defecto y anterior
  con --historico. --lock-adquirido se retira solo en la rama histórica.

## Shells: destinos reales y entradas compatibles

Todos los nombres de esta tabla tienen una entrada anterior homónima en src/.
Los wrappers delegan argumentos y códigos de salida; el dominio no cambia la
secuencia ni el protocolo del shell.

| Directorio de implementación | Shells |
|---|---|
| scripts/preparacion/ | preparar_multicampana.sh, descargar_2020_2021_scl.sh, rehacer_2018_2019_scl.sh, descargar_extension_este.sh, preparar_extension_este.sh |
| scripts/entrenamiento/ | entrenar_multicampana_unet.sh |
| scripts/evaluacion/ | inferencia_multicampana.sh |
| scripts/mantenimiento/ | recuperar_extension_este.sh, optimizar_lectura_multicampana.sh |
| scripts/historico/ | pipeline_dicmay.sh, pipeline_dicmay_1718.sh, evaluar_dicmay.sh, preparar_ano_no_visto.sh |

Rehacer2018/2019 termina infiriendo/evaluando/graficando modelos base anteriores.
Preparar año no visto requiere SALIDA y logs_multicampana existentes; conserva
fd8 de preparación, PID/trap/exitcode/fin, espera MNC y fd9 GPU. Mantenimiento
puede reemplazar rasters, invalidar arrays o detener/reanudar entrenamiento:
no ejecutar esos shells como diagnóstico ni para validar la organización.

## Dependencias y contratos conservados

- Preparación multicampaña usa datos/campanas.py, máscarasMNC2021 y generadores.
  Recuperación carga preparación/composites_scl al ejecutar; no los importa como
  efecto de diagnosticar. Nivel3 importa REMAP_NIVEL3 del generador03c trasladado.
- El entrenador y el dispatcher vigente llaman a evaluacion/test_actual. TEST
  usa utils/test_actual.py para split/preflight y datos/campanas.py para fuentes.
- Evaluación temporal importa directamente el postprocesador histórico y comparte
  LEGENDS con figuras/historico; conserva roles/tiles del protocolo anterior.
- Inferencia raster usa paso por float16 y publicación atómica; inferir_tile CLI
  conserva float32, escritura directa, CPU y modelos de24/28 canales. No unificarlos.
- Los comandos de parches5/6 conservan NPZ, modelos28, ventanas256 y distintos
  remapeos dataset_T. Los mapas inferidos en bloques512 no son métricas de parches.
- Históricos8–12 conservan calendarios, pesos, leyendas, remapeos, prefijos y salidas.
  9 calcula acuerdo solo en cultivos; 10 cuenta pred0 en confusión/F1/IoU pero
  su accuracy impresa usa subbloque1–4; 11 excluye pred0. No corregir esas
  diferencias ni convertirlos a TEST vigente cambiando nombres de tiles.
- Manfredi pertenece a entrenamiento:20HMK, cinco clases, checkpoint original,
  VAL0.15 y seed42; no fija semillas ni crea directorios al importar.

## Configuración, imports y alcance

config/almacenamiento.json y utils/rutas.py centralizan almacenamiento y aliases.
config/split_extension_este.json define TEST vigente; utils.multicampana.SPLIT
conserva el conjunto anterior de preparación/protocolos históricos.
config/campanas.json y datos/campanas.py centralizan dic–may, bandas y rutas
para preparación multicampaña, extensión y evaluador vigente. Su consumo es
parcial: variantes históricas, preparación específica y shells conservan
configuraciones explícitas. Cambiar ese catálogo no parametriza todo el proyecto.
En2019/2020, composites_multicampana y composites_extension difieren a propósito.

Los módulos revisados no ejecutan comandos, leen datasets/rasters/pesos, escriben
salidas, fijan semillas ni consultan GPU al importar. Pueden cargar configuración
JSON pequeña, crear tablas en memoria y seleccionar backendAgg para figuras.
Recuperar multicampaña no admite --help: cualquier argumento ejecuta mantenimiento.
No usar ese comando en auditorías CLI. Moldear máscaras, composites JP2 originales
y generador03 conservan main() con
configuración fija: no invocar --help en esas entradas suponiendo que tienen parser.
Los comandos con parser admiten ayuda; postprocesar tiene dispatcher propio.
Las APIs puras no necesitan CLI.

La reorganización permite localizar y reutilizar responsabilidades. No demuestra
generalización ML ni parametriza todas las regiones, bandas, clases o protocolos.
No incluye nuevos algoritmos, experimentos, traslados de volúmenes ni reorganizar
todo utils. Validación y evidencia de cierre en el
[registro fechado](../docs/historico/reorganizacion_src_20261011.md).

# Revisión de wrappers de src — 11/10/2026

Referencia: floryacy, HEAD observado `a6d096eabe395012a0b57f8207713be2b7e9b6fe`. Revisión del checkout versionado, AST/imports y literales de comandos/pruebas; los comentarios y docstrings Python no se cuentan como llamadas operativas.

Se inventariaron 49 entradas compatibles: 36 Python y 13 shells. No se encontraron llamadas/imports desde implementaciones Python o shells operativos del repo. Los consumidores observados son pruebas de compatibilidad y referencias documentales. El índice general de rutas se excluyó del conteo para no confundir su tabla de entradas con consumidores.

No se encontraron procesos del usuario ejecutando wrappers en el momento de la revisión. Crontab: `sin_crontab`. Directorio de unidades systemd del usuario presente: `False`. Esto no demuestra ausencia de comandos externos, tareas de otros usuarios, notebooks ajenos al checkout o imports construidos sin literales detectables.

Las 36 entradas Python están incluidas además en la auditoría genérica de imports de `test/infraestructura/test_estructura_src.py`. Las referencias explícitas siguientes cubren 47 wrappers; `01d_mascaras_mnc_2020_2021.py` tiene la cobertura genérica y `entrenar_multicampana_unet.sh` no tiene una prueba específica del wrapper identificada. No confundir pruebas de los módulos destino con consumidores de estas rutas.

## Propuesta de retiro, aún sin implementar

1. Retirar primero los 13 shells de la raíz de src. Migrar las pruebas que comparan wrappers y destinos a pruebas de scripts/; mantener locks, argumentos y códigos de salida. Actualizar referencias vigentes de compatibilidad; conservar relatos históricos.
2. Retirar los 34 wrappers Python que delegan a una implementación. Migrar las pruebas de imports/CLI/exports a los módulos de dominio. Mantener intactos los algoritmos y protocolos históricos en sus carpetas.
3. Decidir por separado los dos contratos adicionales: postprocesar_multicampana.py (main importado histórico, CLI vigente y exports perezosos) y preparar_evaluar_extension.py (CLI --modo y exports de preparación/evaluación). Sus sustitutos ya existen, pero retirar esas APIs requiere explicitar la ruptura de compatibilidad.

Un retiro rompe deliberadamente los comandos/imports antiguos. El inventario no autoriza borrar código histórico, datos ni evidencia. En esta revisión no se retiraron wrappers.

## Evidencia por entrada

La columna pruebas enumera referencias literales específicas; la auditoría genérica Python indicada arriba se aplica adicionalmente. Las referencias documentales/históricas son menciones, no ejecución.

| Entrada | Pruebas con referencias | Documentación vigente | Referencias históricas |
|---|---|---|---|
| `src/00_descargar_stac.py` | `test/datos/test_fuentes_composites.py`:177 | — | — |
| `src/01_moldear_mascaras.py` | `test/infraestructura/test_cli_fija.py`:17 | — | — |
| `src/01b_mascaras_mnc_2018_2019.py` | `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/01c_mascaras_nivel3_2017_2018.py` | `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/01d_mascaras_mnc_2020_2021.py` | — | — | — |
| `src/02_composites_mensuales.py` | `test/datos/test_fuentes_composites.py`:177; `test/infraestructura/test_cli_fija.py`:18 | — | — |
| `src/02b_composites_2019_2020.py` | `test/datos/test_fuentes_composites.py`:177; `test/infraestructura/test_cli_fija.py`:19 | — | — |
| `src/02c_composites_cog.py` | `test/datos/test_fuentes_composites.py`:177,200 | — | — |
| `src/02d_composites_scl.py` | `test/datos/test_fuentes_composites.py`:177,208 | — | `docs/historico/geografia_20261009.md`:209 |
| `src/03_generar_datasets_npz.py` | `test/infraestructura/test_cli_fija.py`:20 | — | — |
| `src/03b_convertir_datasets_a_mmap.py` | `test/datos/test_cli_datos.py`:18,109,127 | — | — |
| `src/03c_generar_dataset_cordoba_f16.py` | `test/datos/test_generadores_arrays.py`:149; `test/infraestructura/test_rutas_checkout.py`:15 | — | `docs/historico/migraciones/reorganizacion_20261010.json`:216,217,667,668 |
| `src/03d_generar_dataset_cordoba_2017_f16.py` | `test/datos/test_generadores_arrays.py`:149; `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/0_verificar_pipeline.py` | `test/entrenamiento/test_verificador_finetune.py`:22,39; `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/0b_calcular_pesos.py` | `test/datos/test_cli_datos.py`:18,108,119 | `src/entrenamiento/README.md`:7 | — |
| `src/10_metricas_test.py` | `test/evaluacion/test_cli_historicas.py`:22; `test/evaluacion/test_comparaciones_trasladadas.py`:44 | — | — |
| `src/11_evaluar_contra_mnc.py` | `test/evaluacion/test_mnc_1718_trasladados.py`:30,48; `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/12_comparar_con_1718.py` | `test/evaluacion/test_mnc_1718_trasladados.py`:30,48,81; `test/infraestructura/test_rutas_checkout.py`:15 | — | — |
| `src/1_train_cordoba.py` | `test/entrenamiento/test_cli_imports.py`:22 | — | `docs/historico/experimentos_20261006_10.md`:135,549; `docs/historico/geografia_20261009.md`:209 |
| `src/1b_finetune_manfredi.py` | `test/entrenamiento/test_verificador_finetune.py`:22,39 | — | — |
| `src/2_inferencia.py` | `test/inferencia/test_inferir_tile.py`:108,117,134 | `docs/guias/multicampana.md`:293; `src/inferencia/README.md`:17 | — |
| `src/4_generar_metricas.py` | `test/evaluacion/test_cli_historicas.py`:22 | — | — |
| `src/5_generar_matriz_confusion.py` | `test/evaluacion/test_cli_historicas.py`:22 | — | — |
| `src/6_generar_comparacion_visual.py` | `test/evaluacion/test_cli_historicas.py`:22 | — | — |
| `src/7_visualizar_inferencia.py` | `test/evaluacion/test_cli_historicas.py`:22; `test/infraestructura/test_organizacion.py`:61 | — | — |
| `src/8_comparar_modelos.py` | `test/evaluacion/test_cli_historicas.py`:22; `test/evaluacion/test_comparaciones_trasladadas.py`:42 | — | — |
| `src/9_comparar_inferencia_etiqueta.py` | `test/evaluacion/test_cli_historicas.py`:22; `test/evaluacion/test_comparaciones_trasladadas.py`:43 | — | — |
| `src/descargar_2020_2021_scl.sh` | `test/infraestructura/test_shells_preparacion.py`:12,125,131,133 | — | — |
| `src/descargar_extension_este.sh` | `test/infraestructura/test_shells_preparacion.py`:12,74,77,112,147,155 | — | `docs/historico/experimentos_20261006_10.md`:403,408 |
| `src/entrenar_multicampana_unet.sh` | — | — | `docs/historico/experimentos_20261006_10.md`:226 |
| `src/evaluar_ano_no_visto.py` | `test/evaluacion/test_postprocesado_separado.py`:140,148,170 | — | `docs/historico/geografia_20261009.md`:209 |
| `src/evaluar_dicmay.sh` | `test/infraestructura/test_shells_historicos.py`:13,215,218,221,222 | — | — |
| `src/evaluar_test_actual.py` | `test/datos/test_campanas.py`:78; `test/evaluacion/test_cli_test_actual.py`:21,35 | `src/evaluacion/README.md`:36 | — |
| `src/inferencia_multicampana.sh` | `test/infraestructura/test_shell_evaluacion.py`:18,19,21,24 | — | `docs/historico/experimentos_20261006_10.md`:317,557 |
| `src/mejorar_calidad_extension.py` | `test/datos/test_mejorar_calidad.py`:49,129 | — | — |
| `src/optimizar_lectura_multicampana.py` | `test/entrenamiento/test_optimizar_lectura.py`:18,33 | — | — |
| `src/optimizar_lectura_multicampana.sh` | `test/entrenamiento/test_optimizar_lectura.py`:18,83 | — | — |
| `src/pipeline_dicmay.sh` | `test/infraestructura/test_shells_historicos.py`:13,158,208 | — | `docs/historico/pipelines_anteriores.md`:271 |
| `src/pipeline_dicmay_1718.sh` | `test/infraestructura/test_shells_historicos.py`:13,209,210 | — | — |
| `src/postprocesar_multicampana.py` | `test/evaluacion/test_componentes.py`:26; `test/evaluacion/test_postprocesado_separado.py`:135,148,170 | — | `docs/historico/experimentos_20261006_10.md`:557; `docs/historico/geografia_20261009.md`:209 |
| `src/preparar_ano_no_visto.sh` | `test/infraestructura/test_shells_historicos.py`:13,125,189,227,233,239 | `docs/guias/multicampana.md`:312 | — |
| `src/preparar_evaluar_extension.py` | `test/evaluacion/test_componentes.py`:27; `test/evaluacion/test_extension_separada.py`:38,57,64 | `docs/guias/multicampana.md`:286; `docs/guias/test_actual.md`:32 | `docs/historico/experimentos_20261006_10.md`:515 |
| `src/preparar_extension_este.sh` | `test/infraestructura/test_shells_preparacion.py`:12,74,118,155,166 | — | `docs/historico/experimentos_20261006_10.md`:515 |
| `src/preparar_multicampana.py` | `test/datos/test_preparar_multicampana.py`:160,174 | `docs/guias/multicampana.md`:252 | `docs/historico/experimentos_20261006_10.md`:108 |
| `src/preparar_multicampana.sh` | `test/datos/test_preparar_multicampana.py`:160,184,200 | `docs/guias/multicampana.md`:252 | `docs/historico/experimentos_20261006_10.md`:115 |
| `src/recuperar_extension_este.sh` | `test/datos/test_mejorar_calidad.py`:134 | — | — |
| `src/recuperar_multicampana.py` | `test/datos/test_recuperar_multicampana.py`:23; `test/infraestructura/test_cli_fija.py`:16 | — | `docs/historico/experimentos_20261006_10.md`:191 |
| `src/regenerar_figuras_test_actual.py` | `test/entrenamiento/test_cli_imports.py`:22; `test/evaluacion/test_cli_test_actual.py`:21,35 | `src/figuras/README.md`:42 | — |
| `src/rehacer_2018_2019_scl.sh` | `test/infraestructura/test_shells_preparacion.py`:12,138 | — | — |

El JSON contiguo `wrappers_src_20261011.json` conserva destinos/imports, archivos y líneas observadas; las líneas corresponden al HEAD auditado. Las búsquedas por literales no reemplazan una revisión de todos los mecanismos dinámicos.

Respaldo anterior, índice y manifiesto SHA256: /mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/revision_wrappers_20261011T143402Z/

Validación de esta entrega: índice contrastado con código actual, destinos existentes, ayuda de CLI fija y auditoría de imports/consumidores; git diff --check.

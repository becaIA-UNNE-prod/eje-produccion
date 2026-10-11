# Resultados y evidencias

Este índice se versiona. En floryacy, `archive/` y `runs/` son enlaces a discos:
los resultados físicos se almacenan en `/mnt`, según `config/almacenamiento.json`.
Las copias ligeras del mirror son solo para revisión.

| Ubicación | Contenido |
|---|---|
| `reports/test_actual/` | Enlace al reporte vigente; en floryacy apunta al reporte de `/mnt` y en el mirror a la copia local. |
| `runs/<fechaUTC_etapa_nombre_ID>/` | Registros livianos: `run.json`, `estado.json`, `exitcode`, logs, configuración e inventarios. Resultados pesados en destinos explícitos registrados. |
| `runs/legacy_octabr/` | Espacio manual de las herramientas históricas si no se especifica `CORDOBA_RUN_DIR`. |
| `archive/legacy_root/figuras/` | PNG, mapas y comparaciones antes sueltos en la raíz. |
| `archive/legacy_root/predicciones/` | GeoTIFF de inferencias históricas. |
| `archive/legacy_root/logs/` | Logs y nohup históricos. |
| `archive/legacy_root/inventarios/` | Inventarios y auditorías existentes. |
| `archive/legacy_root/backups/` | Backups anteriores, incluido el del generador de datasets. |
| `archive/baseline_sin_2017/` | Corrida baseline original, preservada completa. |
| `archive/previo_20260918_mesesmal/` | Resultados con meses incorrectos; evidencia histórica, no TEST vigente. |
| `reports/resultados_test_actual_comparacion_20261010/` | Copia local de figuras y métricas; TIFF y pesos permanecen en el servidor. |

## Rutas operativas vigentes en floryacy

- Reporte TEST: `/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/comparacion_test_actual_20261010T092436`.
- Preparación y exclusiones: usar `preparacion_base` de `config/split_extension_este.json`.
- Experimentos multicampaña y logs históricos: `/mnt/yacy_3/prod_bis/ferreyra/dataset/` y `/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/`.

Los archivos históricos conservan sus nombres. La ubicación no convierte una
imagen o una métrica antigua en un resultado del TEST vigente: consultar protocolo,
campañas, split y criterio de evaluación antes de comparar.

La reorganización y sus SHA256 se registran en
[el manifiesto](../docs/historico/migraciones/reorganizacion_20261010.json).
Para revisar movimientos, interpretar el manifiesto junto al registro de
[mantenimiento](../docs/historico/mantenimiento_20261010.md): algunos destinos
fueron eliminados después por solicitud del usuario. No intentar revertirlos
sin comprobar el estado actual y el alcance de los respaldos.

## Almacenamiento físico

- Archivo: `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/artifacts/archive/`.
- Nuevas corridas: `/mnt/yacy_3/prod_bis/ferreyra/cordoba_ia_unne/artifacts/runs/`.
- Datos compartidos: `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/data/`.
- Respaldos: `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/`.

Ver [traslado a discos](../docs/historico/migraciones/almacenamiento_discos_20261010.json)
y [CONTRIBUTING](../CONTRIBUTING.md).

## Nuevas ejecuciones

Los shells operativos crean su corrida automáticamente. Para comandos Python,
usar `scripts/ejecutar_con_log.py`. Todos los logs de una ejecución quedan en
`runs/<fechaUTC_etapa_nombre_ID>/logs/`; los pasos no reutilizan las carpetas
históricas `logs_scl`, `logs_dicmay` o `logs_multicampana`.

La carpeta de corrida contiene registros pequeños. Imágenes, datasets, caches,
checkpoints y predicciones permanecen en sus ubicaciones; nuevas salidas se
eligen por capacidad del disco. El manifiesto registra las rutas reales y permite
relacionar corridas mediante `--padre`. No se trasladan artefactos para ordenarlos.
Ver [lanzamiento y capacidad](../CONTRIBUTING.md#cómo-ejecutar-sin-ensuciar-la-raíz).

## Mapas de tiles

Abrir `maps/index.html`: mapa vigente. Cada nueva corrida conserva `mapas/mapa_tiles.html`. Ver [guía](../docs/guias/mapas_tiles.md).

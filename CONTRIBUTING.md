# Guía operativa para Codex y otros agentes

## Fuente de referencia y orden de lectura

El repositorio principal de trabajo es **`floryacy:~/cordoba_ia_unne`**
(`/home1/ferreyra/cordoba_ia_unne`). La copia en esta computadora es un mirror.
Cuando difieran, comparar commits, índice y archivos de trabajo de ambos lados:
el HEAD remoto por sí solo no incluye los cambios preparados sin commit.
Conservar aportes locales y sincronizar el mirror después de modificar floryacy.

Leer, en este orden:

1. Este archivo y `git status --short` del lugar donde se va a trabajar.
2. `config/split_extension_este.json` y [TEST vigente](docs/guias/test_actual.md).
3. [Multicampaña](docs/guias/multicampana.md) y la documentación de la etapa afectada.

La configuración y el código actual prevalecen sobre notas de corridas antiguas.
Los archivos de `docs/` que registran una ejecución conservan sus fechas y rutas
históricas; no interpretarlos como una orden de volver a lanzarla.

## Contrato científico del TEST

- TEST vigente: **20HNJ y 20HNH**. TRAIN y VAL están en la configuración común.
- Diciembre–mayo: seis meses, cuatro bandas B02/B03/B04/B08, 24 canales.
- Evaluación principal: mapas con exclusión de las grillas completas de TRAIN/VAL.
  Conservar también las métricas de mapas completos y distinguir ambos criterios.
- Seleccionar el checkpoint con VAL; no entrenar ni elegir pesos usando TEST.
- Usar `src/evaluacion/test_actual.py` (por `-m src.evaluacion.test_actual`)
  y `utils/test_actual.py`; no inventar otro split.
- `20HLK/20HLJ`, los scripts `8`, `9`, `10` y las corridas oct–abr son históricos.
  No basta con cambiar los nombres de tiles en esos scripts para evaluar TEST vigente.
- Conservar ventanas, leyendas, normalización, procedencia y compatibilidad de canales.
  Métricas de parches de 256 y mapas inferidos en bloques de 512 no son equivalentes.

## Almacenamiento: código en el repo, volumen en los discos

**Los datasets, composites, máscaras, GeoTIFF, pesos, caches, resultados de
corridas, logs y backups se almacenan físicamente en `/mnt/yacy_*`.** El repositorio
contiene código, configuración, Markdown y metadatos pequeños. Agregar una salida
a `.gitignore` no autoriza a acumularla físicamente dentro del repositorio.

La fuente de rutas es `config/almacenamiento.json`, consumida por `utils/rutas.py`.
Para un nuevo artefacto, determinar su rol y usar la ruta correspondiente:

| Rol | Ruta física |
|---|---|
| Datos, referencias, máscaras y pesos compartidos | `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/data/` |
| Archivo de corridas históricas | `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/artifacts/archive/` |
| Respaldos y manifiestos de traslado | `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/` |
| Nuevas corridas | `/mnt/yacy_3/prod_bis/ferreyra/cordoba_ia_unne/artifacts/runs/<nombre_FECHA_ID>/` |

`yacy_1` aloja entradas compartidas y archivo histórico; `yacy_3` aloja nuevas
corridas. Esta separación sigue los flujos multicampaña establecidos y la
capacidad revisada al migrar. Antes de elegir otro disco, verificar montaje,
permisos, espacio disponible y la configuración; registrar el cambio.
Los datos y experimentos ya existentes en otras rutas de `/mnt` conservan sus
ubicaciones acreditadas, en particular `preparacion_base` del TEST y el reporte
vigente. No copiar esos volúmenes al nuevo namespace para imponer uniformidad.

```text
~/cordoba_ia_unne/                # Código y metadatos livianos
├── AGENTS.md, CONTRIBUTING.md, README.md
├── config/                     # Split, almacenamiento, leyendas y QML
├── src/, utils/, test/, scripts/
├── docs/                       # Guías y manifiestos pequeños
├── artifacts/README.md         # Índice; resultados físicos en /mnt
│   ├── archive -> /mnt/yacy_1/.../artifacts/archive
│   ├── runs -> /mnt/yacy_3/.../artifacts/runs
│   └── reports/test_actual -> reporte vigente en /mnt
├── data/referencias -> /mnt/yacy_1/.../data/referencias
├── mascaras_procesadas -> /mnt/yacy_1/.../data/mascaras
└── pesos -> /mnt/yacy_1/.../data/pesos
```

Los enlaces de directorio anteriores conservan los consumidores históricos,
pero los bytes viven en los discos. No agregar estos enlaces a Git ni sustituirlos
por directorios locales cuando falte un montaje. No crear enlaces individuales
para antiguas imágenes/logs en la raíz. Nuevos módulos usan las constantes de
`utils/rutas.py`; las etapas históricas pueden seguir usando los aliases listados.

Cada nueva corrida tiene `run.json`, `exitcode`, `logs/`, `figuras/`,
`predicciones/`, `metricas/` e `inventarios/`. Separar corridas por nombre,
fecha UTC y PID, y distinguir artefactos completos de temporales. Los archivos
históricos preservan su agrupación de origen bajo `archive/`.

El mirror local comparte código y configuración. Puede mantener copias ligeras
de figuras/métricas para revisión, pero no replica los datasets, TIFF o pesos.
Los enlaces a `/mnt` pueden no resolverse allí: ejecutar ML en floryacy. Las
pruebas ligeras pueden usar un directorio temporal explícito fuera del repo.

## Cómo ejecutar sin ensuciar la raíz

Trabajar desde la raíz del repositorio. El entorno operativo en floryacy es
**`~/s4a-venv/bin/python`** (Python 3.11). Usar ese intérprete explícitamente
para comandos y pruebas; el proyecto se ejecuta desde su raíz, sin instalarlo
como paquete. El entorno se administra fuera del repositorio: no instalar,
actualizar ni cambiar dependencias como efecto lateral del mantenimiento.

El flujo uv fue retirado del proyecto, incluidos `pyproject.toml`, `uv.lock`
y `.python-version`; no usarlos para recrear o sincronizar un entorno local.
El `.venv` histórico fue eliminado el 10/10/2026. No recrearlo ni introducir
otro gestor de dependencias como reemplazo.

El lanzador crea un directorio con fecha UTC y PID, captura stdout/stderr,
registra el comando, HEAD, estado Git y la firma del split, y guarda `exitcode`:

```bash
~/s4a-venv/bin/python scripts/ejecutar_con_log.py --nombre verificar_test -- \
  ~/s4a-venv/bin/python -m unittest discover -s test -p test_test_actual.py -v
```

La base por defecto del lanzador es `RUNS_ROOT` en `yacy_3`. Rechaza carpetas
de corrida físicas dentro del repositorio y no crea un fallback local si falta
el disco. `CORDOBA_STORAGE_ROOT` y `CORDOBA_RUNS_ROOT` permiten destinos absolutos
alternativos; `--base` puede elegir un destino explícito.

Exporta `CORDOBA_RUN_DIR` para las herramientas históricas `2`, `7`, `8`, `9`,
`10` y el inventario Sentinel. Sin esa variable, usan
`RUNS_ROOT/legacy_octabr/` en `yacy_3`, un espacio manual histórico compartido.
Usar el lanzador o un `CORDOBA_RUN_DIR` nuevo para conservar cada corrida;
para encadenar inferencia y figuras, usar el mismo directorio de corrida.
Los parámetros explícitos `--dir-salida`, `--salida_dir`, `--salida` y
`--dir-pred` prevalecen sobre esos valores por defecto.

Los evaluadores actuales tienen rutas explícitas: pasar `--salida`/`--dir-exp`
a la carpeta de ejecución elegida en `/mnt/yacy_*`. El lanzador captura logs pero no reescribe
los argumentos ni redirige checkpoints o resultados explícitos.
Para trabajos largos, usar el supervisor ya establecido o `nohup` con salida
explícita fuera de la raíz, consultar PID/progreso y verificar el código de
salida. Evitar `nohup.out` y nombres genéricos repetidos en la raíz.

## Migraciones entre filesystems

Mover de `/home1` a `/mnt` implica copiar y retirar el origen. Verificar que no
haya procesos usando el árbol; respaldar código/índice y registrar cada archivo
con origen, destino, tamaño, SHA256 y mtime. Preservar metadatos, verificar la
copia completa y recién entonces retirar el árbol original y crear el alias.
No dejar duplicados permanentes de volúmenes para simular un backup.

La procedencia de datasets guarda rutas resueltas de máscaras. Tras mover una
entrada, actualizar únicamente esas rutas acreditadas en los JSON de procedencia,
conservando los JSON originales en el respaldo y sin cambiar meses, clases,
normalización, hashes o timestamps de los datos. Verificar después las grillas con
`preparar_multicampana.py --solo-verificar` y comprobar explícitamente la
procedencia de los datasets: `--solo-verificar` no compara cada JSON. Un enlace
por sí solo no evita el cambio de `Path.resolve()`. Los manifiestos históricos de migración conservan sus
rutas originales y se interpretan junto al manifiesto del traslado a discos.

## Versionado, limpieza y sincronización

- Versionar código, configuración, documentación y manifiestos pequeños.
  La leyenda QML se versiona en `config/referencias/`; el TIFF Nivel3 y
  cualquier dato grande pertenecen al almacenamiento operativo.
- No agregar resultados generados, logs, predicciones, checkpoints, caches,
  entornos ni backups a Git. `artifacts/` queda ignorado salvo su índice.
- Antes de mover un archivo, buscar consumidores y verificar procesos que lo
  estén usando. Mover corridas terminadas, preservar sus nombres y metadatos,
  registrar origen/destino/tamaño/SHA256 y verificar después del traslado.
- No borrar evidencias para limpiar. `archive/` conserva resultados históricos,
  incluidos experimentos con meses incorrectos, identificados como tales.
- Mantener únicamente los enlaces de directorio listados en el esquema;
  nunca enlaces por imagen/log en la raíz. Actualizar las rutas de consumidores; los comandos antiguos documentados pueden necesitar
  `--dir-pred artifacts/archive/legacy_root/predicciones` y `--dir-salida`.
- No hacer `reset --hard`, descartar cambios ajenos ni incluir todo con `git add .`.
  Mantener los cambios previos preparados y revisar el diff de la tarea.
- Al terminar, sincronizar código/documentación con el mirror y verificar hashes.
  Los artefactos voluminosos quedan en los discos de floryacy; las copias locales se ubican en
  `artifacts/reports/` o `archive/`, preservando ambas versiones si difieren.

## Validación y entrega

Para cambios en TEST o sus rutas, ejecutar en floryacy:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python -m unittest discover -s test -p test_test_actual.py -v
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python -m unittest discover -s test -p test_extension_este.py -v
```

Las pruebas anteriores usan fixtures y no requieren la preparación real en /mnt.
Para comprobar explícitamente los metadatos y la presencia de máscaras acreditadas:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python scripts/diagnostico/verificar_preparacion_test_actual.py
```

Esta comprobación es de solo lectura, falla ante faltantes y queda fuera de
unittest discover. No reemplaza una auditoría raster ni el preflight de checkpoints.

Para el lanzador y las rutas de figuras, ejecutar también `python -m unittest
discover -s test -p test_organizacion.py -v` con el entorno operativo.
Agregar las pruebas pertinentes a los componentes modificados; para una
reorganización, comprobar integridad de archivos, enlaces, imports, sintaxis,
parámetros de salida y que no queden resultados sueltos en la raíz.
No lanzar entrenamientos o inferencias completas para validar un simple cambio
de ubicación si las verificaciones de rutas y los tests cubren ese cambio.
La entrega debe indicar ubicaciones finales, respaldo/manifiesto, validación,
estado de Git y cualquier trabajo que falte.

El dataset histórico de 20JLL (`legacy_repo`, 40 canales, julio 2017–abril 2018)
fue eliminado el 10/10/2026 por solicitud del usuario, junto con el alias
`dataset` y sus dos predicciones/figura exclusivas archivadas. No recrearlo como
fallback. Las máscaras compartidas y los demás datasets que contienen ese tile
se conservan; consultar el manifiesto de eliminación en `docs/historico/migraciones/`.

## Documentación

Usar `docs/README.md` como índice. Mantener procedimientos actuales en `guias/`,
definiciones en `referencias/` y relatos fechados/manifiestos en `historico/`.
No incorporar PID, estados transitorios o autorizaciones de una corrida a una
guía vigente. Actualizar la página existente y contrastarla con código/config
antes de crear otra. Las rutas y splits históricos no son instrucciones actuales.

## Pruebas y diagnóstico

`test/` contiene pruebas agrupadas en datos, entrenamiento, evaluación e
infraestructura; consultar `test/README.md`. Los comandos de descubrimiento
por nombre siguen funcionando recursivamente. Las utilidades manuales están
en `scripts/diagnostico/`, con rutas/coordenadas explícitas. No incorporar
lecturas de datos reales como efecto lateral de importar una prueba.

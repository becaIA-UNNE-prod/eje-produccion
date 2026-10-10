# Reorganización y limpieza — 10/10/2026

Registro cerrado; las rutas operativas actuales están en CONTRIBUTING y
`config/almacenamiento.json`. Sustituye el inventario de retención anterior,
que mezclaba candidatos, cifras previas y eliminaciones posteriores.

- Se agruparon salidas antes sueltas en `artifacts/archive/`, con archivo físico
  en `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/`.
- Se trasladaron 7.665 archivos (33.512.676.945 bytes) desde `/home1` a los discos,
  verificando SHA256/mtime y conservando aliases de los árboles necesarios.
- Se actualizaron rutas de máscaras en 34 JSON de procedencia, respaldando originales.
- Por solicitud del usuario se eliminó el entorno uv antiguo y el enlace `.venv`.
  El entorno operativo `~/s4a-venv` se conservó; ahorro aproximado 318 MiB.
- También se eliminó el dataset antiguo 20JLL: 10 composites de julio 2017–abril
  2018 y 1.764 pares X/Y de 40 canales, junto con el alias `dataset`, dos
  predicciones y una figura exclusiva. Total: 3.541 archivos, 27.883.136.739 bytes.
  Se conservaron máscaras compartidas, entradas Sentinel y otros datasets operativos.
- Los manifiestos describen movimientos pasados; no demuestran que sus destinos
  sigan existiendo después de las eliminaciones. No contienen copias de los datos.

## Evidencia

- [Reorganización inicial](migraciones/reorganizacion_20261010.json).
- [Traslado a discos](migraciones/almacenamiento_discos_20261010.json).
- [Eliminación de 20JLL](migraciones/eliminacion_dataset_20JLL_20261010.json).
- Listados detallados y respaldos físicos bajo
  `/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/`.

## Criterios que siguen siendo útiles

No descartar máscaras por antigüedad: las familias mnc/1819/2021 y etiquetas
históricas `_test` tienen consumidores. La corrida `previo_20260918_mesesmal`
es histórica y contiene meses incorrectos. Revisar el valor documental de cada
corrida antes de eliminarla. DEFLATE redujo ~98 % dos predicciones de prueba,
sin cambiar píxeles/georreferencia; no se aplicó a todos los originales.
No considerar aquel porcentaje un ahorro medido para todo el archivo.

`estructura_sentinel2.json` es un índice histórico tile → fecha → bandas,
generado al recorrer las entradas originales. No contiene imágenes ni acredita
su integridad o participación en TRAIN/TEST. Se puede regenerar, pero la nueva
salida representa otra fecha de observación.

La documentación anterior a esta consolidación está respaldada en
`backups/docs_antes_reorganizacion_20261010.tar` bajo la base física indicada.
Los aportes originales del mirror se respaldaron localmente en
`/tmp/cordoba_docs_antes_20261010.tar`; su auditoría y descripción geográfica
se incorporaron al archivo histórico sin declararlas una nueva verificación.

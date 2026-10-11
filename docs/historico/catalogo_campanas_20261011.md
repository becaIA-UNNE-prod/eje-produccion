# Catálogo por campaña y alcance — 11/10/2026

Paso 5 del mantenimiento, realizado en floryacy. Inventario actual y límites en
src/datos/README.md, sección Catálogo y límites de configuración.

Alcance aplicado: campañas/tiles del flujo original de extensión y campañas que
aportan grillas de exclusión se registran en protocolos.extension_este_original.
preparar_extension deriva las referencias desde el catálogo con su base Y3,
conservando orden, rutas y geometrías; evaluar_extension hereda el mismo alcance.
La procedencia multicampaña consume la normalización ya validada del catálogo.
Se refuerzan validaciones de rutas, nombres, listas, patrones y referencias.

No se unifican métodos JP2/COG/SCL, remapeos, calendarios históricos, dtypes ni
criterios de métricas. 2019/2020 TRAIN/VAL sigue separado de extensión SCL.
2021/2022 no tiene dataset de entrenamiento. El split vigente sigue siendo
config/split_extension_este.json; extensión conserva su alcance original separado.
No se alteraron datos físicos ni protocolos de corridas existentes.

Validación: catálogo inválido y copias de listas, equivalencia de rutas/orden de
referencias, fixtures de preparación/evaluación y suite sin lecturas Python de /mnt.
Resultado: 160 pruebas aprobadas sin skips; log suite_sin_mnt.log en el respaldo.
Respaldo, índice previo y manifiesto SHA256: /mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/paso5_catalogo_20261011T140929Z/
Mirror sincronizado conservando su índice. Finalizan los cinco pasos del plan.

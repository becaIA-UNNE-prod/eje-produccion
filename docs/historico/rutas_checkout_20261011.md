# Rutas del checkout — 11/10/2026

Paso 3 realizado en floryacy. Siete defaults de máscaras dejaron de fijar
/home1/ferreyra/cordoba_ia_unne y ahora usan la raíz calculada desde __file__.
Afecta generadores Córdoba f16 2020/2017, máscaras MNC1819/Nivel3 y los
consumidores históricos comparar_con_1718, evaluar_contra_mnc y verificar_pipeline.
El alias mascaras_procesadas conserva su destino físico: /mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/data/mascaras.
No se trasladaron datos ni se reescribieron JSON de procedencia.
Los demás paths /mnt y /var/tmp son ubicaciones operativas acreditadas y se conservan.
Las rutas documentadas en relatos históricos tampoco se reescribieron.

Funciones y clases conservan su AST, parámetros, cálculos y API.
Validación: checkout reubicado con alias sintético y cwd diferente, imports
sin abrir rasters/arrays ni crear directorios, defaults CLI y wrappers;
fixtures existentes de arrays, máscaras y evaluación MNC/1718; git diff --check.

Respaldo, índice previo y manifiesto SHA256: /mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/paso3_rutas_checkout_20261011T135547Z/
Mirror sincronizado conservando su índice previo. Paso 4 pendiente: separar
pruebas pequeñas y comprobaciones operativas del TEST vigente.

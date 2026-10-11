# Pruebas y comprobación operativa — 11/10/2026

Paso 4 realizado en floryacy. El preflight de test_test_actual.py ahora usa
fixtures temporales para preparación válida, exclusiones obsoletas, TEST/campañas
incompatibles y faltantes de metadatos o de cualquiera de los criterios de máscara.
La CLI operativa se prueba también con fixture y ausencia de un archivo obligatorio.

scripts/diagnostico/verificar_preparacion_test_actual.py conserva la comprobación
real mediante un comando explícito, fuera de unittest discover. Solo lee y no
oculta errores con skips. Revisa metadatos y presencia de máscaras; no certifica
contenido raster ni checkpoints. utils/test_actual.py y el split científico no cambian.

Validación: pruebas con fixtures, suite pequeña sin lecturas de volúmenes,
comprobación explícita de la preparación acreditada en floryacy y git diff --check.
Resultado: 157 pruebas aprobadas sin skips, con aperturas e inventarios Python
de /mnt bloqueados mediante sitecustomize en el proceso de pruebas y sus hijos.
La primera ejecución detectó dos lecturas indirectas desde pruebas del entrenador;
ambas ahora usan --split-config temporal, manteniendo el preflight del entrenador.
Se conservan suite_sin_mnt_inicial.log y suite_sin_mnt.log en el respaldo.
La comprobación explícita real aprobó 20HNJ/20HNH y las cuatro campañas.
Respaldo, índice previo y manifiesto SHA256: /mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/backups/paso4_pruebas_operativas_20261011T140029Z/
Mirror sincronizado conservando su índice. Paso 5 pendiente: inventariar
consumidores y diferencias de configuración por campaña/protocolo.

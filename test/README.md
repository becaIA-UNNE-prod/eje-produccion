# Pruebas automatizadas

Esta carpeta verifica código; no almacena el dataset científico TEST.
Las herramientas manuales están en `scripts/diagnostico/`.

| Carpeta | Alcance |
|---|---|
| `datos/` | Integridad raster, arrays, remapeo y procedencia multicampaña |
| `entrenamiento/` | Lectura por bloques, caches, carga de pesos y reanudación |
| `evaluacion/` | Inferencia, métricas, extensión este y split vigente |
| `infraestructura/` | Logs, metadatos y salidas externas del lanzador |

Desde la raíz de floryacy:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
```

Los `__init__.py` permiten descubrimiento recursivo. Se pueden seleccionar por
nombre, conservando los comandos anteriores, por ejemplo `-p test_test_actual.py`.
Para un grupo, usar `-s test/datos`. No ejecutar herramientas de diagnóstico al
importar módulos de pruebas. Usar entradas temporales, sin entrenamientos completos.

La suite usa fixtures y CPU; no requiere datasets, TIFF, pesos ni preparación
acreditada del servidor. Requiere las dependencias Python del proyecto; esta
independencia de volúmenes no elimina esas dependencias.
`evaluacion/test_test_actual.py` valida preflight con metadatos y archivos mínimos
temporales: estos fixtures prueban presencia y contrato, no integridad raster.

La comprobación operativa queda fuera del descubrimiento de pruebas. Pedirla
explícitamente en floryacy:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python scripts/diagnostico/verificar_preparacion_test_actual.py
```

Lee el split vigente y su preparación real; falla si faltan volúmenes, metadatos
o máscaras, sin skips ni inferencia. No sustituye auditorías de bloques ni la
verificación de checkpoints del evaluador. Ver [TEST vigente](../docs/guias/test_actual.md).
Las pruebas de compatibilidad histórica cubren código aún disponible y no definen
el split vigente, que se toma de `config/split_extension_este.json`.

Las pruebas de infraestructura incluyen compatibilidad de shells en réplicas
/tmp y auditoría de imports de los dominios revisados. Las esperas/descargas/
modelos/procesos se simulan; los locks probados son archivos temporales.
Los wrappers se importan con src.<nombre>, sin agregar src globalmente a
sys.path. El catálogo de entradas y excepciones CLI está en src/INDICE.md.

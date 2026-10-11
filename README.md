# Córdoba IA — clasificación de cultivos

Repositorio principal: `floryacy:~/cordoba_ia_unne`. Esta copia es un mirror.
TEST vigente: **20HNJ y 20HNH**, diciembre–mayo, 24 canales.

- [CONTRIBUTING](CONTRIBUTING.md): reglas para agentes, ejecución y almacenamiento.
- [Mapa del código](src/INDICE.md): dominios, entradas actuales y compatibilidad.
- [Arquitecturas](model/README.md): modelos y bloques compartidos.
- [Documentación](docs/README.md): guías actuales, etiquetas y archivo histórico.
- [Configuración del split](config/split_extension_este.json).
- [Resultados y ubicaciones](artifacts/README.md).

Código en `src/` y `utils/`; pruebas en `test/`. Los datos y resultados físicos
viven en `/mnt/yacy_*`, según [almacenamiento](config/almacenamiento.json).

Ejecutar desde la raíz con `~/s4a-venv/bin/python` en floryacy (Python 3.11).
El entorno operativo se administra fuera del repo; ver [CONTRIBUTING](CONTRIBUTING.md).

# Diagnóstico manual

Ejecutar desde la raíz con `~/s4a-venv/bin/python`. No son pruebas automatizadas.

```bash
~/s4a-venv/bin/python scripts/diagnostico/escanear_sentinel.py \
  --base /mnt/yacy_1/prod/ferreyra/sentinel2_cordoba
~/s4a-venv/bin/python scripts/diagnostico/buscar_tile_por_coordenada.py \
  --lat -31.816667 --lon -63.6 \
  --dir-composites /mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_filtrado
```

El primer comando registra tile → fecha → bandas JP2; no certifica integridad
ni pertenencia al split. Su salida por defecto va a la corrida externa configurada;
`--salida` permite una ubicación explícita fuera del repo. Usar el lanzador para
separar inventarios por corrida. El segundo ejemplo conserva las coordenadas
aproximadas antes fijadas para Manfredi; se pueden pasar otras coordenadas.
La búsqueda comprueba huella, índice y muestra de la primera banda; no acredita
que el píxel tenga datos válidos ni que se haya usado para entrenar.

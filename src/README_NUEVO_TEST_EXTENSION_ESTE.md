# Nuevo TEST: extensión este

Preparación y evaluación iniciadas con `nohup ... &` el 09/10/2026 a las 19:22 (Argentina), en floryacy. Supervisor inicial PID 19987; continúa al cerrar terminal o SSH.

## Alcance

- Tiles nuevos: **20HNJ y 20HNH**.
- Cuatro campañas: **2018/2019, 2019/2020, 2020/2021 y 2021/2022**.
- Ventana fija: diciembre a mayo, cuatro bandas B02/B03/B04/B08, 24 canales, 10 m.
- U-Net multicampaña congelada, mejor checkpoint de VAL (época 60). Se copia el checkpoint al directorio del experimento y se verifica su SHA256 antes de inferir.
- Se guarda una configuración separada `config/split_extension_este.json`: TRAIN y VAL originales; TEST nuevo con ambos tiles; TEST anterior fuera de ese split. No se modifica la configuración de la corrida anterior ni sus métricas.

## Etapas en ejecución

1. Leer **todos los bloques y bandas** de los TIFF seleccionados descargados. Si hay corrupción, detener y reportar el archivo.
2. Generar **48 compuestos mensuales SCL**, dos workers. Directorios `composites_AAAA_AAAA_scl` dentro de la extensión este.
3. Auditar íntegramente esos compuestos y comprobar que cada grupo mensual comparte grilla.
4. Reproyectar por vecino más cercano el mapa INTA de cada campaña y verificar estrictamente los códigos. Código 20 se ignora exclusivamente para 2020/2021, según la excepción autorizada previamente. No se aplica esa excepción a 2021/2022.
5. Guardar **16 máscaras de clases** (dos criterios × dos tiles × cuatro campañas) y leer todos sus bloques.
6. Esperar el lock de inferencia GPU; ejecutar **ocho mapas** con la U-Net congelada, sin reentrenar ni ajustar con este TEST.
7. Guardar figuras de comparación, errores y exclusión, matrices de confusión, métricas por tile y agregadas por campaña, y resumen CSV.

No se generan arrays X/Y de parches: para la evaluación de mapas completos no hacen falta y ocuparían espacio adicional.

## Tratamiento del solapamiento espacial

Se publican dos resultados sobre las mismas predicciones:

- **Completa:** todos los píxeles con referencia INTA remapeada >0 del tile.
- **Independiente:** excluye píxeles que intersectan las grillas completas de cualquier tile TRAIN o VAL de las tres campañas originales. La rasterización usa `all_touched=True` para excluir también los píxeles tocados por el borde.

La exclusión es conservadora: puede excluir píxeles de grillas TRAIN/VAL que no llegaron a aportar un parche. Evita puntuar áreas ya incluidas geográficamente en los conjuntos anteriores. NoData se ignora como referencia; predecir 0 en una referencia válida cuenta como error. El contexto de una inferencia convolucional puede cruzar el borde de la máscara; no se afirma aislamiento de todo el contexto visual.

2021/2022 constituye una prueba temporal y espacial; las otras tres campañas son zonas nuevas en años ya usados para entrenar. La referencia es un mapa clasificado de INTA, no observaciones independientes de campo. Los nuevos compuestos 2019/2020 usan SCL, a diferencia del preprocesamiento original de esa campaña: comparar con las métricas antiguas mezcla diferencias de territorio y preparación.

## Rutas y logs

Base remota:
`/mnt/yacy_1/prod/ferreyra/extension_este_multicampana`

Resultados de esta ejecución:
`/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/evaluacion_20261009T192204`

Desde esta computadora:

```bash
ssh floryacy 'tail -f /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/ultima_preparacion.log'
```

Metadatos del supervisor: `ultima_preparacion.json`. El resultado final se registra en `preparacion.exitcode` (0 = éxito) y `fin.txt`. Durante inferencia, `progreso.json` indica campaña, tile y tarea. Antes de inferencia, el avance se sigue en el log.

## Verificación del código

Pasaron tres pruebas nuevas: excepción del código 20 exclusiva de 2021, NoData y predicción 0 como error, y exclusión parcial por solapamiento. También pasaron las cuatro pruebas existentes del postprocesado. Se verificaron sintaxis de Python y Bash, compatibilidad real del checkpoint con `SimpleUNet(24,5)`, y coincidencia del split original guardado con el usado para excluir áreas.

Scripts remotos: `src/preparar_extension_este.sh`, `src/preparar_evaluar_extension.py`, `test/test_extension_este.py`. Cambios preparados con git add, sin commit.

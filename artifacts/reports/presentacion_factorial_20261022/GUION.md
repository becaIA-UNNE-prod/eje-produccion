# Guion para una presentación de 12 a 15 minutos

Presentar las páginas 1 a 13. Las páginas 14 y 15 son anexos. La metodología está en las páginas 2 a 5, antes de los resultados. Abrir PRESENTACION.html, activar Modo exposición y usar las flechas del teclado. Escape vuelve al informe. El archivo funciona sin internet. Para compartir con anexos, conservar juntas ambas carpetas de reportes.

## 1. Superficie vs. campañas: cómo distribuir 3.072 parches

Abrir con la decisión que queremos tomar: cómo repartir una cantidad fija de ejemplos. El ensayo identifica un candidato general y muestra un problema pendiente con el maní. Los resultados son exploratorios y no incluyen una validación de esta comparación en Chaco.

## 2. Cuatro condiciones: superficie × campañas

Los factores se cruzan. No hay una superficie exclusiva de varias campañas: tanto tres como catorce tiles tienen una condición de una campaña y otra de tres. Para estudiar campañas se comparan las columnas de una misma fila. Para estudiar superficie se comparan las filas de una misma columna. En cada casillero hay 3.072 parches, no 3.072 por campaña. Fuente: protocolo.json y selecciones firmadas del factorial.

## 3. Para comparar campañas, conservamos las ubicaciones

El control espacial evita confundir diversidad temporal con nuevas ubicaciones. En la mezcla no se observa cada ubicación durante tres años, sino que se le asigna uno de los años. Cada ejemplo sigue conteniendo seis meses de una única campaña. Por lo tanto no es una secuencia temporal de tres años concatenados ni un aumento a 9.216 ejemplos. Fuente: protocolo.json y selección por condición.

## 4. Para comparar superficie, repartimos los mismos parches

Este ensayo responde cómo distribuir un presupuesto fijo. Si aumentáramos simultáneamente tiles y número de parches, no separaríamos la diversidad geográfica de la cantidad de datos. Los dominios de tres y catorce tiles son dos selecciones concretas, no cantidades demostradas como óptimas. La comparación no mide qué sucedería al aumentar a la vez territorio y parches. Fuente: protocolo.json.

## 5. Qué mantuvimos igual en los ocho entrenamientos

Los pesos de clase comunes son 0 para NoData, 0,793 para Fondo, 1,002 para Maíz, 0,861 para Soja y 3,922 para Maní. Se mantiene el régimen de optimización y la normalización. Semilla común no garantiza idéntica secuencia de aumentos entre arquitecturas. Las ventanas no se duplican dentro de una condición. La mezcla mantiene un total de 615 parches con maní. En las condiciones de 14 tiles hay 228 parches de 20HLJ. La auditoría IPAVE ignora conflictos únicamente en TRAIN, sin modificar VAL o TEST. El TEST se usó en trabajos anteriores y una época equivalente no asegura convergencia equivalente. Fuentes: protocolo.json, auditoria_mani.json y procedencia_checkpoints.json.

## 6. Mejor mIoU: U-Net, 14 tiles y 3 campañas

Leer primero la última pareja: U-Net obtuvo 54,57 de mIoU frente a 51,18 del Transformer. Después señalar que las parejas cambian de orden. El candidato general es el mejor observado con este presupuesto, no una superioridad universal de arquitectura.

## 7. Efecto de variar las campañas: de 1 a 3

En U-Net, mezclar campañas mejora 2,39 puntos en la 3 tiles y 4,81 en la 14 tiles. Segmenter cambia 0,26 y menos 1,34. La evidencia favorece la mezcla para U-Net en estas condiciones. No atribuir causalidad a un único mecanismo ni extrapolar a cualquier Transformer.

## 8. Efecto de variar la superficie: de 3 a 14 tiles

Ampliar el dominio obliga a repartir el mismo presupuesto entre más ubicaciones. U-Net empeora con una campaña y mejora con tres. Segmenter muestra el patrón contrario. No concluir que más superficie siempre ayuda o siempre perjudica.

## 9. El mejor promedio todavía oculta una falla en maní

Separar la decisión general de la decisión sobre maní. Frente a MNC, más del 94 % de los píxeles predichos como maní no coincide con esa clase. Esto no es una auditoría de campo. La referencia contiene apenas unos 0,084 % de píxeles de maní, de modo que su rareza hace especialmente importante medir falsos positivos.

## 10. El maní sí estuvo reforzado en el entrenamiento

Confirmar expresamente la inclusión del núcleo manisero. Es un refuerzo real, pero no garantiza cobertura exhaustiva de todos los lotes. IPAVE es una referencia auxiliar modelada. No se reetiquetaron VAL ni TEST y esta comparación no mide por separado la ganancia de incorporar IPAVE.

## 11. Un ejemplo permite ver los errores, no estimar su frecuencia

Leer las etiquetas de cada panel antes de comparar. El RGB, la referencia y ambas predicciones corresponden al mismo encuadre. Usar el caso para mostrar desacuerdos espaciales y volver a las métricas agregadas para hablar de frecuencia. No presentar esta ventana dirigida como un ejemplo típico.

## 12. El Transformer necesita una conclusión acotada

El mínimo tardío de Segmenter sugiere revisar convergencia, pero no demuestra que más épocas mejorarían el TEST. Otras limitaciones: una campaña individual frente a una mezcla específica y dominios concretos. El salto entre validación y TEST en maní requiere auditoría, sin atribuirlo automáticamente a etiquetas erróneas.

## 13. La siguiente decisión es validar maní, antes de ampliar la búsqueda

Cerrar con una decisión concreta: no buscar más modelos por inercia. Primero resolver si el sistema separa maní de las alternativas y si la referencia representa bien el territorio. Mantener el candidato general para comparar cualquier mejora futura sobre una base conocida.

## 14. Los ocho resultados, sin abreviaturas de experimentos

Anexo para preguntas. La mejor F1 de maní fue 10,74 del Transformer con 3 tiles y 1 campaña, prácticamente igual a 10,73 de U-Net con 3 tiles y 3 campañas. Ninguno alcanza una precisión adecuada para un mapa operativo de maní. Fuente: resumen_2021_2022.json.

## 15. Del relato a la evidencia completa

Usar esta página como índice único. Las métricas por otras campañas y los mapas completos quedan en el archivo técnico. No mezclar sus valores con los de este TEST principal.

## Preguntas previsibles

**¿Ganó U-Net?** Obtuvo el mejor mIoU con tres campañas y 14 tiles. El Transformer fue mejor en las dos condiciones de una campaña. Una sola semilla y mínimos de validación tardíos del Transformer limitan la conclusión.

**¿Agregamos maní intensivo de Córdoba?** Sí. Se incluyeron 228 parches de 20HLJ en cada condición de 14 tiles y un cupo de 615 parches con al menos 1 % de maní en cada condición. No se demuestra cobertura exhaustiva ni el efecto aislado del refuerzo.

**¿Más territorio equivale a más superficie de entrenamiento?** No. Se dispersan las mismas 3.072 ventanas sobre un dominio mayor. Los píxeles exactos de maní y los válidos tras ignorar conflictos no son iguales.

**¿Sirve para Chaco?** Este factorial no lo demuestra. La transferencia regional requiere una evaluación específica y separada.

**¿Por qué no usar accuracy como resultado central?** Las clases frecuentes pueden dominarla. mIoU promedia el solapamiento por clase, y maní necesita además precisión, recall y F1.

## Procedencia

Fuente numérica: ../factorial_unet_segmenter_20261022/metricas/resumen_2021_2022.json. Épocas de mínimo VAL: validacion_y_epocas.csv. Diseño y cupos: protocolo.json y auditoria_mani.json del mismo reporte. Caso visual: figuras/detalle_mani_amplia_tres_2021_2022_20HNH.png, seleccionado por maní de referencia. No se ejecutó entrenamiento ni inferencia para esta presentación.

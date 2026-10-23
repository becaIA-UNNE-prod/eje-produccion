# Guion para una presentación de 10 a 12 minutos

Presentar las páginas 1 a 11. Las páginas 12 y 13 son anexos. Abrir PRESENTACION.html, activar Modo exposición y usar las flechas del teclado. Escape vuelve al informe. El archivo funciona sin internet. Para compartir con anexos, conservar juntas ambas carpetas de reportes.

## 1. Superficie vs. campañas: cómo distribuir 3.072 parches

Abrir con la decisión que queremos tomar: cómo repartir una cantidad fija de ejemplos. El ensayo identifica un candidato general y muestra un problema pendiente con el maní. Los resultados son exploratorios y no incluyen una validación de esta comparación en Chaco.

## 2. Cambiamos la distribución, mantuvimos 3.072 parches

Presentar los dos factores y sus niveles. En cada dominio se conservan las ubicaciones al cambiar el tratamiento temporal. U-Net y Segmenter reciben la misma selección en cada condición. No son 3.072 parches por campaña: la mezcla completa suma 3.072.

## 3. Los ocho modelos enfrentaron el mismo examen

El control permite comparar las condiciones dentro de este ensayo. El TEST se usó en trabajos anteriores, por lo que no es una confirmación final sobre datos nunca observados. Se igualó el presupuesto de entrenamiento, no se demostró convergencia equivalente.

## 4. Mejor mIoU: U-Net, superficie extendida y 3 campañas

Leer primero la última pareja: U-Net obtuvo 54,57 de mIoU frente a 51,18 del Transformer. Después señalar que las parejas cambian de orden. El candidato general es el mejor observado con este presupuesto, no una superioridad universal de arquitectura.

## 5. Efecto de variar las campañas: de 1 a 3

En U-Net, mezclar campañas mejora 2,39 puntos en la superficie acotada (3 tiles) y 4,81 en la superficie extendida (14 tiles). Segmenter cambia 0,26 y menos 1,34. La evidencia favorece la mezcla para U-Net en estas condiciones. No atribuir causalidad a un único mecanismo ni extrapolar a cualquier Transformer.

## 6. Efecto de variar la superficie: de 3 a 14 tiles

Ampliar el dominio obliga a repartir el mismo presupuesto entre más ubicaciones. U-Net empeora con una campaña y mejora con tres. Segmenter muestra el patrón contrario. No concluir que más superficie siempre ayuda o siempre perjudica.

## 7. El mejor promedio todavía oculta una falla en maní

Separar la decisión general de la decisión sobre maní. Frente a MNC, más del 94 % de los píxeles predichos como maní no coincide con esa clase. Esto no es una auditoría de campo. La referencia contiene apenas unos 0,084 % de píxeles de maní, de modo que su rareza hace especialmente importante medir falsos positivos.

## 8. El maní sí estuvo reforzado en el entrenamiento

Confirmar expresamente la inclusión del núcleo manisero. Es un refuerzo real, pero no garantiza cobertura exhaustiva de todos los lotes. IPAVE es una referencia auxiliar modelada. No se reetiquetaron VAL ni TEST y esta comparación no mide por separado la ganancia de incorporar IPAVE.

## 9. Un ejemplo permite ver los errores, no estimar su frecuencia

Leer las etiquetas de cada panel antes de comparar. El RGB, la referencia y ambas predicciones corresponden al mismo encuadre. Usar el caso para mostrar desacuerdos espaciales y volver a las métricas agregadas para hablar de frecuencia. No presentar esta ventana dirigida como un ejemplo típico.

## 10. El Transformer necesita una conclusión acotada

El mínimo tardío de Segmenter sugiere revisar convergencia, pero no demuestra que más épocas mejorarían el TEST. Otras limitaciones: una campaña individual frente a una mezcla específica y dominios concretos. El salto entre validación y TEST en maní requiere auditoría, sin atribuirlo automáticamente a etiquetas erróneas.

## 11. La siguiente decisión es validar maní, antes de ampliar la búsqueda

Cerrar con una decisión concreta: no buscar más modelos por inercia. Primero resolver si el sistema separa maní de las alternativas y si la referencia representa bien el territorio. Mantener el candidato general para comparar cualquier mejora futura sobre una base conocida.

## 12. Los ocho resultados, sin abreviaturas de experimentos

Anexo para preguntas. La mejor F1 de maní fue 10,74 del Transformer con superficie acotada (3 tiles) y 1 campaña, prácticamente igual a 10,73 de U-Net con superficie acotada (3 tiles) y 3 campañas. Ninguno alcanza una precisión adecuada para un mapa operativo de maní. Fuente: resumen_2021_2022.json.

## 13. Del relato a la evidencia completa

Usar esta página como índice único. Las métricas por otras campañas y los mapas completos quedan en el archivo técnico. No mezclar sus valores con los de este TEST principal.

## Preguntas previsibles

**¿Ganó U-Net?** Obtuvo el mejor mIoU con tres campañas y superficie extendida (14 tiles). El Transformer fue mejor en las dos condiciones de una campaña. Una sola semilla y mínimos de validación tardíos del Transformer limitan la conclusión.

**¿Agregamos maní intensivo de Córdoba?** Sí. Se incluyeron 228 parches de 20HLJ en cada condición de superficie extendida (14 tiles) y un cupo de 615 parches con al menos 1 % de maní en cada condición. No se demuestra cobertura exhaustiva ni el efecto aislado del refuerzo.

**¿Más territorio equivale a más superficie de entrenamiento?** No. Se dispersan las mismas 3.072 ventanas sobre un dominio mayor. Los píxeles exactos de maní y los válidos tras ignorar conflictos no son iguales.

**¿Sirve para Chaco?** Este factorial no lo demuestra. La transferencia regional requiere una evaluación específica y separada.

**¿Por qué no usar accuracy como resultado central?** Las clases frecuentes pueden dominarla. mIoU promedia el solapamiento por clase, y maní necesita además precisión, recall y F1.

## Procedencia

Fuente numérica: ../factorial_unet_segmenter_20261022/metricas/resumen_2021_2022.json. Épocas de mínimo VAL: validacion_y_epocas.csv. Diseño y cupos: protocolo.json y auditoria_mani.json del mismo reporte. Caso visual: figuras/detalle_mani_amplia_tres_2021_2022_20HNH.png, seleccionado por maní de referencia. No se ejecutó entrenamiento ni inferencia para esta presentación.

# U-Net frente a Segmenter — 20/10/2026

Cobertura amplia, tres campañas y exactamente la misma selección de 3072 parches. Semilla 42, batch 32, 100 épocas, pesos comunes y checkpoints elegidos con VAL. TEST 20HNJ/20HNH; criterio independiente. Métricas en porcentaje.

| Modelo | Accuracy | mIoU | F1 macro | F1 maíz | F1 soja | F1 maní | Precisión maní | Recall maní |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNet | 83.87 | 54.77 | 64.65 | 78.68 | 87.32 | 8.80 | 4.90 | 43.23 |
| Segmenter | 81.29 | 51.49 | 61.84 | 77.19 | 85.81 | 5.50 | 2.90 | 52.09 |

La tabla corresponde a 2021/2022, ausente de TRAIN y VAL. Los otros años están en el CSV. Una sola semilla por arquitectura: resultados exploratorios, sin significación estadística. Igual presupuesto no implica igual convergencia. El transformer usa tokens de 16 y puede perder detalle de bordes; las vistas generales también reducen la resolución para visualizar. Los detalles de maní se eligieron por referencia, no por ventaja de un modelo: son ejemplos dirigidos, no muestras aleatorias. No se cambió ningún checkpoint con TEST.

Segmenter completó 100 épocas y eligió la 87 por pérdida VAL; U-Net eligió la 30. La corrida Segmenter terminó con código 1 al iniciar TEST por GPU no disponible, después de guardar los pesos. La evaluación se recuperó por separado con esos mismos pesos, sin reentrenar. Se conservan la corrida fallida y la recuperación.

Las imágenes muestran el mismo encuadre, paleta y referencia para ambos modelos; fuera del criterio independiente se muestra gris. Errores verdes=aciertos, rojos=desaciertos. RGB de febrero, no composición de todos los meses.

Abrir `index.html` para navegar las figuras. Los TIFF y pesos permanecen en floryacy.

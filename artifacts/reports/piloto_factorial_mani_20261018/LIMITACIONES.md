# Piloto factorial de maní: cuatro modelos y sus limitaciones

Decisión del 18/10/2026: detener las colas anteriores de 24 y nueve modelos
y sustituirlas por este piloto. Las corridas anteriores quedan canceladas,
con checkpoints, protocolos, métricas y logs conservados; no se reutilizan
sus pesos. La nueva cola es independiente y no espera que aquellas terminen.

## Pregunta y diseño

Comparar la extensión geográfica del dominio TRAIN y la diversidad temporal,
con igual cantidad de ejemplos y presupuesto de optimización, manteniendo
Maní y Soja como clases separadas.

| Modelo | Dominio espacial | Campañas de TRAIN |
|---|---|---|
| reducida_2020_2021 | 20HLG/20HLH/20HMG, subregión sur | 2020/2021 |
| reducida_tres | Misma subregión y mismas ubicaciones | 2018/2019 + 2019/2020 + 2020/2021 |
| amplia_2020_2021 | Dominio TRAIN ampliado, con 20HLJ | 2020/2021 |
| amplia_tres | Mismo dominio amplio y mismas ubicaciones | Las tres campañas |

U-Net desde cero, cinco clases (NoData/Fondo/Maíz/Soja/Maní), semilla 42,
3072 parches únicos de 256×256, 100 épocas sin parada temprana, batch 32,
9600 minibatches previstos por modelo. Se selecciona el checkpoint de menor
pérdida VAL. Diciembre–mayo, B02/B03/B04/B08: 24 canales, normalización /10000.
Los pesos de clase son comunes, derivados solo del pool TRAIN admisible.

Dentro de cada dominio se seleccionan las mismas 3072 ubicaciones para ambos
tratamientos temporales. Una campaña usa todas en 2020/2021; la mezcla asigna
un solo año a cada ubicación, exactamente 1024 por año. No se repiten lugares
entre años dentro del mismo modelo. Hay 615 parches con ≥656 píxeles de maní
(aproximadamente 1 %) en ambos tratamientos; en la mezcla son 205 por año.
Un flujo integral asigna años sin cambiar ubicaciones ni relajar esos cupos.
Si no hay una selección factible, el preparador falla antes de entrenar.

Se reutilizan las coordenadas acreditadas y las fuentes existentes; HLJ se
vuelve a acreditar contra todas sus etiquetas y muestras X. Se descartan
ventanas que tocan las grillas VAL/TEST y solapamientos entre ventanas TRAIN.
Se conserva VAL completo de 20HKJ/20HMH en las tres campañas. TEST mantiene
20HNJ/20HNH y cuatro campañas, con 2021/2022 fuera de TRAIN y VAL.

Los cuatro modelos usan la misma preparación TEST con exclusión de la unión
de todos los tiles TRAIN posibles + VAL. Es conservadora para el dominio
reducido, pero evita evaluar cada modelo en superficies diferentes. El split
global vigente no se altera: el protocolo pasa un split experimental explícito.
Los polígonos IPAVE son referencia auxiliar y no reemplazan etiquetas MNC.

## Limitaciones que condicionan las conclusiones

1. **Una sola semilla.** No estima la variación por inicialización, muestreo y
   optimización. Una diferencia observada es exploratoria; no justifica por sí
   sola afirmar superioridad estadística. Una segunda semilla sería otro ensayo
   completo de cuatro condiciones; no está encolada automáticamente.
2. **Un solo año individual.** Se eligió 2020/2021 por su cercanía temporal a
   2021/2022, antes de ejecutar este piloto. La comparación responde “ese año
   frente a esta mezcla”; no demuestra que tres campañas superen a cualquier
   campaña individual ni separa cantidad de años de identidad de los años.
3. **Superficie significa extensión/distribución del dominio.** El área sumada
   de los parches es idéntica: 3072 × 6,5536 km², aproximadamente 20.133 km²,
   sin solapamientos dentro de cada selección. Se comparan esos ejemplos
   concentrados en una subregión o dispersos en un dominio mayor; no un aumento
   de área total observada. Los dominios son anidados, las muestras no tienen
   por qué serlo. Es un contraste geográfico concreto, no múltiples réplicas
   de “superficies pequeñas” y “superficies grandes”.
4. **La ampliación incluye HLJ y otros ambientes.** El contraste espacial no
   aísla el efecto exclusivo de añadir HLJ. Tampoco separa paisaje, proporciones
   de cultivos, calidad de las imágenes y otros atributos geográficos.
5. **Cupo de maní aproximado por densidad.** Igualar parches con ≥1 % de maní
   no iguala la cantidad exacta de píxeles, lotes, bordes, pureza ni la distribución
   de las otras clases. Esos conteos se registran para interpretar diferencias.
   El enriquecimiento cambia la distribución respecto de un muestreo natural;
   aquí no hay un control uniforme para medir ese efecto por separado.
6. **Selección asistida por etiquetas de las tres campañas TRAIN.** Las
   ubicaciones deben ser válidas en los tres años y permitir cupos pareados.
   Esto excluye otros lugares y condiciona la población a la que generaliza el
   resultado. El tratamiento de una campaña usa solo ese año en los gradientes,
   pero su marco de muestreo, pesos comunes y VAL utilizan información de las
   tres campañas permitidas. No es una ausencia absoluta de exposición a los
   otros años durante el diseño/selección del checkpoint.
7. **Procesamiento y referencia cambian con el año.** La mezcla no separa
   meteorología, rotación, cambios de etiquetas y procesamiento de composites.
   En particular, 2019/2020 conserva el procesamiento acreditado diferente del
   SCL usado por las otras dos campañas. No se uniformiza retrospectivamente.
8. **Referencia cartográfica, no verdad de campo independiente.** MNC contiene
   incertidumbre. IPAVE también publica cartografía modelada de uso regional;
   la concordancia entre ambas no acredita validación independiente a campo.
   El mapa auxiliar no se usa para reetiquetar ni como ausencia=negativo.
9. **TEST pequeño y ya consultado históricamente.** Hay dos tiles, una campaña
   temporal reservada y referencias compartidas con el ecosistema de preparación.
   Los píxeles no son réplicas independientes. Los resultados de otros modelos
   en este TEST ya se habían revisado: no es un holdout completamente desconocido
   para todo el proceso de investigación. No elegir pesos, cupos ni umbrales
   usando los resultados del nuevo TEST. La exclusión geométrica no garantiza
   aislamiento de todo el contexto convolucional. HLJ fue TEST histórico y sus
   métricas antiguas no son evaluación ciega de modelos que ahora lo entrenan.
10. **Mismo presupuesto, no mismo grado de convergencia.** Cien épocas fijan
    el presupuesto, pero las cuatro condiciones pueden converger a distinta
    velocidad. Se conserva el mejor VAL y el historial completo; el scheduler
    responde a VAL. La semilla no garantiza determinismo bit a bit en GPU.

## Lectura prevista de resultados

Primarios: F1, precisión y recall de Maní; revisar confusiones Soja↔Maní y
costos sobre las demás clases. Conservar mIoU y F1 macro, mapas completos y
el criterio independiente principal. No comparar directamente con métricas
agrupadas de cuatro clases ni mezclar parches con mapas completos.

Contrastes descriptivos predefinidos:
- Efecto espacial en una campaña: amplia_2020_2021 − reducida_2020_2021.
- Efecto espacial en la mezcla: amplia_tres − reducida_tres.
- Efecto temporal dentro de cada dominio: tres − 2020_2021.
- Interacción: diferencia entre los dos efectos temporales.

Informar por tile y campaña, dando especial atención a 2021/2022, sin tratar
los miles de píxeles como réplicas de la semilla. Si las diferencias son pequeñas,
la conclusión será que este piloto no las resuelve. Antes de una afirmación
fuerte, repetir las cuatro condiciones con otra semilla o ampliar la evaluación
independiente, conservando el protocolo y un presupuesto comparable.

## Ejecución, trazabilidad y almacenamiento

Configuración: `config/piloto_factorial_mani.json`.
Entrada: `scripts/entrenamiento/piloto_factorial_mani.py --preparar --ejecutar`.
Salida física:
`/mnt/yacy_3/prod_bis/ferreyra/dataset/piloto_factorial_mani_20261018_v1/`.
Cada modelo tiene una corrida propia del lanzador estándar y sus checkpoints.
El protocolo conserva fuentes, selección, conteos, semilla, hashes y versión
del asignador. Los arrays, TIFF, pesos y logs permanecen en /mnt/yacy_*.
El mirror guarda solo código y metadatos ligeros.

El supervisor toma los locks de entrenamiento y ejecuta solo cuatro modelos
en secuencia. Un fallo o cambio de código/configuración/manifiestos detiene
la cola. Se inicia con nohup, stdin /dev/null y &; el cierre de la terminal no
lo interrumpe. La estimación inicial es de 9–12 horas de ejecución, a revisar
con los tiempos de las primeras épocas y evaluaciones; no es un plazo garantizado.
No se mantienen activas las colas anteriores ni se encolan repeticiones extra.

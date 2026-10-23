# Factorial U-Net y Segmenter con auditoría de maní — 21/10/2026

Ocho entrenamientos nuevos autorizados: dos arquitecturas × dos coberturas ×
dos tratamientos temporales. Chaco queda para una etapa posterior. La cola se
lanzó con `nohup ... &`, stdin `/dev/null` y salidas físicas en `/mnt/yacy_3`.
Se verificó que el supervisor PID 30398 quedó con PPID 1 después de cerrar SSH.
Ese PID registra el lanzamiento, no una referencia permanente.

## Condiciones congeladas

| Condición | Cobertura | Campañas | Arquitecturas |
|---|---|---|---|
| reducida_2020_2021 | 20HLG, 20HLH, 20HMG | 2020/2021 | U-Net, Segmenter |
| reducida_tres | Mismos tres tiles y ubicaciones | 2018/19, 2019/20, 2020/21 | U-Net, Segmenter |
| amplia_2020_2021 | Trece tiles originales TRAIN + 20HLJ | 2020/2021 | U-Net, Segmenter |
| amplia_tres | Mismo dominio amplio y ubicaciones | Las tres campañas | U-Net, Segmenter |

3072 parches únicos de 256×256 por modelo; 1024 por campaña en la mezcla;
semilla 42, batch 32, 100 épocas sin parada temprana, entrenamiento desde cero.
Cada par de arquitecturas comparte exactamente el manifiesto y las etiquetas.
Se conservan los marcos espaciales pareados del piloto del 18/10 para no
introducir otro cambio de muestreo. La amplia contiene 228 parches de 20HLJ.
Extensión geográfica y distribución varían; el área sumada de ventanas no.

Leyenda: Fondo, Maíz, Soja y Maní, con NoData/ignorados separados. Fondo
incluye todo lo que no pertenece a esos tres cultivos. La implementación
conserva cinco logits (incluido NoData) por compatibilidad. Las cuatro clases
válidas tienen métricas; una predicción NoData sobre referencia válida sigue
contando como error y no debe desaparecer al presentar la matriz resumida.

El checkpoint se elige por pérdida VAL. VAL completo: 20HKJ/20HMH, tres
campañas. TEST: 20HNJ/20HNH, cuatro campañas y máscara de exclusión común.
2021/2022 no se usa en TRAIN/VAL. El split global no se modifica; se reutiliza
el split experimental común acreditado. No se reactiva ninguna cola anterior.

## Auditoría y tratamiento conservador de IPAVE

Referencias: GeoJSON IPAVE 2018/19, 2019/20 y 2020/21 ya descargados en
`/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/data/referencias/mani_idecor_20261018_r2/`.
Se rasterizan por ventana utilizando su transformación original; geometrías
inválidas se reparan solo en la derivación. No se descargan nuevos composites.

Regla fija aplicada exclusivamente a TRAIN: donde IPAVE identifica maní y MNC
asigna otra clase válida, el píxel pasa a ignorado. No se reasigna ningún
cultivo. Ausencia en IPAVE no se trata como negativo; se conservan los positivos
MNC fuera de IPAVE. VAL/TEST permanecen con su referencia MNC original.
Son máscaras Y externas para los ejemplos seleccionados; los X/Y compartidos
originales quedan intactos. Cada overlay acredita SHA256 e índices ordenados.

| Condición | Maní MNC (ha) | Maní MNC∩IPAVE (ha) | Desacuerdo ignorado (ha) |
|---|---:|---:|---:|
| Reducida, una campaña | 36.525,70 | 32.391,87 | 28.214,03 |
| Reducida, tres campañas | 33.815,98 | 26.977,21 | 41.993,47 |
| Amplia, una campaña | 33.281,56 | 28.990,32 | 28.777,48 |
| Amplia, tres campañas | 31.328,72 | 24.452,97 | 33.273,34 |

Estas cifras corresponden a las ventanas seleccionadas; no a toda la provincia.
Los 615 parches con ≥656 píxeles MNC de maní se conservan en cada condición,
205 por campaña en la mezcla. Los píxeles de maní no se igualan exactamente
entre condiciones. Se comprobó maní coincidente en cada campaña efectiva de
cada condición; esto confirma representación, no cobertura exhaustiva del
núcleo manisero ni validación de campo independiente.

Pesos comunes: `[0, 0.793, 1.002, 0.861, 3.922]`. Se calcularon una sola vez
con la unión única `(campaña, ubicación)` de TRAIN después de ignorar
desacuerdos, sin utilizar VAL/TEST. Se utilizan igual en los ocho modelos.

El mapa `mapa_nucleo_mani.png` muestra las referencias IPAVE y las ventanas:
ocre = contornos IPAVE de campañas TRAIN; rojo = ventanas con ≥1% de maní
MNC en 2020/21; azul = otras ventanas. La geometría se conserva también como
GeoJSON. La densidad se ilustra sin consultar predicciones.

## Supervisión, validación y tiempos

Pasaron 28 pruebas: overlays limitados a ignorados, integridad e índices,
presupuesto/pareo, Segmenter y checkpoints, CLI, TEST vigente y exclusiones.
Preparación completada con código 0 en la corrida
`20261021T120551Z_preparacion_factorial8_auditoria_mani_7115e5be4845`.

Cola padre:
`20261021T120943Z_entrenamiento_factorial8_unet_segmenter_f1b1c0ab2ae3`.
Inicio: 21/10/2026 12:09:43 UTC (09:09:43 de Argentina).
Primer modelo: `reducida_2020_2021_unet`; luego Segmenter de esa condición,
el par amplia_tres, el par amplia_2020_2021 y el par reducida_tres.

Se mantienen locks histórico y runtime de entrenamiento durante toda la cola.
Cada modelo primero entrena y guarda VAL/pesos con `--omitir-evaluacion-test`;
el proceso sale y solo entonces otro proceso evalúa TEST. Esto evita mantener
simultáneamente dos contextos CUDA cuando la GPU usa Exclusive Process, causa
compatible con el fallo de evaluación de Segmenter del ensayo anterior.
La cola registra cada etapa, se detiene ante fallos y preserva sus evidencias.
Las versiones de código, referencias, selección y split se verifican por firma
antes de cada modelo. No editar esas fuentes durante la cola.

Estimación inicial: **15–18 horas totales** desde el inicio, incluyendo las
ocho evaluaciones. Se basa en cuatro U-Net de aproximadamente dos horas y
cuatro Segmenter de aproximadamente una hora y media, más margen de I/O.
No es un plazo garantizado: cambiarán según carga del disco y tiempos efectivos.
Ventana orientativa de finalización: madrugada del 22/10, aproximadamente
00:10–03:10 de Argentina. La corrida sigue siendo exploratoria: una semilla,
presupuesto igual sin garantía de convergencia igual y TEST ya consultado en
trabajos anteriores. No se estiman intervalos de confianza entre semillas.

## Ubicaciones

Salida física:
`/mnt/yacy_3/prod_bis/ferreyra/dataset/factorial_unet_segmenter_mani_20261021_v1/`.
Consultar `estado.json`, `lanzamiento.log`, `protocolo.json`,
`auditoria_mani.json`, `selecciones/`, `etiquetas_train/` y `experimentos/`.
Al finalizar, `comparacion_ocho_modelos.json` reúne las métricas de los ocho.
Los pesos, TIFF, overlays y logs permanecen físicamente fuera del repositorio.

Copia ligera local de protocolo, auditoría y mapa:
`artifacts/reports/factorial_unet_segmenter_mani_20261021/`.
El código y este documento se sincronizan entre floryacy y el mirror sin
descartar cambios pendientes ni crear commits.

## Resultados finales — 22/10/2026

Terminaron los ocho entrenamientos y las ocho evaluaciones con código 0.
Duración de la cola: 14 h 40 min 23 s. Finalizó el 21/10 a las 23:50 de
Argentina (22/10 02:50 UTC). No se iniciaron entrenamientos adicionales para
este análisis.

### Campaña no vista 2021/2022, TEST independiente

Métricas en porcentaje, sobre los mismos 177.822.530 píxeles y las mismas
referencias MNC. No se promedian F1 por tile: se usa la matriz agregada.

| Condición | Modelo | Accuracy | mIoU | F1 macro | F1 maní | Precisión maní | Recall maní |
|---|---|---:|---:|---:|---:|---:|---:|
| Reducida, una | U-Net | 81,20 | 50,81 | 61,25 | 5,14 | 3,02 | 17,42 |
| Reducida, una | Segmenter | 81,34 | 51,80 | 62,85 | 10,74 | 5,86 | 64,92 |
| Reducida, tres | U-Net | 82,06 | 53,20 | 63,86 | 10,73 | 5,92 | 57,36 |
| Reducida, tres | Segmenter | 81,59 | 52,05 | 62,77 | 8,97 | 4,73 | 86,94 |
| Amplia, una | U-Net | 78,18 | 49,75 | 61,21 | 9,95 | 5,48 | 53,65 |
| Amplia, una | Segmenter | 82,50 | 52,53 | 62,74 | 6,53 | 3,54 | 41,12 |
| Amplia, tres | U-Net | 84,09 | 54,57 | 64,34 | 7,71 | 4,31 | 36,50 |
| Amplia, tres | Segmenter | 81,89 | 51,18 | 60,96 | 1,36 | 0,74 | 8,12 |

### Efectos descriptivos de cobertura y campañas

Diferencias en puntos porcentuales; positivas significan aumento de la métrica.

| Contraste | U-Net Δ mIoU | Segmenter Δ mIoU | U-Net Δ F1 maní | Segmenter Δ F1 maní |
|---|---:|---:|---:|---:|
| Tres menos una campaña, cobertura reducida | +2,39 | +0,26 | +5,59 | −1,77 |
| Tres menos una campaña, cobertura amplia | +4,81 | −1,34 | −2,24 | −5,17 |
| Amplia menos reducida, una campaña | −1,06 | +0,73 | +4,81 | −4,22 |
| Amplia menos reducida, tres campañas | +1,37 | −0,87 | −3,03 | −7,61 |

El resultado depende de la combinación de factores. Segmenter supera a U-Net
en mIoU con una campaña en ambas coberturas; U-Net supera a Segmenter con
tres campañas en ambas. La mayor mIoU general corresponde a U-Net amplia/tres.
Ni ampliar cobertura ni agregar campañas mejora uniformemente todas las
clases o arquitecturas.

Maní sigue sin resolverse: los mayores F1 (10,74 y 10,73) son prácticamente
iguales. Todas las precisiones son inferiores al 6%; bajo la referencia MNC,
más del 94% de los píxeles predichos como maní son falsos positivos.
El alto recall de Segmenter reducida/tres (86,94%) viene acompañado de
precisión de solo 4,73%. No es evidencia de un mapa de maní confiable.

### VAL, convergencia y alcance de la conclusión

Las épocas de menor pérdida VAL fueron 16, 22, 43 y 57 para U-Net y
90, 93, 98 y 99 para Segmenter. Ese comportamiento exige cautela: se igualó
presupuesto, pero no se demostró convergencia equivalente. Que Segmenter alcance
su mínimo cerca del final no prueba que más épocas vayan a resolverlo.
El F1 de maní en VAL estuvo entre 44,45% y 57,18%, muy por encima del
1,36–10,74% del TEST no visto; hay una brecha importante de generalización
y/o referencia y prevalencia que requiere análisis por lote.

Mantendría U-Net amplia/tres como referencia para desempeño general en este
ensayo. Antes de invertir en otro entrenamiento, revisaría lotes de maní,
falsos positivos y consistencia de referencia. Estos datos no permiten
afirmar superioridad general de CNN frente a transformers ni significación
estadística: hay una sola semilla por celda y un régimen de optimización.

No se atribuye el cambio frente al piloto previo exclusivamente a IPAVE:
también cambiaron los pesos comunes. Las nuevas máscaras de ignorados solo
afectan TRAIN; VAL/TEST conservan MNC. Acuerdo con MNC no equivale a una
verificación independiente a campo. Se igualaron parches y pasos, no píxeles
de maní ni cantidad de píxeles válidos después de ignorar discrepancias.

### Reporte y figuras locales

Ruta local:
`artifacts/reports/factorial_unet_segmenter_20261022/`.

- `index.html`: galería con tabla principal y filtro por condición, campaña y tile.
- `RESULTADOS.md`: informe completo de resultados, contrastes y limitaciones.
- `figuras/`: 79 figuras nuevas: tres resúmenes, cuatro curvas, 32 matrices,
  32 panoramas pareados y ocho detalles de maní de 2021/2022.
- `metricas/`: resultados de los dos criterios y cuatro campañas, contrastes
  en puntos porcentuales, métricas VAL, épocas y procedencia de checkpoints.
- `mapa_nucleo_mani.png`: copia del mapa de auditoría previo al entrenamiento.
- `sha256.json`: integridad de las copias.

Las matrices presentan conteos y normalización por referencia. Se ignora la
fila NoData de referencia, pero se conserva una columna NoData predicha si
contiene errores, para no perderlos. Las figuras pareadas usan el mismo encuadre,
paleta y máscara independiente. Las ventanas de detalle se eligieron por mayor
presencia de maní de referencia, sin consultar predicciones; son ejemplos
dirigidos, no muestras aleatorias.

Fuente física del reporte:
`/mnt/yacy_3/prod_bis/ferreyra/dataset/factorial_unet_segmenter_mani_20261021_v1/resultados_20261022_v1/`.
Corrida de análisis:
`20261022T112557Z_analisis_comparacion_factorial8_d4a2561aa724`.
Pesos, TIFF y overlays permanecen en floryacy. Solo se sincronizan figuras,
métricas, código y documentación.


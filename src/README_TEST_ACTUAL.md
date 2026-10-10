# TEST vigente integrado

Desde el 10/10/2026, los flujos relevantes de diciembre–mayo usan **20HNJ y 20HNH** como TEST, con criterio principal **fuera de las grillas completas de TRAIN y VAL**. Se conservan también las métricas de los tiles completos.

## Configuración y entrenamiento

La fuente de configuración es `config/split_extension_este.json` en floryacy. TRAIN y VAL conservan los tiles del experimento original. 20HLK y 20HLJ dejan de participar en el TEST vigente; sus resultados previos quedan como históricos.

`src/1_train_cordoba.py` carga esta configuración por defecto en modo multicampaña y en las corridas de diciembre–mayo con `--sin-2017`. Solo carga arrays de TRAIN y VAL; el TEST se evalúa como mapas completos al finalizar, con el mejor checkpoint de VAL. Antes de entrenar comprueba que la preparación del TEST corresponda a TRAIN y VAL actuales. Si cambian esas particiones, exige regenerar las exclusiones.

Las métricas de VAL por parches permanecen en `metricas_multicampana.json`. Las métricas e imágenes de TEST vigente quedan en `evaluacion_test_actual/`, más `test_actual.json` en el directorio del experimento. No se mezclan métricas de parches de 256 con métricas de mapas inferidos en bloques de 512.

`--split-config` permite elegir la configuración; `--test-historico` reproduce el TEST antiguo de parches. Las ventanas distintas de diciembre–mayo mantienen el flujo histórico, y el TEST nuevo requiere 24 canales.

## Inferencia y comparación de modelos existentes

`src/postprocesar_multicampana.py` y el wrapper `src/inferencia_multicampana.sh` ahora usan el evaluador de TEST vigente por defecto. Para reproducir la inferencia antigua debe usarse `--historico` en el script Python.

Se lanzó con nohup y & la comparación de:

- U-Net multicampaña, entrenada con 2018/2019, 2019/2020 y 2020/2021.
- U-Net 2019/2020.
- FCRTransUNet 2019/2020.

Son **24 mapas** (tres modelos × dos tiles × cuatro campañas). Se reutilizan las predicciones multicampaña existentes únicamente si coinciden SHA256 de pesos y firmas de las seis entradas. Las demás se generan con checkpoints congelados nuevos dentro del reporte. No se reentrena ni se seleccionan pesos usando TEST.

Cada modelo recibe los mismos compuestos mensuales y se evalúa con exactamente las mismas máscaras. Se crean imágenes individuales, comparaciones por tile, matrices de confusión, curvas históricas (con advertencia de distintos VAL), métricas JSON y resumen CSV. Al finalizar cada experimento recibe un `test_actual.json` que referencia esta comparación y sus métricas.

La comparación mezcla arquitecturas y años de entrenamiento; no aísla solamente el efecto arquitectónico. 2021/2022 es un año nuevo para los tres modelos; en los otros años la exposición temporal depende del modelo. La referencia es INTA, no verdad de campo independiente. Permanece la advertencia de marzo de 2019 con ~6,12 % de relleno de menor azul.

## Ejecución y resultados

Inicio: 10/10/2026 09:24 (Argentina). Supervisor inicial PID 29188.

Resultados remotos:
`/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/comparacion_test_actual_20261010T092436`

Log desde esta computadora:

```bash
ssh floryacy 'tail -f /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/ultima_comparacion_test_actual.log'
```

Copia local automática de figuras y métricas: [resultados_test_actual_comparacion_20261010](resultados_test_actual_comparacion_20261010/). Los TIFF y pesos quedan en floryacy por tamaño.

El cierre se registra en `inferencia.exitcode`; la actualización final de títulos y leyendas se registra en `figuras.exitcode`. Ambos deben valer 0 antes de la copia local. `progreso.json` muestra el mapa que se está procesando.

## Validación y Git

Pasaron tres pruebas nuevas de configuración, disjunción de particiones, rechazo de exclusiones obsoletas y ejecución real de `--verificar-datos` sin arrays de TEST; cuatro pruebas de etiquetas/exclusión, cuatro de inferencia y tres de cachés/checkpoints. Se acreditó la compatibilidad real de los tres checkpoints y se inspeccionó una figura generada.

Cambios preparados con git add, sin commit. Los resultados históricos y los pesos originales se conservan.

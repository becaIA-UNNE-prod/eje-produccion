# TEST vigente: 20HNJ y 20HNH

Fuente: [config/split_extension_este.json](../../config/split_extension_este.json).
Aplica al flujo diciembre–mayo de 24 canales desde el 10/10/2026.
TRAIN/VAL mantienen sus tiles; 20HLK/20HLJ quedan fuera del TEST vigente.
No elegir checkpoints ni ajustar modelos con TEST.

## Preparación y criterio científico

Dos tiles × cuatro campañas (2018/2019 a 2021/2022), seis meses y cuatro bandas
B02/B03/B04/B08 a 10 m: 48 composites mensuales.
La preparación acreditada está en `preparacion_base` de la configuración.
Sus 16 máscaras cubren dos criterios por tile y campaña:

- **Independiente**, principal: excluye las grillas completas de TRAIN/VAL,
  rasterizadas con `all_touched=True`.
- **Completa**, descriptiva: conserva toda referencia remapeada válida del tile.

NoData de referencia se ignora; predecir 0 donde la referencia es válida cuenta
como error. La exclusión es conservadora y no garantiza aislamiento de todo el
contexto visual convolucional. Si cambian TRAIN/VAL, regenerar exclusiones:
el preflight rechaza una preparación con particiones obsoletas.
No se necesitan arrays X/Y del TEST para esta evaluación de mapas.

El orden general y la secuencia SCL se describen en
[Multicampaña](multicampana.md); para TEST usar siempre los tiles y la preparación
acreditada de esta guía.

Shells en `scripts/preparacion/`: `descargar_extension_este.sh DIRECTORIO_LOGS`
y `preparar_extension_este.sh DIRECTORIO_RESULTADOS`. El shell llama por `-m` a
`src.datos.preparar_extension` y `src.evaluacion.evaluar_extension`;
`src/preparar_evaluar_extension.py --modo preparar|evaluar` conserva compatibilidad; las rutas antiguas de los shells delegan. Antes de relanzarlos revisar
configuración, salidas, locks y metadatos de la preparación existente. Auditar
íntegramente fuentes/composites/máscaras; una lectura de ventana de descarga
no acredita todos los bloques. La leyenda depende de la campaña, especialmente
el código 0 de 2018/2019 frente a 2019/2020 y la excepción 20 de 2020/2021.

## Evaluar checkpoints existentes

Desde la raíz del repo, con el entorno `~/s4a-venv/bin/python`:

```bash
~/s4a-venv/bin/python -m src.evaluacion.test_actual \
  --exp /mnt/yacy_3/prod_bis/ferreyra/dataset/EXPERIMENTO \
  --salida /mnt/yacy_3/prod_bis/ferreyra/cordoba_ia_unne/artifacts/runs/NUEVA_CORRIDA \
  --verificar
```

Sustituir los nombres de ejemplo por rutas reales. `--verificar` realiza preflight
sin inferencia; requiere checkpoints compatibles. Quitar esa opción inicia mapas
y métricas. Por defecto compara el principal con U-Net y FCRTransUNet 2019/2020;
`--sin-comparar` evalúa solo el principal. `--modelo` permite explicitar arquitectura.
El evaluador verifica pesos de 24 canales/5 clases, grillas y preparación.
La ejecución completa requiere GPU, congela pesos y acredita SHA256 y entradas.
Una predicción existente se reutiliza solo con firmas compatibles.

El entrenador y `scripts/evaluacion/inferencia_multicampana.sh` usan este TEST por defecto
en los flujos correspondientes; el postprocesador ofrece `--historico` para
reproducir el anterior. No adaptar los scripts históricos 8/9/10 cambiando tiles.

## Salidas e interpretación

Guardar protocolo, predicciones, métricas por tile/campaña/criterio y figuras en
la corrida externa. La comparación existente de tres modelos produce 24 mapas.
Los tres comparten entradas y máscaras, pero mezclan arquitectura y campañas de
entrenamiento: no aíslan el efecto arquitectónico. 2021/2022 es temporalmente
nuevo para los tres; los demás años tienen distinta exposición de entrenamiento.
La referencia INTA no es una verdad de campo independiente.
Marzo de 2019 en 20HNJ conserva aproximadamente 6,12 % de relleno de menor azul.
No comparar métricas de parches 256 con mapas inferidos en bloques 512.

El [índice de artefactos](../../artifacts/README.md) localiza el reporte existente.
Su cierre se verificó el 10/10/2026: `inferencia.exitcode` y `figuras.exitcode`
valían 0. Eso acredita aquella ejecución, no un proceso activo permanente.
Los detalles y recuperación están en el [historial](../historico/experimentos_20261006_10.md).

## Ubicación de las entradas

El evaluador está en `src/evaluacion/test_actual.py` y el regenerador en
`src/figuras/regenerar_test_actual.py`. Desde la raíz, ejecutar por -m; las
rutas anteriores de src conservan compatibilidad. Para figuras existentes:
`python -m src.figuras.regenerar_test_actual --salida CORRIDA [--esperar]`.
Conservar la preparación acreditada y revisar protocolo/estado antes de ejecutar.

## Orden de los shells de extensión

Si faltan fuentes, descargar_extension_este.sh busca/descarga/selecciona las
cuatro campañas para 20HNJ/20HNH. Después, preparar_extension_este.sh audita
fuentes, compone, prepara máscaras y finalmente evalúa una U-Net congelada con
el comando mixto existente, conservando su lock GPU. Ambos están bajo
scripts/preparacion/. No relanzar sobre entradas acreditadas solo por un traslado.
Para evaluar otro checkpoint con el protocolo vigente, usar la entrada canónica
src.evaluacion.test_actual descrita arriba; no asumir que el shell mixto es
intercambiable con ella. El orden detallado de SCL está en la guía multicampaña.

## Coordinación con lock y estado

Desde la raíz, `bash scripts/evaluacion/inferencia_multicampana.sh EXP SALIDA`
adquiere inferencia.lock y registra PID, inicio, fin y exitcode en SALIDA.
Ambos directorios son explícitos; SALIDA debe existir en almacenamiento externo.
El shell llama a src.evaluacion.postprocesar_multicampana por -m, con
--lock-adquirido, y deriva al evaluador canónico. La entrada shell anterior delega.
El dispatcher acepta --historico desde Python para reproducir el protocolo
anterior; conserva su propio preflight/split y no aplica exclusiones vigentes.
Los wrappers anteriores conservan exports; main importado del postprocesador
sigue siendo histórico. Para API/CLI vigente usar src.evaluacion.test_actual.

## Pruebas pequeñas y comprobación operativa

Las pruebas `test/evaluacion/test_test_actual.py` usan fixtures temporales;
no leen la preparación acreditada. La comprobación real se solicita por separado:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/s4a-venv/bin/python scripts/diagnostico/verificar_preparacion_test_actual.py
```

Verifica metadatos, TRAIN/VAL, tiles/campañas y presencia de máscaras completas
e independientes. Solo lee; falla ante incompatibilidades o archivos faltantes,
sin skips. No valida contenido raster, hashes, grillas ni checkpoints: conservar
las auditorías y el evaluador --verificar para esos controles.
--split-config permite seleccionar explícitamente otra configuración.

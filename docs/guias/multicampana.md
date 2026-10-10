# Preparación y entrenamiento multicampaña

Referencia: `floryacy:~/cordoba_ia_unne`. Entorno: `~/s4a-venv/bin/python`.
Almacenamiento y lanzamientos: [CONTRIBUTING](../../CONTRIBUTING.md).

## Procedimiento para correr el flujo

1. Conectarse con `ssh floryacy` y hacer `cd ~/cordoba_ia_unne`.
   Leer CONTRIBUTING, revisar `git status --short`, almacenamiento y split.
   Usar `~/s4a-venv/bin/python`; conservar los cambios pendientes.
2. Identificar las entradas existentes de las campañas de la tabla siguiente:
   fuentes, composites, máscaras y arrays. Revisar sus procedencias y grillas.
   No descargar ni reconstruir datos ya acreditados por el solo hecho de correr
   el proyecto. Si faltan entradas, revisar los comandos de preparación de la
   campaña correspondiente y sus salidas antes de ejecutarlos.
3. Ejecutar las verificaciones de preparación y arrays de la sección
   «Verificar antes de preparar o entrenar». Revisar también los JSON de
   procedencia; estas verificaciones no sustituyen su comparación.
4. Si hay que preparar TRAIN/VAL, revisar `src/preparar_multicampana.sh` y
   ejecutar esa preparación con log externo según CONTRIBUTING. El proceso
   audita composites, prepara máscaras MNC2021 y genera arrays; requiere
   espacio y no entrena. Reutiliza salidas solo con procedencia compatible.
5. Crear un directorio nuevo de experimento en `/mnt/yacy_*`, verificar recursos
   y ejecutar el shell de entrenamiento de la sección «Entrenar y evaluar».
   Seleccionar el checkpoint con VAL; el cierre multicampaña evalúa TEST vigente.
6. Para evaluar un checkpoint existente sin entrenar, seguir
   [TEST vigente](test_actual.md): primero `--verificar`, luego evaluación en
   una corrida nueva. Usar la preparación acreditada 20HNJ/20HNH; no sustituirla
   por los arrays históricos de 17 tiles.
7. Revisar código de salida, logs, protocolo, métricas y figuras en la corrida
   externa. Distinguir mapas completos del criterio independiente principal.
   Registrar decisiones duraderas en documentación y conservar los artefactos
   en los discos operativos.

### Entradas específicas de máscaras

Para revisar argumentos, usar `~/s4a-venv/bin/python -m` seguido de:
`src.datos.mascaras_mnc_2018_2019`, `src.datos.mascaras_nivel3_2017_2018`
o `src.datos.mascaras_mnc_2020_2021`, con `--help`.
Son alternativas por fuente/campaña, no tres pasos obligatorios consecutivos.
Nivel3 requiere `--ref` con su grilla COG. MNC2021 admite
`--solo-verificar` y exige la política explícita para el código 20 desconocido.
Las rutas numeradas anteriores siguen funcionando.
`src.datos.moldear_mascaras` conserva la prueba original fija sobre JP2;
no es el preparador multicampaña y no debe ejecutarse como paso del flujo vigente.

## Entradas

Base existente: `/mnt/yacy_3/prod_bis/ferreyra/`.

| Campaña | Composites bajo la base | Dataset bajo `dataset/` |
|---|---|---|
| 2018/2019 | `composites_2018_2019_scl` | `train_cordoba_dicmay_1819_scl_f16` |
| 2019/2020 | `composites_dicmay_2019_2020` | `train_cordoba_dicmay_f16` |
| 2020/2021 | `composites_2020_2021_scl` | `train_cordoba_dicmay_2021_f16` |

Cada campaña preparada contiene 17 tiles y seis meses diciembre–mayo.
Por mes se usan B02/B03/B04/B08, en ese orden: 24 canales. Parches 256×256,
paso 256; X float16 normalizado /10000, Y uint8 remapeado; carga por mmap.
Las máscaras compartidas viven en la ruta configurada de `/mnt/yacy_1`.
Los JSON de procedencia acreditan entradas, meses, leyenda y normalización.
No reutilizar la variante 1819 anterior a SCL como equivalente a la acreditada.

El split proviene de [configuración](../../config/split_extension_este.json).
El entrenador vigente carga solo TRAIN/VAL en modo multicampaña; los dos tiles
antiguos 20HLK/20HLJ pueden existir en disco sin participar en TEST vigente.
2017/2018 no se agrega automáticamente. La preparación de 17 tiles por campaña
no significa que el entrenador requiera arrays del TEST nuevo.

## Etiquetas y calidad

Esquema: 0=NoData, 1=Fondo, 2=Maíz, 3=Soja, 4=Maní. Consultar
[leyendas](../referencias/leyenda_mnc.md) y el remapeo por campaña en el código.
En 2018/2019 el código crudo 0 es Maíz; en 2019/2020 se excluye como NoData.
La excepción explícita `--ignorar-codigo20` excluye únicamente el código 20
no documentado de 2020/2021. No extenderla a otras campañas o códigos.

2018/2019 y 2020/2021 usan composites SCL. El procesamiento original de
2019/2020 difiere; sus meses diciembre–abril y mayo también difieren en dtype
y descripciones de bandas. Mantener ese contexto al comparar resultados.
Las grillas completas pueden incluir provincias vecinas: no existe un recorte
provincial automático. La [auditoría histórica](../historico/geografia_20261009.md)
explica el alcance y los límites de su verificación por muestras.

## Verificar antes de preparar o entrenar

Desde la raíz del repo:

```bash
~/s4a-venv/bin/python src/preparar_multicampana.py --solo-verificar --ignorar-codigo20
~/s4a-venv/bin/python -m src.entrenamiento.entrenar_cordoba --verificar-datos \
  --campana 2018_2019 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_1819_scl_f16 \
  --campana 2019_2020 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16 \
  --campana 2020_2021 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_2021_f16
```

`--solo-verificar` comprueba entradas y grillas; comprobar por separado los JSON
cuando cambien rutas resueltas o procedencia. `--verificar-datos` valida arrays
sin entrenar. Para regenerar datos, revisar primero `src/preparar_multicampana.sh`:
reutiliza pares solo con procedencia compatible y aborta ante incompatibilidades.
Consultar auditoría y códigos de salida bajo `logs_multicampana/`; no interpretar
un log antiguo como estado actual. No regenerar lo ya acreditado para probar documentación.

## Entrenar y evaluar

`scripts/entrenamiento/entrenar_multicampana_unet.sh DIRECTORIO_EXPERIMENTO` ejecuta U-Net con
las tres campañas. Crear previamente un directorio nuevo en `/mnt/yacy_*`,
comprobar recursos/procesos y lanzar con log externo según CONTRIBUTING.
El wrapper requiere ese directorio existente para escribir PID y estado.

`--campana` es repetible; `--dir-exp` selecciona la salida.
Los pesos de clase se calculan con TRAIN y el mejor checkpoint se elige con VAL.
`--reanudar` usa un checkpoint completo; `--init-pesos` comienza una nueva
optimización desde pesos, y no equivale a reanudar. Revisar `--help` para opciones
de lectura por bloques y caches; un manifiesto de cache depende de sus rutas reales.
No mover ni eliminar caches acreditadas sin revisar consumidores y procedencia.

El cierre evalúa [TEST vigente](test_actual.md). `--test-historico` reproduce el
flujo anterior explícitamente. Las métricas de parches y mapas completos no son
intercambiables. Los detalles de las corridas de octubre quedan en el
[registro histórico](../historico/experimentos_20261006_10.md).

El entrenador está en src/entrenamiento/entrenar_cordoba.py. Las entradas
anteriores Python y shell de src conservan compatibilidad.

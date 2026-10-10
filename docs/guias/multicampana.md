# Preparación y entrenamiento multicampaña

Referencia: `floryacy:~/cordoba_ia_unne`. Entorno: `~/s4a-venv/bin/python`.
Almacenamiento y lanzamientos: [CONTRIBUTING](../../CONTRIBUTING.md).

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
~/s4a-venv/bin/python src/1_train_cordoba.py --verificar-datos \
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

`src/entrenar_multicampana_unet.sh DIRECTORIO_EXPERIMENTO` ejecuta U-Net con
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

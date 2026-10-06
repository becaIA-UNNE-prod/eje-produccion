# Entrenamiento multicampana: Cordoba

Proyecto: `floryacy:~/cordoba_ia_unne`. Datos: `/mnt/yacy_3/prod_bis/ferreyra`.
Preparacion solicitada el 2026-10-06. No se inicia entrenamiento durante la preparacion.
Se excluye 2017/2018 (otra capa de etiquetas).

## Campanas y datos

| Campana | Composites bajo la raiz de datos | Dataset bajo `dataset/` | Leyenda |
|---|---|---|---|
| 2018/2019 | `composites_2018_2019_scl` | `train_cordoba_dicmay_1819_scl_f16` | 1819 |
| 2019/2020 | `composites_dicmay_2019_2020` | `train_cordoba_dicmay_f16` (existente) | 2020 |
| 2020/2021 | `composites_2020_2021_scl` | `train_cordoba_dicmay_2021_f16` | 2021 |

Cada campana tiene 17 tiles y seis meses diciembre-mayo: 102 composites. Se usan
B02, B03, B04 y B08 en ese orden por mes: 24 canales. Parches 256x256, paso 256,
X float16 normalizado por 10000; Y uint8 ya remapeado. Carga por mmap.

2019/2020 conserva el procesamiento 02b/02c vigente: dic-abr son 85 composites
float32 sin nombres de bandas embebidos; mayo son 17 uint16 con nombres de bandas. Su orden se acredita por esos scripts. 2018/2019 y
2020/2021 usan composites SCL y descripciones explicitas. No se homogeneiza ahora
el filtro de nubes de 2019/2020; considerar esa diferencia al interpretar resultados.

El dataset previo `train_cordoba_dicmay_1819_f16` fue escrito antes de finalizar
los composites SCL vigentes y carece de procedencia suficiente. Se conserva y se
genera una variante SCL en otro directorio. Los nuevos pares incluyen archivos
`dataset_<TILE>_procedencia.json` con entradas, fechas de modificacion, meses,
leyenda y normalizacion. La preparacion solo reutiliza pares completos con
procedencia coincidente y realiza auditoria de contenido al finalizar.

## Leyendas verificadas

Esquema comun: 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani.

- 1819: 0/3 -> Maiz; 1/2 -> Soja; 4 -> Mani; 5,6,7,8,9,11,12,13,14,15 -> Fondo;
  16 -> NoData. Los dobles cultivos se asignan por cultivo de verano.
- 2020: 10 -> Maiz, 11 -> Soja, 16 -> Mani; 12,13,14,15,17,18,19,21,22 -> Fondo;
  255 -> NoData.
- 2021: mismos codigos objetivo y fondo que 2020, **mas 26 (Papa) y 28 (Verdeo de
  Sorgo) -> Fondo**. 255 -> NoData. Codigos no documentados abortan la preparacion.

Registro oficial: https://zenodo.org/records/8286318. El checksum MD5 local coincide
con el publicado: `9b650b09b405e89e752fa513b610da25`.

Fuente 2021: `/mnt/yacy_1/prod/ferreyra/MNC_INTA/2020_2021/verano-2021.tif` e
`informe_mnc_2020_2021.pdf` en el mismo directorio, Tabla 2, pagina impresa 7.
El GeoTIFF es uint8, EPSG:4326, 70641x43994, NoData=255. La verificacion recorre el
raster completo por franjas antes de preparar mascaras; deja conteos en el log.
Se detectaron 81 pixeles nacionales con codigo 20, ausente de la Tabla 2; 65
intersectan los bounds de 20HLG. No se le asigna un significado sin evidencia:
el 2026-10-06 el usuario eligio excluir exclusivamente el codigo 20 como NoData=0.
No se le asigna una clase de cultivo o Fondo.
La opcion `--ignorar-codigo20` permite excluir exclusivamente ese codigo como
NoData=0, dejando la excepcion en logs y procedencia. No habilita otros codigos
desconocidos. La preparacion de esta corrida usa esa opcion por la decision expresa anterior.

Las mascaras preservan IDVER crudo, vecino mas cercano y grilla exacta de composites,
con nombre `mascaras_procesadas/etiqueta_mnc2021_<TILE>_10m.tif`.

## Split y entrenador

Mismos tiles reservados en todos los anios, sin 2017 agregado automaticamente:

- TRAIN: 19HGC, 20HLG, 20HLH, 20HMJ, 20HMK, 20HNK, 20JML, 20HKH, 20HMG,
  20HNG, 20HPG, 20HPH, 20JNL.
- VAL: 20HKJ, 20HMH.
- TEST: 20HLK, 20HLJ.

`--campana AAAA_AAAA DIRECTORIO` es repetible. Concatena TRAIN/VAL/TEST de cada
campana y exige los 17 pares por anio. Rechaza nombres/rutas duplicados y mezcla
con `--dir-train`, `--dir-train-extra` o `--sin-2017`. Los argumentos anteriores
siguen funcionando fuera del modo multicampana. `--dir-exp` sigue disponible;
el default multicampana es un directorio nuevo que incluye anios y modelo.

Antes de crear salidas/modelos se validan X/Y, dimensiones, dtypes, clases y
valores finitos. `--verificar-datos` termina ahi sin entrenar ni crear un experimento.
Pesos automaticos: raiz de frecuencia inversa sobre TRAIN de todos los anios,
NoData=0; `--pesos` permite override. `0b_calcular_pesos.py --dirs ...` usa la misma
funcion. No se usan etiquetas VAL/TEST para estimar pesos.

Checkpoint elegido por perdida sobre VAL conjunto; al finalizar se reporta TEST
conjunto y VAL/TEST por campana, con confusion, accuracy, precision, recall, F1 e IoU.
Las predicciones NoData sobre referencias validas cuentan como errores. Se guardan
`config_multicampana.json` y `metricas_multicampana.json` junto a los checkpoints.
Se conserva loss CE+Dice, seed 42, lr=1e-4, weight_decay=1e-3, maximo 100 epocas y
patience 30; el entrenamiento no se ejecuta en esta preparacion.

## Preparacion en segundo plano

Entorno operativo: `$HOME/s4a-venv/bin/python` (NumPy 1.26.4, Rasterio 1.4.3,
PyTorch 2.3.1+cu118). El pyproject del repo no refleja este entorno; no modificarlo
ni instalar dependencias para esta tarea. Servidor: RTX 3090 24 GiB, RAM 124 GiB.
Al inspeccionar habia 541 GB libres en yacy_3; la preparacion exige 210 GiB libres.

Primero verificar codigo y datos (sin entrenamiento):

```bash
cd ~/cordoba_ia_unne
~/s4a-venv/bin/python -m unittest discover -s test -p test_multicampana.py -v
~/s4a-venv/bin/python src/preparar_multicampana.py --solo-verificar --ignorar-codigo20
```

Luego lanzar solo preparacion (4 workers, tiles secuenciales):

```bash
mkdir -p /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana
nohup bash src/preparar_multicampana.sh --ignorar-codigo20 > /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/preparacion.log 2>&1 < /dev/null &
```

Secuencia: verificar labels y composites, generar/verificar 17 mascaras 2021,
generar dataset 2021, generar variante SCL 1819, auditar los tres datasets y calcular
pesos TRAIN. Ninguna etapa llama al entrenador o a inferencia. Tiene bloqueo flock
para impedir dos preparaciones simultaneas y escritura temporal para salidas.

Consultar `logs_multicampana/preparacion.pid`, `preparacion.log` y
`preparacion.exitcode` (solo se escribe al salir, 0=exito). Al terminar se generan
`auditoria.json` con estado completo y `campanas.json`. Un relanzamiento valida
procedencia antes de reutilizar pares. Si hay un par incompleto o incompatible,
aborta y requiere revisar el tile; no borra salidas automaticamente.
El log del ultimo arranque reemplaza al anterior; conservarlo antes de relanzar si
se necesita el historial. Verificar tambien que los timestamps de exitcode y
auditoria correspondan al arranque actual.

Comando para verificar luego las tres campanas, sin entrenar:

```bash
~/s4a-venv/bin/python src/1_train_cordoba.py --verificar-datos \
  --campana 2018_2019 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_1819_scl_f16 \
  --campana 2019_2020 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16 \
  --campana 2020_2021 /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_2021_f16
```

Para una futura corrida, quitar `--verificar-datos` y especificar modelo y salida.
No se ha autorizado su lanzamiento en esta etapa.

Los cambios locales previos se preservan. El conflicto de Git preexistente en
`nohup.out` no se modifica. No se hace commit, merge ni reset.


## Estado de la corrida lanzada

Lanzada el 2026-10-06 a las 10:34 ART aproximadamente, mediante nohup y &.
PID del supervisor: **15570**; proceso Python observado: **15574**.
Se comprobo que el supervisor quedo con PPID=1 tras cerrar SSH. El precheck
productivo paso: raster MNC completo y 306 composites. Ya se escribio la primera
mascara 2021 y la preparacion continua en segundo plano. No se lanzo entrenamiento.

Validacion de codigo: 12 pruebas CPU/sinteticas aprobadas; comprobacion de diff
sin errores de espacios. La excepcion 20 -> NoData fue elegida por el usuario y
queda registrada en `logs_multicampana/labels_2021.json` y en las procedencias.

Este estado corresponde al lanzamiento, no implica datasets finalizados. El
estado actualizado se consulta en `preparacion.log`, `preparacion.exitcode` y
`auditoria.json` bajo `logs_multicampana/`. Backup de archivos anteriores en
`/tmp/cordoba-antes-multicampana-20261006T102130` (temporal del servidor).


## Recuperacion del fallo de lectura (2026-10-06)

La primera preparacion salio con codigo 1 a las 10:50 ART. Completo 12 tiles de
2020/2021 y fallo en 20JNL: `20JNL_202104_median.tif` contiene un bloque DEFLATE
ilegible (banda 1, offsets de bloque X=19/Y=13, error ZIPDecode/incorrect data
check). Se reprodujo la lectura fallida; el header y la grilla eran correctos.
No fue un error de remapeo de clases. La causa fisica de la corrupcion no esta
establecida; el mensaje de GDAL solo demuestra que el contenido no se decodifica.

La seleccion original de ese mes usa `20210429`. La recuperacion verifica todas
las bandas originales y SCL, recompone con el mismo algoritmo/seleccion, lee todos
los bloques de la salida y la publica solo si pasa. Conserva el TIFF defectuoso y
los logs previos en `logs_multicampana/recuperacion_20261006T111112/`.

Se incorporo `utils/integridad_raster.py`: lectura completa de todas las bandas
por bloques. La auditoria recorre los 306 composites antes de generar datasets.
`integridad_composites.json` guarda verificaciones y firmas de archivo (ruta real,
inode, tamano, mtime y ctime); los archivos cambiados requieren una nueva lectura.
Si otro composite SCL falla, se reconstruye desde sus fuentes; si falla una fuente
o un composite legacy, se detiene con diagnostico, sin inventar valores ni saltar
pixeles. Los tiles ya generados que dependan de un composite reemplazado se mueven
al respaldo para regenerarlos; los pares compatibles se conservan.

13 pruebas aprobadas, incluida una regresion con TIFF truncado de header legible.
No se inicio entrenamiento. Recuperacion lanzada con nohup a las 11:11 ART,
PID **16560**, script `src/recuperar_multicampana.py`. Sigue en segundo plano:

```bash
tail -f /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/recuperacion.log
```

`recuperacion.json` registra las reparaciones y el estado; `recuperacion.exitcode`
se escribe al salir. Tras validar las entradas, reanuda automaticamente
`preparar_multicampana.sh --ignorar-codigo20` y su log `preparacion.log`.
El wrapper limpia el exitcode anterior al arrancar para no confundir el fallo
viejo con la corrida nueva. No se declara completado hasta que auditoria y ambos
exitcodes confirmen el final. No ejecutar otra preparacion en paralelo.


## Preparacion finalizada y verificada

Recuperacion y preparacion terminaron con codigo 0. Finalizacion UTC: `2026-10-06T16:32:17Z`.
La recuperacion reanudo automaticamente los datasets; no se requiere otro nohup.
Se verificaron 51 pares X/Y y las firmas de los 306 composites al reconectar el 2026-10-06.

- 2018_2019: 17 pares X/Y; parches {'train': 22121, 'val': 3470, 'test': 3275}.
- 2019_2020: 17 pares X/Y; parches {'train': 21934, 'val': 3470, 'test': 3273}.
- 2020_2021: 17 pares X/Y; parches {'train': 22676, 'val': 3528, 'test': 3526}.

Pesos conjuntos TRAIN [NoData, Fondo, Maiz, Soja, Mani]: `[0.0, 0.765, 1.057, 0.855, 6.139]`.
Estado definitivo en `logs_multicampana/auditoria.json`, `campanas.json` y `recuperacion.json`.
No se inicio entrenamiento. Los datos estan listos para una futura corrida multicampana.


## Entrenamiento U-Net iniciado: 20261006T163403

El usuario autorizo solo U-Net primero. Lanzado con nohup y & el 2026-10-06 a las 16:34 ART.
Supervisor PID 18334; se valida la independencia de SSH (PPID=1).
Experimento: `/mnt/yacy_3/prod_bis/ferreyra/dataset/exp_multicampana_1819_1920_2021_unet_20261006T163403`. Log: `train.log` en ese directorio.
Se conservan las tres campanas, split espacial fijo, pesos automaticos TRAIN, seed 42, batch 32, maximo 100 epocas y patience 30.
No se lanzo FCRTransUNet. Wrapper: `src/entrenar_multicampana_unet.sh`; bloqueo de entrenamiento multicampana, PID e inicio/fin/exitcode por experimento.
Estimacion basada en la corrida anterior U-Net (3h30 para 58 epocas, 21934 parches TRAIN): la nueva tiene 66731 parches. Extrapolacion 11-18h; rango operativo 12-24h y margen hasta 30h por I/O al superar la RAM. No es una garantia.
El estado actualizado se consulta en el log; la finalizacion se registra en `train.exitcode` y `fin.txt`. Checkpoints y metricas quedan en el mismo experimento.

> Registro histórico consolidado el 10/10/2026. Los estados, PID, comandos y splits describen su fecha original; no son instrucciones vigentes. Leer las guías en `../guias/`.

# Registro de experimentos, 6–10/10/2026

## Multicampaña, 6–9 de octubre

> TEST vigente desde 10/10/2026: 20HNJ y 20HNH. El entrenador y la inferencia cargan config/split_extension_este.json; la evaluacion principal excluye TRAIN/VAL. Las referencias anteriores a 20HLK/20HLJ corresponden al split historico. Ver README_TEST_ACTUAL.md.

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


## Lectura por bloques (2026-10-07)

La primera corrida paso 16 h 35 min para completar 10 epocas: TRAIN tiene
195,5 GiB de X, excede la RAM de 124 GiB y shuffle global provoca lectura
aleatoria del HDD sdb. Se midieron 46,3 MiB/s con 91,7% de ocupacion y lectores
bloqueados en filemap_fault/blk_mq_get_tag; GPU esperando datos.

`--lectura-bloques --bloque-parches 512 --bloques-pool 16 --workers 4 --prefetch 2`
precarga secuencialmente grupos de bloques (aprox. 24 GiB de X mas Y en page
cache), mezcla sus indices y visita todos los parches una vez por epoca. El
orden de bloques y de parches cambia con la epoca. La mezcla ya no es una
permutacion global uniforme: las muestras se mezclan dentro de cada pool;
se conservan etiquetas, augmentacion y split. No necesita duplicar datasets.
VAL/TEST usan hasta dos workers sin persistencia. El log informa precarga por
pool y tiempos de TRAIN/espera/epoca; espera_loader_s mide espera del consumidor,
no tiempo CUDA sincronizado. Medir una epoca completa antes de extrapolar.

`--init-pesos RUTA/best_model.pth` inicia desde pesos previos con optimizador,
scheduler y contador de epocas nuevos. La corrida original solo guardaba el
mejor modelo (epoca 4), no un checkpoint de la epoca 10. Se conserva intacta.
Ahora `ultimo_checkpoint.pth` se actualiza atomicamente cada epoca con modelo,
optimizador, scheduler, scaler, historial y RNG del proceso principal; tambien
se escribe historial.npy cada epoca. `--reanudar RUTA/ultimo_checkpoint.pth`
recupera ese estado en el mismo directorio de experimento. El RNG de los
workers de augmentacion no se conserva; no promete equivalencia bit a bit.
El lanzador acepta argumentos adicionales despues del directorio de salida.

Relanzamiento optimizado: `/mnt/yacy_3/prod_bis/ferreyra/dataset/exp_multicampana_1819_1920_2021_unet_bloques_20261007T094404`.
Supervisor PID 31879. Log estable: `/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/ultimo_entrenamiento_unet.log`.


## Precarga anticipada y caches en tres discos (2026-10-07, tarde)

Las epocas 9-11 tardaron 47,5-51,4 minutos. Epoca 11: TRAIN 2686 s,
espera del loader 1879 s, total 3031 s. Cada pool se precargaba de forma
sincrona, dejando la GPU esperando 2-3 minutos entre pools. Ademas, a las
18:48 otro usuario (urturi, PID 9563) ocupaba 13,5 GiB de GPU y compartia
el computo; nvidia-smi pmon mostraba reparto entre ambos procesos.

Correccion de procedencia: el primer relanzamiento registraba --init-pesos
pero no ejecutaba load_state_dict para esa opcion. Esa corrida empezo desde
cero. Se corrigio la carga y se agrego una prueba que verifica los pesos
cargados desde el archivo. El siguiente relanzamiento conserva el estado de
la corrida vigente usando --reanudar, incluida su epoca y optimizador.

`--anticipar-pool` precarga el siguiente pool durante el computo actual.
Mantiene como maximo dos pools en page cache (~48 GiB de X mas Y); un lector
por dispositivo realiza lecturas consecutivas y hasta tres dispositivos
trabajan en paralelo. No cambia la permutacion respecto del sampler anterior.

El coordinador `optimizar_lectura_multicampana.sh EXP PID_TRAIN`, lanzado
con nohup y &, prepara solo copias TRAIN:
- 2018_2019: /var/tmp/ferreyra_cordoba_ia_unne/train_cache (NVMe).
- 2019_2020: /mnt/yacy_1/prod/ferreyra/cache_multicampana_train (HDD sda).
- 2020_2021: originales en yacy_3 (HDD sdb).
Las copias conservan los bytes y se verifican con SHA-256. Hay reserva minima
de 40 GiB en NVMe y 100 GiB en yacy_1. VAL/TEST siguen en sus originales.
El manifiesto cache_lectura.json registra firmas de originales/destinos y
se rechaza si cambian; tambien la validacion completa de TRAIN usa las copias.
La copia tiene prioridad de I/O baja y puede reducir temporalmente el ritmo
de entrenamiento. Los originales se conservan. /var/tmp debe conservarse
mientras se use el manifiesto; si se elimina una cache, hay que reconstruirla.

Una vez listas, se espera un nuevo checkpoint completo (max. 3 h), se
interrumpe la corrida y se relanza en el mismo EXP con --reanudar. Se conserva
checkpoint_antes_precarga_paralela.pth y train_antes_paralelo_epochN.log;
train.log se mantiene y se agrega el nuevo proceso al final. Si ya termino
por early stopping o maximo de epocas, no se reinicia automaticamente.
Estado en optimizacion_lectura.json, logs en optimizacion_lectura.log y codigo
final del coordinador en optimizacion_lectura.exitcode.

No se promete 100% continuo: persisten la primera precarga y VAL, y la GPU
esta compartida. Medir tiempos por epoca y pmon por PID luego del cambio.

A pedido del usuario (19:06), la corrida lenta se detuvo durante epoca 12.
Se recuperara el ultimo checkpoint completo (epoca 11), sin esperar otra
epoca. El coordinador usa --sin-esperar-epoca y --usar-copia-pid para
conservar la copia que ya estaba en marcha; solo se lanzara la corrida nueva.


## Inferencia y comparaciones (2026-10-09)

Entrenamiento terminado 2026-10-09 03:46:19 ART, exitcode 0. Early stopping
en epoca 90; mejor checkpoint epoca 60 (VAL loss 0.3744356).
TEST de parches: accuracy 0.8829207, F1 macro 0.8070624, mIoU 0.6922312.

`bash src/inferencia_multicampana.sh EXP SALIDA` se lanza con nohup y &.
Usa best_model.pth y crea 51 mapas U-Net multicampana (17 tiles x 3 campanas),
mas 12 mapas TEST de U-Net y FCRTransUNet previos entrenados en 2019/2020
dic-may. Los 18 mapas TEST se hacen primero; TRAIN/VAL se completan despues.
Se verifica que los tres checkpoints carguen con 24 canales y 5 clases,
y que los 306 composites y las 51 mascaras coincidan en grilla/bandas.

Inferencia: TIFF abiertos una vez por tile, bloques 512 y batch 4, AMP en
GPU, normalizacion /10000 redondeada a float16 como los datasets, padding
edge de bordes y NoData para pixeles sin imagen en todos los meses. Los TIFF
se escriben temporales y se publican atomicamente al completar el tile.
Tienen tags de checkpoint SHA-256 y firma de insumos; no se mezclan modelos.

Se generan curvas, matrices de confusion por campana/rol, PNG RGB-MNC-mapas
de modelos-errores (vista reducida a 1024 px), barras de accuracy/F1 y F1 por
clase, CSV y JSON a resolucion completa. MNC>0 define la mascara comun y
pred=0 cuenta como error. MNC2021 codigo 20 sigue ignorado como NoData.

La comparacion justa usa TEST 20HLK/20HLJ con mismos composites, ventana,
leyenda y pixeles de evaluacion. Multicampana vio TRAIN de todos los anios;
no es evaluacion de anio no visto. TRAIN solo es descriptivo. VAL de tres
anios y VAL historica 2019/2020 son conjuntos distintos, y sus pesos por
clase pueden diferir: curvas se muestran con contexto, no como ranking.
No se comparan directamente metricas de parches 256 con mapas completos 512.
Se omite la variante con 2017/2018 para evitar otra leyenda y posible
solapamiento espacial historico con los tiles TEST actuales.

Estado por tarea en progreso.json, log inferencia.log, PID inferencia.pid y
resultado final inferencia.exitcode. Figuras en figuras/; mapas en
predicciones/MODELO/CAMPANA/. Las cuatro pruebas verifican bordes/georreferencia,
normalizacion, rechazo de grillas incompatibles, NoData y excepcion codigo 20.

Salida de esta corrida: `/mnt/yacy_3/prod_bis/ferreyra/dataset/exp_multicampana_1819_1920_2021_unet_bloques_20261007T094404/inferencia_20261009T093206`.


## Evaluacion temporal en anio nunca visto: 2021/2022

El usuario eligio los cuatro tiles espaciales VAL/TEST (20HKJ,20HMH,20HLK,20HLJ).
Se congela best_model.pth de epoca 60 con SHA-256 en protocolo.json; 2021/2022
no aparece en config_multicampana.json. No se reentrena, calibra ni selecciona
un checkpoint usando etiquetas nuevas. Ventana diciembre 2021 - mayo 2022,
bandas B02/B03/B04/B08 y mismo procesamiento SCL (incluye correccion BOA).

Fuente: https://zenodo.org/records/8284081, MNC_verano2022.tif,
MD5 f88ffac06865b36038916ec22c1ffe28. Informe MD5
9111c2166f1c074430e435aa90cb6c59. Tabla 2, pagina impresa 6 (PDF 7),
verificada visualmente: 10 Maiz,11 Soja,16 Mani; resto de codigos documentados
12,13,14,15,17,18,19,21,22,26,28 a Fondo. 255 es NoData. Codigos desconocidos
aborton; la excepcion de codigo 20 en 2021 no se extiende a 2022.

El pipeline descarga STAC/SCL, selecciona escenas, descarga bandas, compone
y verifica integridad de TIFF, proyecta mascaras con nearest y verifica los
codigos dentro de los cuatro tiles antes de inferir. Espera las fuentes
oficiales verificadas (transferidas desde la maquina local por fallo TLS de
Zenodo en floryacy); no se desactiva la verificacion TLS.

La inferencia espera el lock compartido de la corrida anterior y compara
U-Net multicampana con U-Net/FCRTransUNet 2019/2020 sobre los mismos pixeles.
Resultados de TEST espacial y VAL espacial se separan dentro del anio nuevo.
MNC>0 define la mascara, pred=0 cuenta como error; es concordancia con mapa
INTA, no exactitud contra muestras de campo independientes. Se generan mapas,
comparaciones visuales, matrices y evaluacion_temporal.json.

Salida: `/mnt/yacy_3/prod_bis/ferreyra/evaluacion_ano_no_visto_2021_2022_20261009T101637`.


---

## Descarga de la extensión este, 9 de octubre

# Descarga de la extensión este

Iniciada el 9 de octubre de 2026, a las 15:26 (Argentina), en floryacy.

Se incorporan los tiles **20HNJ y 20HNH**, entre 20HNK y 20HNG, para cubrir la franja faltante del este de Córdoba y sur de Santa Fe. Su existencia y ubicación se comprobaron en el catálogo Sentinel-2 Earth Search.

Campañas: **2018/2019, 2019/2020, 2020/2021 y 2021/2022**, todas de diciembre a mayo. Son 48 combinaciones tile-mes.

Los tiles antiguos encontrados anteriormente se descartan como candidatos a este cambio. No se borraron archivos históricos. El split actual y los modelos existentes no se modificaron: primero se descargan los datos nuevos.

## Ubicación y ejecución

- Datos: `/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/sentinel2_AAAA_AAAA`.
- Logs: `/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/logs_20261009T152638`.
- Metadatos: `/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/ultima_descarga.json`.
- Supervisor inicial: PID **17785**.
- Script remoto: `~/cordoba_ia_unne/src/descargar_extension_este.sh`, preparado con `git add`; sin commit.

Se inició mediante nohup, entrada `/dev/null`, salida a archivo y sesión separada. Continúa aunque se cierre la terminal o SSH. El comando equivalente desde el proyecto es:

```bash
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash src/descargar_extension_este.sh /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/logs_20261009T152638 > /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/logs_20261009T152638/descarga.log 2>&1 < /dev/null &
```

No volver a ejecutar ese comando mientras siga activa la descarga. El script tiene un lock que evita duplicarla.

## Flujo

Para cada campaña, secuencialmente:

1. Buscar escenas Sentinel-2 L2A de ambos tiles en diciembre–mayo.
2. Descargar SCL de las escenas candidatas y verificar apertura y una ventana de lectura.
3. Seleccionar escenas por cobertura despejada, con el criterio SCL ya usado en el proyecto.
4. Comprobar que estén presentes los 12 tile-mes de esa campaña.
5. Descargar B02, B03, B04 y B08 de las escenas seleccionadas, con seis descargas simultáneas y reintentos por archivo.
6. Verificar apertura y ventana 64×64 de cada archivo seleccionado; guardar `descarga_verificada.json`.

Esta verificación no es una auditoría exhaustiva de cada bloque TIFF. Se deja una reserva mínima de 100 GiB al iniciar cada campaña. Se utiliza yacy_1, con aproximadamente 1,2 TB libres al iniciar, para evitar agotar yacy_3.

Si una campaña falla, queda registrado y se intentan las restantes. El supervisor termina con código distinto de cero si alguna falló. Una nueva ejecución reutiliza archivos descargados y las selecciones guardadas.

No se generan aún máscaras, datasets ni inferencias; tampoco se cambia TEST. Eso corresponde al siguiente paso, después de verificar la descarga.

## Cómo seguir el avance

Desde esta computadora:

```bash
ssh floryacy 'tail -f /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/logs_20261009T152638/2018_2019.log'
```

Los otros logs son `2019_2020.log`, `2020_2021.log` y `2021_2022.log`. `descarga.log` contiene el resumen de campañas. Cada campaña tiene un `.exitcode`; el cierre global queda en `descarga.exitcode` y `fin.txt`. Código 0 significa éxito.

## Cierre verificado

La descarga terminó correctamente el **09/10/2026 a las 17:44:23** (Argentina), con código global 0 y código 0 en las cuatro campañas. No quedaron procesos activos de descarga en la revisión posterior.

| Campaña | Escenas seleccionadas | Espacio incluyendo SCL candidatas |
|---|---:|---:|
| 2018/2019 | 43 | 28.2 GiB |
| 2019/2020 | 27 | 17.0 GiB |
| 2020/2021 | 29 | 18.4 GiB |
| 2021/2022 | 31 | 19.2 GiB |

Están las 48 combinaciones tile-mes; no hay archivos seleccionados faltantes ni descargas `.part` pendientes. La verificación realizada es apertura y ventana 64×64, no lectura exhaustiva de todos los bloques.

Pendientes para usar estos tiles como TEST: componer las 48 imágenes mensuales, auditar íntegramente los compuestos, generar/verificar las máscaras INTA de cada campaña y preparar la evaluación espacial con un split documentado. Los tiles 20HNJ y 20HNH todavía no están asignados a TRAIN, VAL ni TEST. No se inició preparación adicional ni se cambió el split durante esta revisión.


---

## Preparación y recuperación de la extensión este

# Nuevo TEST: extensión este

Preparación y evaluación iniciadas con `nohup ... &` el 09/10/2026 a las 19:22 (Argentina), en floryacy. Supervisor inicial PID 19987; continúa al cerrar terminal o SSH.

## Alcance

- Tiles nuevos: **20HNJ y 20HNH**.
- Cuatro campañas: **2018/2019, 2019/2020, 2020/2021 y 2021/2022**.
- Ventana fija: diciembre a mayo, cuatro bandas B02/B03/B04/B08, 24 canales, 10 m.
- U-Net multicampaña congelada, mejor checkpoint de VAL (época 60). Se copia el checkpoint al directorio del experimento y se verifica su SHA256 antes de inferir.
- Se guarda una configuración separada `config/split_extension_este.json`: TRAIN y VAL originales; TEST nuevo con ambos tiles; TEST anterior fuera de ese split. No se modifica la configuración de la corrida anterior ni sus métricas.

## Etapas en ejecución

1. Leer **todos los bloques y bandas** de los TIFF seleccionados descargados. Si hay corrupción, detener y reportar el archivo.
2. Generar **48 compuestos mensuales SCL**, dos workers. Directorios `composites_AAAA_AAAA_scl` dentro de la extensión este.
3. Auditar íntegramente esos compuestos y comprobar que cada grupo mensual comparte grilla.
4. Reproyectar por vecino más cercano el mapa INTA de cada campaña y verificar estrictamente los códigos. Código 20 se ignora exclusivamente para 2020/2021, según la excepción autorizada previamente. No se aplica esa excepción a 2021/2022.
5. Guardar **16 máscaras de clases** (dos criterios × dos tiles × cuatro campañas) y leer todos sus bloques.
6. Esperar el lock de inferencia GPU; ejecutar **ocho mapas** con la U-Net congelada, sin reentrenar ni ajustar con este TEST.
7. Guardar figuras de comparación, errores y exclusión, matrices de confusión, métricas por tile y agregadas por campaña, y resumen CSV.

No se generan arrays X/Y de parches: para la evaluación de mapas completos no hacen falta y ocuparían espacio adicional.

## Tratamiento del solapamiento espacial

Se publican dos resultados sobre las mismas predicciones:

- **Completa:** todos los píxeles con referencia INTA remapeada >0 del tile.
- **Independiente:** excluye píxeles que intersectan las grillas completas de cualquier tile TRAIN o VAL de las tres campañas originales. La rasterización usa `all_touched=True` para excluir también los píxeles tocados por el borde.

La exclusión es conservadora: puede excluir píxeles de grillas TRAIN/VAL que no llegaron a aportar un parche. Evita puntuar áreas ya incluidas geográficamente en los conjuntos anteriores. NoData se ignora como referencia; predecir 0 en una referencia válida cuenta como error. El contexto de una inferencia convolucional puede cruzar el borde de la máscara; no se afirma aislamiento de todo el contexto visual.

2021/2022 constituye una prueba temporal y espacial; las otras tres campañas son zonas nuevas en años ya usados para entrenar. La referencia es un mapa clasificado de INTA, no observaciones independientes de campo. Los nuevos compuestos 2019/2020 usan SCL, a diferencia del preprocesamiento original de esa campaña: comparar con las métricas antiguas mezcla diferencias de territorio y preparación.

## Rutas y logs

Base remota:
`/mnt/yacy_1/prod/ferreyra/extension_este_multicampana`

Resultados de esta ejecución:
`/mnt/yacy_1/prod/ferreyra/extension_este_multicampana/evaluacion_20261009T192204`

Desde esta computadora:

```bash
ssh floryacy 'tail -f /mnt/yacy_1/prod/ferreyra/extension_este_multicampana/ultima_preparacion.log'
```

Metadatos del supervisor: `ultima_preparacion.json`. El resultado final se registra en `preparacion.exitcode` (0 = éxito) y `fin.txt`. Durante inferencia, `progreso.json` indica campaña, tile y tarea. Antes de inferencia, el avance se sigue en el log.

## Verificación del código

Pasaron tres pruebas nuevas: excepción del código 20 exclusiva de 2021, NoData y predicción 0 como error, y exclusión parcial por solapamiento. También pasaron las cuatro pruebas existentes del postprocesado. Se verificaron sintaxis de Python y Bash, compatibilidad real del checkpoint con `SimpleUNet(24,5)`, y coincidencia del split original guardado con el usado para excluir áreas.

Scripts remotos: `src/preparar_extension_este.sh`, `src/preparar_evaluar_extension.py`, `test/test_extension_este.py`. Cambios preparados con git add, sin commit.

## Recuperación tras error de etiquetas

La primera ejecución se detuvo después de auditar correctamente los 650 TIFF seleccionados y los 48 compuestos, al detectar el código 0 en MNC verano 2020. Se comprobó que proviene del raster oficial original: MD5 `e5a7b99560a9a9aa37712c69250a3541`, coincidente con Zenodo (registro 8286310). Afecta 183 píxeles del tile 20HNJ y 1.691 del 20HNH tras reproyección. El remapeo del dataset original ya excluía el código 0 como NoData; se corrige el validador de este nuevo flujo para conservar ese criterio exclusivamente en 2019/2020. En 2018/2019 el código 0 sigue siendo Maíz. En 2020/2021 y 2021/2022 no se admite silenciosamente código 0. Nueva prueba de regresión pasada.

También se detectaron advertencias de calidad: 20HNJ marzo de 2019 tenía 93,7 % despejado (resto relleno de menor azul), y 20HNH marzo de 2021 tenía 98,6 % con datos. No son errores de integridad. La recuperación amplía las escenas de esos dos meses con todas las candidatas ya catalogadas y SCL disponibles, descarga las bandas faltantes, audita completamente y recompone. Solo promueve el compuesto nuevo si no disminuye cobertura ni porcentaje despejado; preserva el anterior como respaldo.

La recuperación corre nuevamente con nohup y &. Reutiliza los compuestos y las auditorías vigentes, y conserva los archivos de salida del fallo con sufijo `.fallo_labels_FECHA`. Los mensajes antiguos del error permanecen en el log como evidencia; la nueva sección comienza con `RECUPERACION`.

## Cierre confirmado el 10/10/2026

La recuperación y evaluación finalizaron el **09/10/2026 a las 20:34:27** (Argentina), código de salida 0. Se completaron ocho inferencias, 48 compuestos auditados y 16 máscaras auditadas. Los resultados se copiaron a `resultados_nuevo_test_extension_este/`, con 17 figuras PNG, JSON de métricas y CSV. No quedaron procesos de esta preparación activos.

Las mejoras logradas fueron: 20HNH marzo de 2021 pasó de 98,56 % a 100 % de cobertura con datos; 20HNJ marzo de 2019 pasó de 93,71 % a 93,88 % despejado y mantuvo 100 % con datos. Este último aún contiene aproximadamente 6,12 % rellenado por el criterio de menor azul y se conserva como limitación de calidad; no es corrupción del TIFF.

La configuración del nuevo TEST permanece separada del entrenador original. No se inició un nuevo entrenamiento.

La ruta de copia anterior es histórica. Para localizar la preparación remota actual,
consultar `preparacion_base` en [la configuración vigente](../../config/split_extension_este.json).


---

## Integración y comparación del TEST, 10 de octubre

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

Copia local automática de figuras y métricas: [reporte del TEST vigente](../../artifacts/reports/test_actual/). Los TIFF y pesos quedan en floryacy por tamaño.

El cierre se registra en `inferencia.exitcode`; la actualización final de títulos y leyendas se registra en `figuras.exitcode`. Ambos deben valer 0 antes de la copia local. `progreso.json` muestra el mapa que se está procesando.

## Validación y Git

Pasaron tres pruebas nuevas de configuración, disjunción de particiones, rechazo de exclusiones obsoletas y ejecución real de `--verificar-datos` sin arrays de TEST; cuatro pruebas de etiquetas/exclusión, cuatro de inferencia y tres de cachés/checkpoints. Se acreditó la compatibilidad real de los tres checkpoints y se inspeccionó una figura generada.

Cambios preparados con git add, sin commit. Los resultados históricos y los pesos originales se conservan.

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

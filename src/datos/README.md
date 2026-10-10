# Leyendas por campaña

leyendas.py reúne remapeos, patrones de máscaras y validación de códigos.
No abre rasters ni importa comandos. La leyenda 2021/2022 se lee del JSON
versionado solo al pedirla. Se conservan las diferencias: crudo 0 es Maíz
en 2018/2019 y NoData en 2019/2020; la excepción 20 es exclusiva de 2020/2021.
utils.leyendas_mnc conserva los imports anteriores.
Los demás generadores históricos mantienen sus remapeos específicos hasta
una migración propia; no se reinterpretan como equivalentes automáticamente.

## Campañas y rutas

campanas.py consume config/campanas.json y las rutas operativas registradas
en config/almacenamiento.json mediante utils.rutas. Proporciona meses,
composites, fuentes MNC y las tuplas de preparación ya usadas.
Las ubicaciones existentes y el alias de máscaras se conservan.
El catálogo no define TRAIN/VAL/TEST: el split vigente sigue en
config/split_extension_este.json; preparación histórica mantiene 17 tiles.
Los wrappers shell y otros generadores aún tienen configuración propia
y se migrarán por separado.

## Conversión mmap

`convertir_datasets_a_mmap.py` convierte NPZ a pares X/Y con verificación
y temporales. Entrada: `python -m src.datos.convertir_datasets_a_mmap DIR`.
El script numerado anterior conserva compatibilidad; --borrar-original sigue
siendo explícito y se aplica tras verificar. Los pares existentes se omiten.

## Preparación de máscaras

Las implementaciones están en moldear_mascaras.py, mascaras_mnc_2018_2019.py,
mascaras_nivel3_2017_2018.py y mascaras_mnc_2020_2021.py. Las entradas numeradas
anteriores conservan compatibilidad. Las variantes por campaña se ejecutan con
`python -m src.datos.<modulo> --help`; importar no ejecuta la preparación.
Moldear conserva su ejecución fija original sobre JP2 y no tiene parser CLI.
Se mantienen nearest, grillas, códigos crudos, NoData y salidas por campaña.
Nivel3 sigue leyendo REMAP_NIVEL3 del generador 03c; su traslado queda pendiente.
No unificar estos contratos por parecido de implementación.
El procedimiento operativo está en [Multicampaña](../../docs/guias/multicampana.md).

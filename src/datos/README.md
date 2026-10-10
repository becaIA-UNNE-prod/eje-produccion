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
Nivel3 importa REMAP_NIVEL3 directamente de generar_dataset_cordoba_f16;
la tabla permanece única.
No unificar estos contratos por parecido de implementación.
El procedimiento operativo está en [Multicampaña](../../docs/guias/multicampana.md).

## Fuentes y composites

Entradas por módulo: descargar_stac.py (buscar/descargar/verificar),
composites_cog.py y composites_scl.py (seleccionar/componer), con crear_parser
y main(argv=None). Las rutas numeradas delegan y reexportan funciones.
composites_mensuales.py y composites_2019_2020.py conservan main() con sus
configuraciones fijas originales, sin inventar parámetros CLI.

| Módulo | Contrato conservado |
|---|---|
| composites_mensuales | JP2; mediana de las fechas por banda, dtype original |
| composites_2019_2020 | JP2; primera fecha con B04 válida al menos 80%; siete meses oct–abr; float32 |
| composites_cog | COG; orden por nubosidad, primer píxel válido y relleno; offset condicionado por baseline/flag; uint16 |
| composites_scl | COG/SCL; clases inválidas y dilatación, filtro azul, primera observación despejada y respaldo de menor azul; uint16 |

Los TIFF se siguen llamando _median.tif por compatibilidad; el nombre no define
el cálculo. No intercambiar variantes ni reconstruir entradas acreditadas.
Descarga conserva productos.json, bandas, selección, reintentos y códigos de
salida. Su verificación lee una ventana, no todos los bloques.
Consultar el orden de ejecución en [Multicampaña](../../docs/guias/multicampana.md).

## Generadores de arrays

| Implementación | Contrato conservado |
|---|---|
| generar_arrays_originales.py | Entrada 03; pares dataset_T<TILE>_X/Y.npy; remapeo original 0–8; calendario fijo original o meses explícitos por función |
| generar_dataset_cordoba_f16.py | Entrada 03c; pares dataset_<TILE>_X/Y.npy, procedencia JSON, leyendas por campaña, modo estricto y política de código 20 |
| generar_dataset_cordoba_2017_f16.py | Entrada 03d; pares dataset_<TILE>_2017_X/Y.npy; oct–abr y remapeo específico que mantiene dobles cultivos como Fondo |

Las tres salidas son X float16 normalizado /10000 e Y uint8. El nombre antiguo
03_generar_datasets_npz.py se conserva como wrapper, aunque ahora produce NPY;
no se crea otra conversión NPZ. Los wrappers reexportan también el worker privado.
03c/03d exponen crear_parser/main(argv=None); 03 conserva main() con configuración
fija, sin ejecutar generación al importar. Los consumidores llaman por -m.
Las ventanas siguen excluyendo el inicio en el borde size exacto, como en el
código original; mover no corrige esa política ni cambia el número de parches.
No unificar remapeos Nivel3 dic–may y 2017 oct–abr: difieren en códigos 20/21.
Ver el procedimiento y ejemplo explícito de arrays en la guía multicampaña.

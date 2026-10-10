# Leyendas por campaña

leyendas.py reúne remapeos, patrones de máscaras y validación de códigos.
No abre rasters ni importa comandos. La leyenda 2021/2022 se lee del JSON
versionado solo al pedirla. Se conservan las diferencias: crudo 0 es Maíz
en 2018/2019 y NoData en 2019/2020; la excepción 20 es exclusiva de 2020/2021.
utils.leyendas_mnc conserva los imports anteriores.
Los demás generadores históricos mantienen sus remapeos específicos hasta
una migración propia; no se reinterpretan como equivalentes automáticamente.

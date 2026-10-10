# Recomposición de dos imágenes mensuales

`mejorar_calidad_extension.py` intenta recomponer 20HNJ marzo 2019 y 20HNH
marzo 2021 con más escenas Sentinel. Verifica fuentes e imagen resultante,
conserva respaldos y rechaza reducciones de despejado o cobertura.
Entrada desde la raíz: `python -m src.mantenimiento.mejorar_calidad_extension SALIDA`.
SALIDA debe existir en el almacenamiento operativo; no inicia trabajo al importar.
`scripts/mantenimiento/recuperar_extension_este.sh` controla el lock y los
archivos de estado, ejecuta la recomposición y continúa con preparación y evaluación.
Las dos entradas anteriores de src conservan compatibilidad.

## Reconstrucción de imágenes corruptas

`recuperar_multicampana.py` audita las imágenes mensuales y reconstruye las
corruptas usando las fechas acreditadas de la selección SCL. Respalda el TIFF
original y los registros; aparta arrays X/Y y procedencia de tiles afectados
para regenerarlos. Rechaza composites legacy que requieren otro método.
Conserva bloqueos y estado, y continúa con preparación sin entrenamiento.
Entrada: `python -m src.mantenimiento.recuperar_multicampana` desde la raíz.
Importarlo no carga los comandos de preparación/composición ni repara datos.
La entrada anterior en src delega y conserva los códigos de salida.

## Cachés de lectura y reanudación del entrenamiento

`optimizar_lectura_multicampana.py` copia arrays TRAIN a las cachés existentes,
verifica identidad/hash y carga el manifiesto. Luego espera un checkpoint,
verifica el PID del entrenamiento, lo detiene y relanza desde una copia del
checkpoint completo. --solo-copiar limita el trabajo a preparar cachés.
El shell scripts/mantenimiento/optimizar_lectura_multicampana.sh EXP PID controla
el bloqueo y estado. Las rutas anteriores de src conservan compatibilidad.
Entrada Python: `python -m src.mantenimiento.optimizar_lectura_multicampana`.
Se mantienen ubicaciones acreditadas de cachés (incluida /var/tmp), reservas,
espera y opciones de entrenamiento; este traslado no mueve ni crea cachés reales.

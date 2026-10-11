# Cierre de la reorganización de src — 11/10/2026

Referencia: floryacy:~/cordoba_ia_unne. La estructura final y los procedimientos
vigentes se describen en [src/INDICE.md](../../src/INDICE.md),
[Multicampaña](../guias/multicampana.md) y [TEST vigente](../guias/test_actual.md).
Este registro conserva decisiones y evidencia de cierre; no ordena relanzar corridas.

## Resultado y alcance

Una implementación por dominio: datos, entrenamiento, inferencia, evaluación,
figuras, mantenimiento, histórico y soporte. Arquitecturas en model/, una por
archivo con bloques compartidos. Raíz de src con wrappers y compatibilidad;
shells en scripts/ por responsabilidad. Utils conserva infraestructura y APIs.

Los últimos cinco pasos acordados fueron: verificador/Manfredi; comparaciones8–10;
evaluaciónMNC/comparación2017/2018; cuatro shells anteriores; auditoría/cierre.
Quedaron implementados. No se agregó un árbol cli ni se extendió el refactor
a nuevos algoritmos, experimentos o parametrización completa de regiones/clases.

Se conservaron TEST20HNJ/20HNH, selección con VAL, exclusiones de grillas
completas TRAIN/VAL, métricas completas/independientes y canales/meses/bandas.
Los protocolos anteriores mantienen sus diferencias: parches256 y mapas512,
oct–abr y dic–may, tipos/normalización, remapeos y tratamiento de pred0.
El índice documenta esas diferencias y el contrato especial de main importado
del wrapper postprocesar. El catálogo de campañas tiene consumidores parciales;
las variantes científicas no se consolidaron por parecido.

Datos, TIFF, pesos, caches, resultados y logs conservaron sus ubicaciones en
los discos; no se recreó el dataset20JLL ni el entorno. Se preservaron cambios
e índices locales/remotos; sincronización exclusivamente de archivos ligeros.

## Validación y procedencia

Los traslados compararon funciones/cálculos por AST y resultados sintéticos
antes/después, con ayuda e imports desde otros directorios. Las pruebas cubren
wrappers/exports, rutas, firmas/procedencia, arrays, leyendas y métricas.
No se lanzaron entrenamientos, inferencias de modelos reales ni descargas.
Hubo una ejecución accidental de mantenimiento durante la auditoría CLI;
su alcance y resolución se registran abajo.

- Paso1:131 pruebas; cinco auditorías/stdout/retornos y evaluación de Manfredi iguales.
- Paso2:136 pruebas; curvas/mapa8 y cuatroPNG9 byte por byte, tensores/confusión/stdout iguales.
- Paso3:142 pruebas; cinco evaluacionesMNC completas con NPY/CSV/JSON/PNG/stdout iguales;
  variantes1718/scl y deltas iguales. Estilo local con rc_context, sin cambios de estilo al importar.
- Paso4:146 pruebas;27 escenarios shell equivalentes en eventos/stdout/estado/exitcodes,
  ocho entradas bash -n. Python/procesos/esperas simulados y locks de archivos temporales.
- Cierre: comprobación transversal de imports, CLI, consumidores, enlaces, sintaxis,
  documentación e integridad; 148 pruebas aprobadas; sintaxis de148 Python y27 shells; ayuda de32 módulos
  y31 wrappers, más despacho histórico. Imports sin efectos operativos.

Respaldos ligeros temporales de los cinco grupos en floryacy:

| Grupo | Respaldo |
|---|---|
| Verificador/Manfredi | /tmp/cordoba_paso1_que8g5l8/ |
| Comparaciones8–10 | /tmp/cordoba_paso2_87xcq6id/ |
| EvaluaciónMNC/1718 | /tmp/cordoba_paso3_i3mjdxb4/ |
| Shells anteriores | /tmp/cordoba_paso4_dmsbb4aq/ |
| Cierre | /tmp/cordoba_cierre_src_4ipsnl4q/ |

Incluyen código e índice previo y manifiestos pequeños; /tmp no es almacenamiento
permanente y su disponibilidad no está garantizada. Los grupos anteriores quedaron
incorporados por commits humanos. El agente preparó cada grupo específicamente
y no hizo commits. El temporal de continuidad era exclusivamente local: se retiró
al finalizar y su información útil quedó en el índice y este registro.

## Incidente de la auditoría CLI

La auditoría invocó por error recuperar_multicampana.py --help, que no admite parser y lanzó mantenimiento. Se detuvo todo su árbol de procesos. recuperacion.json registra 306 rasters verificados y cero reparaciones. El log registra máscaras reutilizadas y ambos datasets existentes omitidos; no se encontraron archivos recientes en esos dos datasets. Se interrumpió la validación de arrays. Se restauraron preparacion.log/pid/exitcode desde el respaldo automático. Los nuevos registros de integridad y labels_2021 se conservan; no hay copia anterior para restaurarlos. Evidencia: /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/recuperacion_20261011T095804/ejecucion_accidental.

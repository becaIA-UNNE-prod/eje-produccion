# Figuras reutilizables

- comunes.py: colores, nombres, matrices de confusión y curvas de entrenamiento.
- test_actual.py: comparación de un modelo y comparación entre modelos del TEST.
  Recibe explícitamente la ruta RGB, máscaras y predicciones.
- historico.py: comparación espacial histórica; recibe máscaras y split explícitos.

Estos módulos no importan scripts de src, no eligen checkpoints, no hacen
inferencia ni calculan métricas científicas sobre las imágenes reducidas.
Los scripts conservan wrappers/imports compatibles y coordinan las salidas.
Las gráficas de resumen ligadas a cada protocolo todavía están en los comandos.
